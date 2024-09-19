#include <iostream>
#include <algorithm>
#include <random>
#include <cassert>
#include <cmath>
#include <tuple>
#include "euclideanMst/euclideanMst.h"

#include "parlay/parallel.h"
#include "parlay/utilities.h"
#include "pargeo/point.h"
#include "pargeo/parseCommandLine.h"
#include "spatialGraph/spatialGraph.h"

#include <sqlite3.h>
#include <boost/program_options.hpp>

#include <Eigen/Dense>
#include "cnpy.h" 

namespace po = boost::program_options;
using namespace std;
using namespace parlay;
using namespace pargeo;

// Function to execute an SQL command
void executeSQL(sqlite3* db, const std::string& sql) {
    char* errmsg;
    int rc = sqlite3_exec(db, sql.c_str(), 0, 0, &errmsg);

    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << errmsg << std::endl;
        sqlite3_free(errmsg);
    }
}

// Function to write data to the SQLite database
void writeToDatabase(const std::string& dbFilename, int num_points, double mst_length, double normalized_mst_length) {
    sqlite3* db;
    int rc = sqlite3_open(dbFilename.c_str(), &db);

    if (rc) {
        std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
        return;
    }

    // Set a busy timeout of 5 seconds (5000 milliseconds)
    sqlite3_busy_timeout(db, 5000);

    // Create a table if it doesn't exist
    std::string createTableSQL = "CREATE TABLE IF NOT EXISTS data ("
                                 "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                                 "num_points INTEGER, "
                                 "mst_length REAL, "
                                 "normalized_mst_length REAL);";
    executeSQL(db, createTableSQL);

    // Begin transaction to avoid locking issues during multiple operations
    executeSQL(db, "BEGIN TRANSACTION;");

    // Prepare SQL insert statement
    std::string insertSQL = "INSERT INTO data (num_points, mst_length, normalized_mst_length) VALUES (?, ?, ?);";
    sqlite3_stmt* stmt;
    rc = sqlite3_prepare_v2(db, insertSQL.c_str(), -1, &stmt, 0);

    if (rc != SQLITE_OK) {
        std::cerr << "Can't prepare SQL statement: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return;
    }

    // Bind values to the SQL statement
    sqlite3_bind_int(stmt, 1, num_points);
    sqlite3_bind_double(stmt, 2, mst_length);
    sqlite3_bind_double(stmt, 3, normalized_mst_length);

    // Execute the SQL statement with retry logic if the database is locked
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        std::cerr << "Execution failed: " << sqlite3_errmsg(db) << std::endl;
    }

    // Finalize the statement to release resources
    sqlite3_finalize(stmt);

    // Commit the transaction
    executeSQL(db, "COMMIT;");

    // Close the database connection
    sqlite3_close(db);
}


Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> 
loadNpyToEigenRowMajor(const std::string& filename) {
    cnpy::NpyArray arr = cnpy::npy_load(filename);

    if (arr.word_size != sizeof(double)) {
        throw std::runtime_error("Data type mismatch! Expected double.");
    }

    std::vector<size_t> shape = arr.shape;
    if (shape.size() != 2) {
        throw std::runtime_error("Expected a 2D matrix.");
    }

    size_t rows = shape[0];
    size_t cols = shape[1];

    double* data = arr.data<double>();

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>  matrix(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix(i, j) = data[i * cols + j];  // Row-major access
        }
    }
    return matrix;
}


template<int dim, typename T, typename internal>
void fill_from_uniform_in_unitbox(T &pts, int num, unsigned seed) {
  random_device rd;  
  mt19937 gen(rd()); 
  uniform_real_distribution<double> distr(0.0, 1.0);
  for (int i = 0; i < num; ++i) {
    for (int k = 0; k < dim; k++) {
      double tmp = distr(gen);
      pts[i].x[k] = (internal)tmp;
    }
  } 
}


template<int dim, typename T, typename internal>
void fill_from_uniform_in_unitball(T &pts, int num, unsigned seed) {
  random_device rd;  
  mt19937 gen(rd()); 
  normal_distribution<internal> normalDist(0.0, 1.0);
  uniform_real_distribution<internal> uniformDist(0.0, 1.0);

  for (int i = 0; i < num; ++i) {
    std::vector<internal> v(dim);
    internal normSq = 0.0;
    for (int k = 0; k < dim; ++k) {
        v[k] = normalDist(gen);
        normSq += v[k] * v[k];
    }
    internal radius = pow(uniformDist(gen), 1.0 / dim);
    for (int k = 0; k < dim; ++k) {
        v[k] *= radius / sqrt(normSq);
        pts[i].x[k] = v[k];
    }
  }
}


template<int dim, int p>
std::tuple<int, double> callEuclideanMst(int num, std::string shape) {
  double emst_len = 0.0;
  parlay::sequence<pargeo::point<dim>> pts(num);
  
  if (shape == "cube") {
    fill_from_uniform_in_unitbox<dim, parlay::sequence<pargeo::point<dim>>, double>(pts, num,1);
  } else if (shape == "ball") {
    fill_from_uniform_in_unitball<dim, parlay::sequence<pargeo::point<dim>>, double>(pts, num,1);
  }
  auto I = euclideanMst<dim>(pts);
  auto S = pts.data();
  
  double sum = 0.0;
  double compensation = 0.0;
  double dist = 0.0;
  for (auto e: I) {
    dist = S[e.u].dist(S[e.v]);
    double pdist = pow(dist,p); // uses that power weighted spanning tree has the same edges
    double y = pdist - compensation;
    double t = sum + y;
    compensation = (t - sum) - y;
    sum = t;
  }
  return std::make_tuple(num, sum);
}


template<int dim, int p>
std::tuple<int, double> callEuclideanMstFromFile(std::string inputFile) {
  int rows;
  int cols;
  
  Eigen::MatrixXd X = loadNpyToEigenRowMajor(inputFile);
  rows = X.rows();
  cols = X.cols();

  if (dim != cols) {
    throw std::runtime_error("Dimension mismatch!");
  }

  parlay::sequence<pargeo::point<dim>> pts(rows);
  for (int i = 0; i < rows; ++i) {
    for (int k = 0; k < cols; ++k) {
        pts[i].x[k] = X(i,k);
    }
  }
  
  auto I = euclideanMst<dim>(pts);
  auto S = pts.data();
  
  double sum = 0.0;
  double compensation = 0.0;
  double dist = 0.0;
  for (auto e: I) {
    dist = S[e.u].dist(S[e.v]);
    double pdist = pow(dist,p); // uses that power weighted spanning tree has the same edges
    double y = pdist - compensation;
    double t = sum + y;
    compensation = (t - sum) - y;
    sum = t;
  }
  return std::make_tuple(rows, sum);
}


template <int dim, int p>
std::tuple<int, double> dispatchEuclideanMst(int numPoints, const std::string& shape, const std::string& inputFile) {
    std::tuple<int, double> result;
    if (!inputFile.empty()) {
        result = callEuclideanMstFromFile<dim,p>(inputFile);
    } else {
        result = callEuclideanMst<dim,p>(numPoints, shape);
    }
    return result;
}


using DispatchFn = std::tuple<int, double>(*)(int, const std::string&, const std::string&);

template<int dim, int p>
std::tuple<int, double> dispatchHelper(int numPoints, const std::string& shape, const std::string& inputFile) {
    return dispatchEuclideanMst<dim, p>(numPoints, shape, inputFile);
}

std::tuple<int, double> dispatch(int dim, int p, int numPoints, const std::string& shape, const std::string& inputFile) {
    if (inputFile.empty()) {
        if (shape.empty()) throw std::runtime_error("No shape given!");
        if (numPoints < 0) throw std::runtime_error("Number of points not >0!");
        if (dim < 2 || dim > 9) throw std::runtime_error("Dimension needs to be in {2,...9}");
    }

    // Table of function pointers for p = 1 to 5 and dim = 2 to 9
    static const DispatchFn dispatchTable[8][5] = {
        {&dispatchHelper<2, 1>, &dispatchHelper<2, 2>, &dispatchHelper<2, 3>, &dispatchHelper<2, 4>, &dispatchHelper<2, 5>},
        {&dispatchHelper<3, 1>, &dispatchHelper<3, 2>, &dispatchHelper<3, 3>, &dispatchHelper<3, 4>, &dispatchHelper<3, 5>},
        {&dispatchHelper<4, 1>, &dispatchHelper<4, 2>, &dispatchHelper<4, 3>, &dispatchHelper<4, 4>, &dispatchHelper<4, 5>},
        {&dispatchHelper<5, 1>, &dispatchHelper<5, 2>, &dispatchHelper<5, 3>, &dispatchHelper<5, 4>, &dispatchHelper<5, 5>},
        {&dispatchHelper<6, 1>, &dispatchHelper<6, 2>, &dispatchHelper<6, 3>, &dispatchHelper<6, 4>, &dispatchHelper<6, 5>},
        {&dispatchHelper<7, 1>, &dispatchHelper<7, 2>, &dispatchHelper<7, 3>, &dispatchHelper<7, 4>, &dispatchHelper<7, 5>},
        {&dispatchHelper<8, 1>, &dispatchHelper<8, 2>, &dispatchHelper<8, 3>, &dispatchHelper<8, 4>, &dispatchHelper<8, 5>},
        {&dispatchHelper<9, 1>, &dispatchHelper<9, 2>, &dispatchHelper<9, 3>, &dispatchHelper<9, 4>, &dispatchHelper<9, 5>}
    };

    // Ensure p and dim are within valid ranges
    if (dim < 2 || dim > 9 || p < 1 || p > 5) {
        throw std::invalid_argument("Unsupported dimension or p value");
    }

    return dispatchTable[dim - 2][p - 1](numPoints, shape, inputFile);
}


int main(int argc, char* argv[]) 
{
    std::string inputFile;      // input file (binary numpy tensor)
    std::string dbFile;         // SQLite3 database file
    std::string shape;          // sample from ball or cube
    double volume = 1.0;
    int numPoints = -1;        
    int dim = -1;
    int intdim = -1;
    int p = 1;

    try {
        po::options_description desc("Options");
        desc.add_options()
            ("help,h", "produce help message")
            ("dim,d", po::value<int>(&dim)->required(), "dimensionality of input vectors (R^d).")
            ("intDim,d", po::value<int>(&intdim), "intrinsic_dimensionality of the shape")
            ("p,p", po::value<int>(&p), "power of edge lengths")
            ("numPoints,n", po::value<int>(&numPoints), "Number of points to sample.")
            ("volume,v", po::value<double>(&volume)->required(), "Volume for normalization.")
            ("shape,f", po::value<std::string>(&shape), "Sample from cube or ball.")
            ("inputFile,i", po::value<std::string>(&inputFile), "Numpy matrix input file.")
            ("dbFile,d", po::value<std::string>(&dbFile)->required(), "SQLite3 database file.");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }
        po::notify(vm);
    } catch (const po::error &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

  try {
    std::tuple<int, double> result = dispatch(dim, p, numPoints, shape, inputFile);
    
    numPoints = std::get<0>(result);
    double mst_length = std::get<1>(result);
    if (intdim == -1) intdim = dim;
    double normalization = pow((double)numPoints,1.-1.*p/intdim) * pow(volume,1.*p/intdim);
    double mst_length_normalized = mst_length /normalization;
    std::cout << numPoints << " | MST length: " << mst_length << " | " << "(Normalized) MST length: " << mst_length_normalized << std::endl;
    writeToDatabase(dbFile, numPoints, mst_length, mst_length_normalized);

  } catch (std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return -1;
  }
  return 0;
}
