#include <iostream>
#include <algorithm>
#include <filesystem>
#include <random>
#include <cassert>
#include <iomanip>
#include <tuple>
#include <atomic>
#include <vector>
#include <numbers>
#include <fmt/core.h>
#include <fmt/color.h>
#include <cmath>
#include <stdexcept>
#include <deque>
#include <mutex>

#include "euclideanMst/custom/koch_geometry.h"
#include "euclideanMst/custom/volumes.h"
#include "euclideanMst/custom/sphere.h"
#include "euclideanMst/custom/npy.h"
#include "euclideanMst/custom/sql.h"
#include "euclideanMst/custom/rng.h"
#include "euclideanMst/custom/config.h"
#include "euclideanMst/custom/tri_sampler.h"
#include "euclideanMst/custom/samplers.h"

#include "euclideanMst/euclideanMst.h"

#include "parlay/parallel.h"
#include "parlay/utilities.h"
#include "pargeo/point.h"
#include "pargeo/parseCommandLine.h"
#include "spatialGraph/spatialGraph.h"

#include <boost/program_options.hpp>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include "cnpy.h" 

namespace po = boost::program_options;
using namespace std;
using namespace parlay;
using namespace pargeo;

using DispatchFn = std::tuple<int, double>(*)(const MstConfig&);

/**
 * @brief Computes a numerically stable normalized MST length.
 *
 * Normalizes the minimum spanning tree (MST) length by the asymptotic scaling
 * factor typically used in geometric probability / Euclidean functionals:
 *
 *   mst_normalized = mst_length / ( n^(1 - p/d) * volume^(p/d) )
 *
 * where:
 *   - n      = numPoints
 *   - d      = intdim (intrinsic dimension)
 *   - p      = exponent parameter
 *
 * This implementation performs the normalization in log-space to improve
 * numerical stability for large n and/or small/large volume.
 *
 * @param mst_length Positive MST length (unnormalized).
 * @param numPoints  Number of points n (must be > 0).
 * @param p          Exponent parameter p (commonly nonnegative).
 * @param intdim     Intrinsic dimension d (must be > 0).
 * @param volume     Domain volume (must be > 0).
 *
 * @return The normalized MST length as a double.
 *
 * @details
 * Computes:
 *
 *   a = 1 - p/d
 *   b = p/d
 *   log_norm     = a * log(n) + b * log(volume)
 *   log_result   = log(mst_length) - log_norm
 *   result       = exp(log_result)
 *
 * Using log-space avoids overflow/underflow when n is large or when the
 * normalization factor spans many orders of magnitude.
 *
 * Preconditions:
 *  - mst_length > 0
 *  - numPoints  > 0
 *  - intdim     > 0
 *  - volume     > 0
 *
 * Exceptions:
 *  - Throws std::runtime_error if any precondition is violated.
 *
 * @note
 * - Uses `expl` (long double exp). The result is returned as `double`;
 *   the final conversion may lose precision but is typically harmless.
 */
static inline double stable_normalized_mst(
    double mst_length,
    int numPoints,
    int p,
    int intdim,
    double volume)
{
    if (mst_length <= 0.0) throw std::runtime_error("mst_length must be > 0");
    if (numPoints <= 0)    throw std::runtime_error("numPoints must be > 0");
    if (p < 1)             throw std::runtime_error("p must be >= 1");
    if (intdim <= 0)       throw std::runtime_error("intdim must be > 0");
    if (!(volume > 0.0))   throw std::runtime_error("volume must be > 0");

    const double a = 1.0 - (double)p / (double)intdim; 
    const double b = (double)p / (double)intdim;
    const double log_norm = a * std::log((double)numPoints) + b * std::log(volume);
    const double log_mst_norm = std::log(mst_length) - log_norm;
    return expl(log_mst_norm);
}


template<int dim, int p, typename PointSeq>
static inline double mst_p_sum(
    PointSeq& pts,
    const MstConfig& cfg
) {
    static_assert(p >= 0, "p must be >= 0");
    static_assert(dim >= 2, "dim must be >= 2");
    
    // call to Pargeo's MST implementation
    auto I = euclideanMst<dim>(pts);
    auto S = pts.data();

    double sum = 0.0;
    double c = 0.0;

    for (auto e : I) {
        double d = S[e.u].dist(S[e.v]);

        if (cfg.use_sphere_geodesic) {
            d = emstExtension::custom::sphere_geodesic_from_chord_atan2(d);
        }

        const double pd = std::pow(d, p);

        const double t = sum + pd;
        if (std::abs(sum) >= std::abs(pd)) c += (sum - t) + pd;
        else                               c += (pd - t) + sum;
        sum = t;
    }

    return sum + c;
}

template<int dim>
void fill_points_from_config(
    parlay::sequence<pargeo::point<dim>>& pts,
    const MstConfig& cfg
) {
    if (!cfg.input_file.empty()) {
        Eigen::MatrixXd X = load_npy_to_eigen_row_major(cfg.input_file);
        if (X.cols() != dim) throw std::runtime_error("Dimension mismatch: file has cols != dim");

        pts = parlay::sequence<pargeo::point<dim>>(static_cast<size_t>(X.rows()));

        for (int i = 0; i < X.rows(); ++i) {
            for (int k = 0; k < dim; ++k) {
                pts[i].x[k] = X(i, k);
            }
        }
        return;
    }

    if (cfg.num_points <= 0) {
        throw std::runtime_error("num_points must be > 0 (unless input_file is provided)");
    }

    pts = parlay::sequence<pargeo::point<dim>>(static_cast<size_t>(cfg.num_points));

    switch (cfg.shape) {
        case Shape::Cube:
            fill_from_uniform_in_unitbox<dim, decltype(pts), double>(pts, cfg.num_points);
            return;
        case Shape::Ball:
            fill_from_uniform_in_unitball<dim, decltype(pts), double>(pts, cfg.num_points);
            return;
        case Shape::Sphere:
            fill_from_uniform_on_unitsphere<dim, decltype(pts), double>(pts, cfg.num_points);
            return;
        case Shape::Koch:
            if (cfg.koch_depth < 0) throw std::runtime_error("koch_depth must be >= 0");
            fill_from_koch_snowflake_2d_cdt(pts, cfg);
            return;
        case Shape::Grassmann:
            if (cfg.gr_n <= 0) throw std::runtime_error("shape=grassmann requires gr_n > 0");
            if (cfg.gr_k < 0 || cfg.gr_k > cfg.gr_n) throw std::runtime_error("shape=grassmann requires 0 <= gr_k <= gr_n");
            if (dim != cfg.gr_n * cfg.gr_n) throw std::runtime_error("For grassmann projection embedding require dim == gr_n*gr_n");
            fill_from_grassmann_svd_projection<dim, decltype(pts), double>(
                pts, cfg.num_points, cfg.gr_n, cfg.gr_k
            );
            return;
    }
    throw std::runtime_error("Unsupported shape in config");
}

template<int dim, int p>
static std::tuple<int, double> run_mst(const MstConfig& cfg) {
    parlay::sequence<pargeo::point<dim>> pts;
    fill_points_from_config<dim>(pts, cfg);
    const double sum = mst_p_sum<dim, p>(pts, cfg);
    return { static_cast<int>(pts.size()), sum };
}

template<int dim, int p>
static std::tuple<int, double> dispatch_helper(const MstConfig& cfg) {
    return run_mst<dim, p>(cfg);
}

static std::tuple<int, double> dispatch(int dim, int p, const MstConfig& cfg) {
    // Validate supported template instantiations
    if (dim < 2 || dim > 16 || p < 1 || p > 5) {
        throw std::invalid_argument("Unsupported dimension or p value");
    }

    static const DispatchFn dispatchTable[15][5] = {
        {&dispatch_helper<2, 1>,  &dispatch_helper<2, 2>,  &dispatch_helper<2, 3>,  &dispatch_helper<2, 4>,  &dispatch_helper<2, 5>},
        {&dispatch_helper<3, 1>,  &dispatch_helper<3, 2>,  &dispatch_helper<3, 3>,  &dispatch_helper<3, 4>,  &dispatch_helper<3, 5>},
        {&dispatch_helper<4, 1>,  &dispatch_helper<4, 2>,  &dispatch_helper<4, 3>,  &dispatch_helper<4, 4>,  &dispatch_helper<4, 5>},
        {&dispatch_helper<5, 1>,  &dispatch_helper<5, 2>,  &dispatch_helper<5, 3>,  &dispatch_helper<5, 4>,  &dispatch_helper<5, 5>},
        {&dispatch_helper<6, 1>,  &dispatch_helper<6, 2>,  &dispatch_helper<6, 3>,  &dispatch_helper<6, 4>,  &dispatch_helper<6, 5>},
        {&dispatch_helper<7, 1>,  &dispatch_helper<7, 2>,  &dispatch_helper<7, 3>,  &dispatch_helper<7, 4>,  &dispatch_helper<7, 5>},
        {&dispatch_helper<8, 1>,  &dispatch_helper<8, 2>,  &dispatch_helper<8, 3>,  &dispatch_helper<8, 4>,  &dispatch_helper<8, 5>},
        {&dispatch_helper<9, 1>,  &dispatch_helper<9, 2>,  &dispatch_helper<9, 3>,  &dispatch_helper<9, 4>,  &dispatch_helper<9, 5>},
        {&dispatch_helper<10, 1>, &dispatch_helper<10, 2>, &dispatch_helper<10, 3>, &dispatch_helper<10, 4>, &dispatch_helper<10, 5>},
        {&dispatch_helper<11, 1>, &dispatch_helper<11, 2>, &dispatch_helper<11, 3>, &dispatch_helper<11, 4>, &dispatch_helper<11, 5>},
        {&dispatch_helper<12, 1>, &dispatch_helper<12, 2>, &dispatch_helper<12, 3>, &dispatch_helper<12, 4>, &dispatch_helper<12, 5>},
        {&dispatch_helper<13, 1>, &dispatch_helper<13, 2>, &dispatch_helper<13, 3>, &dispatch_helper<13, 4>, &dispatch_helper<13, 5>},
        {&dispatch_helper<14, 1>, &dispatch_helper<14, 2>, &dispatch_helper<14, 3>, &dispatch_helper<14, 4>, &dispatch_helper<14, 5>},
        {&dispatch_helper<15, 1>, &dispatch_helper<15, 2>, &dispatch_helper<15, 3>, &dispatch_helper<15, 4>, &dispatch_helper<15, 5>},
        {&dispatch_helper<16, 1>, &dispatch_helper<16, 2>, &dispatch_helper<16, 3>, &dispatch_helper<16, 4>, &dispatch_helper<16, 5>}
    };

    return dispatchTable[dim - 2][p - 1](cfg);
}

int main(int argc, char* argv[]) 
{
    std::string input_file;           // input file (binary numpy tensor)
    std::string db_file;              // SQLite3 database file
    std::string shape;                // sample from ball or cube
    int num_points = -1;              // is set on cmdline
    int dim = -1;                     // is set on cmdline
    int intdim = -1;                  // will be computed
    int p = 1;                        // default p
    int gr_n = -1;                    // ambient dim. of Gr(n,k)
    int gr_k = -1;                    // subspace dim of Gr(n,k)
    int koch_depth = 5;               // default depth of Koch snowflake
    bool use_sphere_geodesic = false; // convert Eucl. distance to geodesic distance on sphere (for shape=sphere)
    double volume = std::numeric_limits<double>::quiet_NaN(); // default to NaN, will be computed
    bool use_triangulation_file = false;
    bool write_triangulation_file = true;
    std::string triangulation_file;
    
    try {
        po::options_description desc("Options");
        desc.add_options()
            ("help", "produce help message")
            ("dim", po::value<int>(&dim)->required(), "dimensionality of input vectors (R^d).")
            ("num_points", po::value<int>(&num_points)->required(), "Number of points to sample.")
            ("shape", po::value<std::string>(&shape)->required(), "Sample from cube|ball|sphere|grassmann|koch.")
            ("db_file", po::value<std::string>(&db_file)->required(), "SQLite3 database file.")
            ("p", po::value<int>(&p), "Power of edge lengths.")
            ("input_file", po::value<std::string>(&input_file), "Numpy matrix input file.")
            ("gr_n", po::value<int>(&gr_n), "Grassmann ambient dimension n (R^n).")
            ("gr_k", po::value<int>(&gr_k), "Grassmann subspace dimension k.")
            ("use_sphere_geodesic", po::bool_switch(&use_sphere_geodesic)->default_value(false), "Convert dist. to chord length.")
            ("koch_depth", po::value<int>(&koch_depth)->default_value(5), "Recursion depth for Koch snowflake.")
            ("use_triangulation_file", po::bool_switch(&use_triangulation_file)->default_value(false), "Write and use triangulation file.")
            ("triangulation_file", po::value<std::string>(&triangulation_file)->default_value(""), "Path to triangulation file.");

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

    std::cout << std::scientific << std::setprecision(12);

    // MST configuration struct to pass to dispatch
    MstConfig cfg;
    cfg.num_points = num_points;
    cfg.input_file = input_file;
    cfg.shape = parse_shape(shape);
    cfg.gr_n = gr_n;
    cfg.gr_k = gr_k;
    cfg.use_sphere_geodesic = use_sphere_geodesic;
    cfg.koch_depth = koch_depth;
    cfg.use_triangulation_file = use_triangulation_file;
    cfg.triangulation_file = triangulation_file;

    try {
        // Dispatch MST computation
        std::tuple<int, double> result = dispatch(dim, p, cfg);

        num_points = std::get<0>(result);
        double mst_length = std::get<1>(result);

        // Compute volume and intrinsic dimension for normalization based on shape.
        if (cfg.shape == Shape::Grassmann) {
            if (cfg.gr_n <= 0 || cfg.gr_k < 0) throw std::runtime_error("--shape grassmann requires --gr_n and --gr_k");
            if (cfg.gr_n <= cfg.gr_k) throw std::runtime_error("--shape grassmann requires gr_n > gr_k");
            if (!std::isfinite(volume)) volume = grassmann_volume(cfg.gr_n, cfg.gr_k);
            intdim = cfg.gr_k * (cfg.gr_n - cfg.gr_k);
        } else if (cfg.shape == Shape::Sphere) {
            if (dim < 2) throw std::runtime_error("Sphere requires dim >= 2");
            if (!std::isfinite(volume)) volume = emstExtension::custom::sphere_surface_area(dim - 1);
            intdim = dim - 1;
        } else if (cfg.shape == Shape::Ball) {
            if (!std::isfinite(volume)) volume = unit_ball_volume(dim);
            intdim = dim;
        } else if (cfg.shape == Shape::Koch) {
            if (dim != 2) throw std::runtime_error("--shape koch requires --dim 2");
            if (cfg.koch_depth < 0) throw std::runtime_error("--koch_depth must be >= 0");
            auto poly = koch_polygon(cfg.koch_depth);
            volume = polygon_area(poly);
            intdim = 2;
        } else if (cfg.shape == Shape::Cube) {
            if (!std::isfinite(volume)) volume = 1.0;
            intdim = dim;
        }
        
        double mst_length_normalized = stable_normalized_mst(mst_length, num_points, p, intdim, volume);
        fmt::print(
            "| n={:>8d} | volume={:>20.16f} | dim={:>2d} | intdim={:>2d} | L_n={:>22.12f} | L_n (norm)={:>22.16f}\n",
            num_points, volume, dim, intdim, mst_length, 
            fmt::styled(mst_length_normalized,fmt::fg(fmt::color::green) | fmt::emphasis::bold)
        );

        write_to_database(db_file, num_points, mst_length, mst_length_normalized);

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return -1;
    }
    return 0;
}
