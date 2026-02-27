#include <cstring>      
#include <stdexcept>

#include "euclideanMst/custom/npy.h"

[[nodiscard]]
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> 
load_npy_to_eigen_row_major(const std::string& filename) {
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
