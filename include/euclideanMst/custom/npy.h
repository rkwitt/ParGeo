#ifndef EUCLIDEAN_MST_CUSTOM_NPY_H
#define EUCLIDEAN_MST_CUSTOM_NPY_H

#include <cassert>
#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include "cnpy.h" 

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> 
load_npy_to_eigen_row_major(const std::string& filename);

/**
 * @brief Writes 2D point coordinates to a NumPy `.npy` file.
 *
 * @tparam T Container type holding at least `num` elements.
 *
 * @param pts  Container of points.
 * @param num  Number of points to write (must be <= pts.size()).
 * @param path Output file path for the `.npy` file.
 *
 * @details
 * Extracts the first two coordinates from each point and writes them
 * to disk in NumPy `.npy` format using `cnpy::npy_save`.
 *
 * The output array has shape:
 *
 *     (num, 2)
 *
 * with row-major layout:
 *
 *     [ [x0, y0],
 *       [x1, y1],
 *       ...
 *       [x{num-1}, y{num-1}] ]
 *
 * Data type is `double` (IEEE-754 64-bit).
 *
 * Requirements on T:
 *  - Must support indexing via `pts[i]`
 *  - `pts[i]` must expose member `x`
 *  - `pts[i].x` must be indexable such that:
 *        pts[i].x[0]  // x-coordinate
 *        pts[i].x[1]  // y-coordinate
 *
 * Preconditions:
 *  - num >= 0
 *  - num <= number of elements in pts
 *
 * Exception safety:
 *  - May throw if file writing fails.
 *  - Strong guarantee (no external side effects except file creation).
 *
 * @note
 * - Existing files at `path` are overwritten (mode "w").
 * - Only the first two coordinates are written, even if points
 *   are higher dimensional.
 *
 * @warning
 * - No bounds checking is performed on `pts`.
 * - Behavior is undefined if `num` exceeds container size.
 */
template <typename T>
void write_pts_as_npy_xy(const T& pts, int num, const std::string& path) {
    assert(num >= 0 && static_cast<size_t>(num) <= pts.size());

    std::vector<double> buf;
    buf.resize((size_t)num * 2);

    for (int i = 0; i < num; ++i) {
        buf[(size_t)i * 2 + 0] = pts[i].x[0];
        buf[(size_t)i * 2 + 1] = pts[i].x[1];
    }
    // shape = (num, 2), row-major
    cnpy::npy_save(path, buf.data(), { (size_t)num, (size_t)2 }, "w");
}

#endif