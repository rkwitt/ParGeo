#ifndef EUCLIDEAN_MST_CUSTOM_SAMPLERS_H
#define EUCLIDEAN_MST_CUSTOM_SAMPLERS_H

#include <cmath>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <vector>
#include <type_traits>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "parlay/parallel.h"

#include "euclideanMst/custom/rng.h"

/*
 * @brief Fills a point container with i.i.d. uniform samples in the unit box [0,1)^dim.
 *
 * @tparam dim       Compile-time spatial dimension (must be > 0).
 * @tparam T         Container type holding at least `num` elements.
 * @tparam internal  Scalar type used for storing coordinates.
 *
 * @param pts  Container of points to be written to.
 * @param num  Number of points to generate (must be <= pts.size()).
 *
 * @details
 * For each index i in [0, num), this routine draws `dim` independent
 * uniform random numbers U ~ Uniform([0,1)) and assigns:
 *
 *   pts[i].x[k] = U_k,   k = 0, ..., dim-1
 *
 * The loop over points is parallelized using `parlay::parallel_for`.
 * Each worker thread uses its own thread-local RNG instance (`tls_rng()`),
 * ensuring:
 *
 *  - No shared RNG state between threads.
 *  - No synchronization overhead.
 *  - Independent streams per thread.
 *
 * Distribution:
 *  - The resulting points are uniformly distributed in the axis-aligned
 *    unit hypercube [0,1)^dim.
 *
 * Requirements on T:
 *  - Must support indexing: pts[i]
 *  - Each element must expose coordinate storage:
 *        pts[i].x[k]
 *    for k = 0 .. dim-1
 *
 * Preconditions:
 *  - dim > 0
 *  - num >= 0
 *  - pts contains at least `num` elements
 *
 * Thread safety:
 *  - Safe for parallel execution provided each index i refers to
 *    disjoint storage in `pts`.
 *
 * Numerical notes:
 *  - Values are drawn using `next_double01()` and cast to `internal`.
 *  - If `internal` is float or lower precision, the distribution is
 *    quantized accordingly.
 *
 * Complexity:
 *  - Work:  O(num * dim)
 *  - Span:  O((num * dim) / P) under ideal parallel scheduling
 *
 * Dependencies:
 *  - Requires Parlay library (`parlay::parallel_for`).
 *  - Requires thread-local RNG via `tls_rng()`.
 */
template <int dim, typename T, typename internal>
void fill_from_uniform_in_unitbox(T &pts, int num) {
    static_assert(dim > 0, "dim must be positive");
    parlay::parallel_for(0, num, [&](int i) {
        auto &rng = tls_rng();
        for (int k = 0; k < dim; ++k) {
            pts[i].x[k] = static_cast<internal>(rng.next_double01());
        }
    });
}

/**
 * @brief Fills a point container with i.i.d. uniform samples in the unit ball B^dim.
 *
 * @tparam dim       Compile-time spatial dimension (must be >= 1).
 * @tparam T         Container type holding at least `num` elements.
 * @tparam internal  Scalar type used for storing coordinates (e.g., float/double).
 *
 * @param pts  Container of points to be written to.
 * @param num  Number of points to generate (must be <= pts.size()).
 *
 * @details
 * Generates points uniformly (with respect to Lebesgue measure) inside the
 * unit ball:
 *
 *   B^dim = { x ∈ R^dim : ||x||_2 <= 1 }.
 *
 * Sampling method (standard and exact):
 *  1. Sample a direction uniformly on the unit sphere S^{dim-1} by drawing
 *     a Gaussian vector g ~ N(0, I) and normalizing:
 *
 *       u = g / ||g||.
 *
 *     This yields u uniformly distributed on the sphere.
 *
 *  2. Sample an independent radius R with CDF F(r) = r^dim on [0, 1] via:
 *
 *       R = U^{1/dim},  where U ~ Uniform([0,1)).
 *
 *  3. Set the point:
 *
 *       x = R * u.
 *
 * Parallelism:
 *  - The outer loop is parallelized with `parlay::parallel_for`.
 *  - Each worker uses a thread-local RNG (`tls_rng()`), avoiding shared state.
 *
 * Requirements on T:
 *  - Must support indexing: pts[i]
 *  - Each element must provide coordinate access: pts[i].x[k] for k=0..dim-1
 *
 * Preconditions:
 *  - dim >= 1
 *  - num >= 0
 *  - pts contains at least `num` elements
 *
 * Thread safety:
 *  - Safe for parallel execution provided each i writes to disjoint storage.
 *
 * Numerical notes / corner cases:
 *  - The Gaussian normalization step requires ||g|| > 0. In exact arithmetic,
 *    P(||g||=0)=0, and with floating-point this is effectively impossible unless
 *    the RNG or implementation is pathological.
 *  - Uses `std::fma` to accumulate norm2 with reduced rounding error (if
 *    hardware supports it).
 *  - Uses `std::pow(U, 1/dim)`; for integer dim this is correct, though for
 *    performance you could special-case dim=2,3,4, etc.
 *
 * Complexity:
 *  - Work:  O(num * dim)
 *  - Span:  O((num * dim) / P) under ideal scheduling
 *
 * Dependencies:
 *  - Requires Parlay library (`parlay::parallel_for`).
 *  - Requires `normal01` and `tls_rng()` for random numbers.
 */
template<int dim, typename T, typename internal>
void fill_from_uniform_in_unitball(T &pts, int num) {
    static_assert(dim >= 1, "dim must be >= 1");

    parlay::parallel_for(0, num, [&](int i) {
        auto& rng = tls_rng();

        // sample direction by normalizing N(0,1)^dim
        internal v[dim];
        internal norm2 = 0;

        for (int k = 0; k < dim; ++k) {
            internal z = static_cast<internal>(normal01(rng));
            v[k] = z;
            norm2 = std::fma(z, z, norm2);
        }

        internal inv_norm = internal(1) / std::sqrt(norm2);

        // radius for uniform ball: U^(1/d)
        internal r = static_cast<internal>(std::pow(rng.next_double01(), 1.0 / double(dim)));

        for (int k = 0; k < dim; ++k) {
            pts[i].x[k] = v[k] * inv_norm * r;
        }
    });
}

/**
 * @brief Samples k-dimensional subspaces in R^n and stores them as flattened projection matrices.
 *
 * @tparam dim       Compile-time feature dimension. Must equal n*n for projection matrices.
 * @tparam T         Output container type holding at least `num` elements.
 * @tparam internal  Floating-point scalar type used for computation/storage (float/double/long double).
 *
 * @param pts Output container. On return, `pts[i].x[0..n*n-1]` contains a flattened
 *            (scaled) orthogonal projection matrix representing the sampled subspace.
 * @param num Number of samples to generate (must be > 0).
 * @param n   Ambient dimension (R^n).
 * @param k   Subspace dimension (0 <= k <= n). Each sample is a point in Gr(n, k).
 *
 * @details
 * This routine generates i.i.d. samples from the Grassmann manifold Gr(n, k)
 * (k-dimensional subspaces of R^n) by sampling a random Gaussian matrix and
 * extracting an orthonormal basis for its column space.
 *
 * Sampling procedure (standard construction):
 *  1. Draw a random matrix A ∈ R^{n×k} with i.i.d. entries:
 *
 *       A_{rc} ~ N(0, 1).
 *
 *  2. Compute an orthonormal basis Q ∈ R^{n×k} for the column space of A.
 *     This is done via the thin-U factor from an SVD:
 *
 *       A = U Σ V^T,  Q := U_{(:, 0..k-1)}.
 *
 *     The resulting subspace span(Q) is distributed according to the
 *     rotationally invariant (Haar / uniform) measure on Gr(n, k).
 *
 *  3. Form the orthogonal projection matrix onto the subspace:
 *
 *       P = Q Q^T  ∈ R^{n×n}.
 *
 *     This representation is basis-invariant (depends only on the subspace, not
 *     on the particular orthonormal basis Q).
 *
 *  4. Symmetrize P to reduce numerical asymmetry:
 *
 *       P ← (P + P^T)/2.
 *
 *  5. Flatten P into a feature vector in row-major order:
 *
 *       pts[i].x[ r*n + c ] = scale * P(r,c),
 *
 *     where `scale = 1/sqrt(2)`. 
 *
 * Parallelism:
 *  - This implementation is currently sequential over i = 0..num-1.
 *    (Eigen SVD objects are often not thread-safe to share across threads.)
 *    If you parallelize over i, create per-thread SVD objects and per-thread
 *    matrices or allocate them inside the loop.
 *
 * Requirements on T:
 *  - Must support indexing: pts[i]
 *  - Each element must support coordinate writes: pts[i].x[idx]
 *  - Coordinate storage must have capacity at least n*n (i.e., dim == n*n).
 *
 * Preconditions:
 *  - num > 0
 *  - n > 0
 *  - 0 <= k <= n
 *  - dim == n*n
 *
 * Exceptions:
 *  - Throws std::runtime_error if num <= 0 or if dim != n*n.
 *  - Eigen operations may throw (or assert) on invalid sizes or allocation failure.
 *
 * Dependencies:
 *  - Requires Eigen (Matrix, BDCSVD).
 *  - Requires RNG utilities: `tls_rng()` and `normal01(...)`.
 */
template<int dim, typename T, typename internal>
void fill_from_grassmann_svd_projection(
    T &pts,
    int num,
    int n,   // ambient R^n
    int k    // k-dimensional subspaces
) {
    static_assert(std::is_floating_point_v<internal>,
        "internal must be a floating-point type");

    if (num <= 0) 
        throw std::runtime_error("num must be > 0");

    // Projection representation: dim must be n*n
    const int d_expected = n * n;
    if (d_expected != dim) {
        throw std::runtime_error("Dimension mismatch: for projection matrices, dim must equal n*n");
    }

    using Mat = Eigen::Matrix<internal, Eigen::Dynamic, Eigen::Dynamic>;
    Mat A(n, k);
    Mat Q(n, k);
    Mat P(n, n);

    Eigen::BDCSVD<Mat> svd;
    svd.compute(A, Eigen::ComputeThinU); // initialize sizes

    const internal scale = (internal)(1.0 / std::sqrt(2.0));

    for (int i = 0; i < num; ++i) {
        auto& rng = tls_rng();

        // A ~ N(0,1)^(n x k)
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < k; ++c)
                A(r, c) = static_cast<internal>(normal01(rng));

        // Q = thin-U (n x k), orthonormal columns
        svd.compute(A, Eigen::ComputeThinU);
        Q.noalias() = svd.matrixU().leftCols(k);

        // P = Q Q^T (basis-invariant)
        P.noalias() = Q * Q.transpose();

        // enforce symmetry (numerical)
        P = (P + P.transpose()) * internal(0.5);

        // Flatten P row-major into pts[i].x[0..n*n-1]
        int idx = 0;
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                pts[i].x[idx++] = scale * static_cast<internal>(P(r, c));
            }
        }
    }
}

/**
 * @brief Fills a container with i.i.d. samples uniformly distributed on the unit sphere S^{dim-1}.
 *
 * @tparam dim       Compile-time ambient dimension (must be >= 1).
 * @tparam T         Container type holding at least `num` elements.
 * @tparam internal  Floating-point scalar type used for computation/storage.
 *
 * @param pts Output container. On return, pts[i].x[0..dim-1] contains a unit vector.
 * @param num Number of samples to generate. If num <= 0, the function returns immediately.
 *
 * @details
 * Generates samples uniformly distributed on the unit sphere
 *
 *   S^{dim-1} = { x ∈ R^dim : ||x||_2 = 1 }.
 *
 * Sampling method (standard Gaussian normalization):
 *
 *  1. Draw a vector g ∈ R^dim with independent components:
 *
 *        g_k ~ N(0, 1).
 *
 *  2. Normalize:
 *
 *        x = g / ||g||_2.
 *
 * It is a classical result that this procedure yields the unique
 * rotationally invariant (Haar) probability measure on the sphere.
 *
 * Requirements on T:
 *  - Must support indexing: pts[i]
 *  - Each element must support coordinate writes: pts[i].x[k], k=0..dim-1
 *
 * Preconditions:
 *  - dim >= 1
 *  - pts contains at least `num` writable elements
 *
 * Thread safety:
 *  - This implementation is sequential.
 *  - RNG uses thread-local state (`tls_rng()`), so parallelization over i
 *    is safe provided each iteration writes to disjoint storage.
 *
 * Dependencies:
 *  - Requires `normal01(rng)` for Gaussian sampling.
 *  - Requires `tls_rng()` for thread-local random generator.
 */
template<int dim, typename T, typename internal>
void fill_from_uniform_on_unitsphere(T &pts, int num)
{
    static_assert(std::is_floating_point_v<internal>,
        "internal must be floating-point");

    if (num <= 0) return;

    for (int i = 0; i < num; ++i) {
        auto& rng = tls_rng();
        internal norm2 = internal(0);
        for (int k = 0; k < dim; ++k) {
            internal v = static_cast<internal>(normal01(rng));
            pts[i].x[k] = v;
            norm2 = std::fma(v, v, norm2);
        }

        internal inv_norm = internal(1) / std::sqrt(norm2);
        for (int k = 0; k < dim; ++k)
            pts[i].x[k] *= inv_norm;
    }
}

#endif 