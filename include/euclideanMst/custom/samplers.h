#include "parlay/parallel.h"

#include "euclideanMst/custom/config.h"
#include "euclideanMst/custom/rng.h"

inline bool file_exists(const std::string& path) {
    return !path.empty() && std::filesystem::exists(path);
}

inline void write_sampler_binary(const KochTriSampler& S, const std::string& path) {
    if (path.empty()) throw std::runtime_error("write_sampler_binary: empty path");
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    // write to temp then rename (avoid partial files)
    std::string tmp = path + ".tmp." + std::to_string(::getpid());

    std::ofstream out(tmp, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open file for write: " + tmp);

    uint32_t magic = KOCHTRI_MAGIC;
    uint32_t ver   = KOCHTRI_VER;
    uint64_t ntri  = (uint64_t)S.tris.size();

    out.write((char*)&magic, sizeof(magic));
    out.write((char*)&ver,   sizeof(ver));
    out.write((char*)&ntri,  sizeof(ntri));
    out.write((char*)&S.total_area, sizeof(S.total_area));

    // store Tri2 as raw doubles (portable enough across same arch; see note below)
    for (const auto& t : S.tris) {
        out.write((char*)&t.ax, sizeof(double));
        out.write((char*)&t.ay, sizeof(double));
        out.write((char*)&t.bx, sizeof(double));
        out.write((char*)&t.by, sizeof(double));
        out.write((char*)&t.cx, sizeof(double));
        out.write((char*)&t.cy, sizeof(double));
        out.write((char*)&t.area, sizeof(double));
    }

    out.close();
    if (!out) throw std::runtime_error("Write failed: " + tmp);

    std::filesystem::rename(tmp, path);
}

/**
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

template<typename T>
void fill_from_koch_snowflake_2d_cdt(T& pts, const MstConfig& cfg) {
    if (cfg.num_points <= 0) return;
    if (cfg.koch_depth < 0) throw std::runtime_error("depth must be >= 0");

    static std::mutex mtx;
    static int cached_depth = -1;
    static std::string cached_path;
    static KochTriSampler sampler;

    auto resolve_cache_path = [&](int depth) -> std::string {
        // If user did not request disk caching at all
        if (!cfg.use_triangulation_file)
            return {};
        // If user explicitly provided a path → use it
        if (!cfg.triangulation_file.empty())
            return cfg.triangulation_file;
        // default
        return "/tmp/koch_tris_depth_" + std::to_string(depth) + ".bin";
    };

    const std::string path = resolve_cache_path(cfg.koch_depth);

    {
        std::lock_guard<std::mutex> lock(mtx);

        bool need_rebuild = (cached_depth != cfg.koch_depth) || (cached_path != path);

        if (need_rebuild) {
            // 1) try load
            bool loaded = false;
            if (cfg.use_triangulation_file && file_exists(path)) {
                sampler = read_sampler_binary(path);
                loaded = true;
            }

            // 2) build if not loaded
            if (!loaded) {
                sampler = build_koch_trisampler(cfg.koch_depth);

                if (cfg.use_triangulation_file && !path.empty()) {
                    try {
                        std::cout << path << std::endl;
                        write_sampler_binary(sampler, path);
                    } catch (const std::exception& e) {
                        std::cerr << "[koch] warning: could not write triangulation cache: " << e.what() << "\n";
                    }
                }
            }

            cached_depth = cfg.koch_depth;
            cached_path = path;
        }
    }

    if (sampler.empty()) throw std::runtime_error("sampler is empty");

    parlay::parallel_for(0, cfg.num_points, [&](int i) {
        auto& rng = tls_rng();
        double x, y;
        sampler.sample(rng, x, y);
        pts[i].x[0] = x;
        pts[i].x[1] = y;
    });
}


/**
 * @brief Fills a point container with i.i.d. uniform samples from a 2D Koch snowflake domain.
 *
 * @tparam T Container type holding at least `num` elements.
 *
 * @param pts   Output container of points; on return, pts[0..num-1] are filled.
 * @param num   Number of points to generate. If num <= 0, the function returns immediately.
 * @param depth Refinement depth for the Koch snowflake polygonal approximation.
 *
 * @details
 * This routine generates points approximately uniformly distributed over the
 * interior of a Koch snowflake in 2D, represented by a polygonal boundary
 * returned by `koch_polygon(depth)`. Sampling is performed via parallel
 * rejection sampling:
 *
 *  1. Construct the polygonal approximation and its edge table:
 *     - `poly = koch_polygon(depth)`
 *     - `E = build_edge_table(poly)` (includes bounding box [xmin,xmax]×[ymin,ymax])
 *
 *  2. In parallel across workers:
 *     - Generate candidate points uniformly in the polygon bounding box.
 *     - Accept those inside the polygon via `point_in_poly_fast`.
 *     - Buffer accepted points locally and flush them into the shared output
 *       array using an atomic index.
 *
 * Parallelism and RNG:
 *  - Uses `parlay::parallel_for` over `parlay::num_workers()` workers.
 *  - Each worker uses a thread-local PRNG via `tls_rng()`.
 *
 * Rejection sampling:
 *  - Candidates are drawn from Uniform([xmin,xmax]×[ymin,ymax]).
 *  - Acceptance criterion: point-in-polygon even–odd test.
 *  - Expected acceptance rate equals (polygon area) / (bounding box area),
 *    which decreases as the snowflake becomes more intricate.
 *
 * Buffering strategy:
 *  - Candidates are generated in chunks of size CAND to amortize RNG cost.
 *  - Accepted points are accumulated in a small local buffer of size ACC to
 *    reduce contention on the atomic output index.
 *
 * Requirements on T:
 *  - Must support indexing: pts[i]
 *  - Each element must support coordinate writes:
 *        pts[i].x[0], pts[i].x[1]
 *
 * Preconditions:
 *  - pts contains at least `num` writable elements.
 *  - depth should be >= 0. (Negative depth behavior depends on koch_polygon.)
 *
 * Thread safety:
 *  - Safe for parallel execution provided each index in pts is independent.
 *  - Writing is coordinated by an atomic counter to prevent overlap.
 *
 * Boundary behavior:
 *  - Points are accepted using `point_in_poly_fast`, whose classification of
 *    boundary points depends on floating-point rounding and its inequality
 *    convention. Boundary has measure zero, so this typically does not affect
 *    uniformity in practice.
 *
 * Performance notes:
 *  - For large `depth`, the polygon has O(4^depth) edges, making point-in-polygon
 *    checks increasingly expensive.
 *  - The bounding box remains fixed scale, but the acceptance rate may drop.
 *
 * @warning
 * - This routine may not terminate quickly if acceptance rate becomes very low
 *   (e.g., if the bounding box is much larger than the polygon area or if the
 *   polygon degenerates). In typical Koch snowflake settings, it converges,
 *   but runtime increases with depth.
 */
template<typename T>
void fill_from_koch_snowflake_band_2d(
    T& pts,
    int num,
    int depth,
    double thickness /* e.g. 1e-3 */,
    bool include_boundary = true
) {
    if (num <= 0) return;
    if (depth < 0) throw std::runtime_error("fill_from_koch_snowflake_band_2d: depth must be >= 0");
    if (!(thickness > 0.0)) throw std::runtime_error("fill_from_koch_snowflake_band_2d: thickness must be > 0");

    // 1) Inner polygon
    std::vector<Vec2> inner = koch_polygon(depth);
    if (inner.size() < 4) throw std::runtime_error("Koch polygon too small");

    // 2) Outer polygon via centroid scaling (same centroid)
    const Vec2 c = polygon_centroid_area(inner);
    const double s = 1.0 + thickness;
    std::vector<Vec2> outer = scale_polygon_about(inner, c, s);

    // 3) Edge tables
    const EdgeTable Ein = build_edge_table(inner);
    const EdgeTable Eout = build_edge_table(outer);

    // Sample from outer bbox (higher acceptance than sampling from a bbox that encloses both)
    const double xmin = Eout.xmin, xmax = Eout.xmax;
    const double ymin = Eout.ymin, ymax = Eout.ymax;

    std::atomic<int> out_idx{0};

    // Candidate generation chunk + accepted buffer sizes
    constexpr int CAND = 4096;
    constexpr int ACC  = 512;

    parlay::parallel_for(0, parlay::num_workers(), [&](int /*tid*/) {
        auto& rng = tls_rng();

        double xs[CAND];
        double ys[CAND];

        // local accepted buffer
        double ax[ACC];
        double ay[ACC];
        int aCount = 0;

        auto flush = [&]() {
            if (aCount == 0) return;
            int start = out_idx.fetch_add(aCount, std::memory_order_relaxed);
            int end = std::min(start + aCount, num);
            int writeN = end - start;
            for (int j = 0; j < writeN; ++j) {
                pts[start + j].x[0] = ax[j];
                pts[start + j].x[1] = ay[j];
            }
            aCount = 0;
        };

        while (out_idx.load(std::memory_order_relaxed) < num) {
            // generate candidates in outer bbox
            for (int i = 0; i < CAND; ++i) {
                xs[i] = uniform_ab(rng, xmin, xmax);
                ys[i] = uniform_ab(rng, ymin, ymax);
            }

            // filter into the band: inside outer AND (not inside inner)
            for (int i = 0; i < CAND; ++i) {
                const double x = xs[i], y = ys[i];

                // quick reject
                if (!point_in_poly_fast(x, y, Eout)) continue;

                // exclude the inner region
                if (point_in_poly_fast(x, y, Ein)) continue;

                // If we need a strict band without boundary points, we would need a
                // predicate that distinguishes boundary vs interior. The current
                // even-odd test treats boundary inconsistently depending on geometry.
                // For most Monte Carlo uses, include_boundary=true is fine.
                (void)include_boundary;

                ax[aCount] = x;
                ay[aCount] = y;
                ++aCount;

                if (aCount == ACC) {
                    flush();
                    if (out_idx.load(std::memory_order_relaxed) >= num) return;
                }
            }
            flush();
        }
    });
}