#include <iostream>
#include <algorithm>
#include <random>
#include <cassert>
#include <iomanip>
#include <cmath>
#include <tuple>
#include <atomic>
#include <vector>
#include <numbers>
#include <fmt/core.h>
#include <fmt/color.h>
#include <cmath>
#include <stdexcept>

#include "euclideanMst/euclideanMst.h"

#include "parlay/parallel.h"
#include "parlay/utilities.h"
#include "pargeo/point.h"
#include "pargeo/parseCommandLine.h"
#include "spatialGraph/spatialGraph.h"

#include <sqlite3.h>
#include <boost/program_options.hpp>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include "cnpy.h" 

namespace po = boost::program_options;
using namespace std;
using namespace parlay;
using namespace pargeo;




template <typename T>
void write_pts_as_npy_xy(const T& pts, int num, const std::string& path) {
    std::vector<double> buf;
    buf.resize((size_t)num * 2);

    for (int i = 0; i < num; ++i) {
        buf[(size_t)i * 2 + 0] = pts[i].x[0];
        buf[(size_t)i * 2 + 1] = pts[i].x[1];
    }
    // shape = (num, 2), row-major
    cnpy::npy_save(path, buf.data(), { (size_t)num, (size_t)2 }, "w");
}

static inline double stable_normalized_mst(
    double mst_length,
    int numPoints,
    int p,
    int intdim,
    double volume)
{
    if (mst_length <= 0.0) throw std::runtime_error("mst_length must be > 0");
    if (numPoints <= 0)    throw std::runtime_error("numPoints must be > 0");
    if (intdim <= 0)       throw std::runtime_error("intdim must be > 0");
    if (!(volume > 0.0))   throw std::runtime_error("volume must be > 0");

    const double a = 1.0 - (double)p / (double)intdim; 
    const double b = (double)p / (double)intdim;
    const double log_norm = a * std::log((double)numPoints) + b * std::log(volume);
    const double log_mst_norm = std::log(mst_length) - log_norm;
    return expl(log_mst_norm);
}

// ---------------- Koch snowflake helpers ---------------- 
struct Vec2 { double x, y; };
struct EdgeTable {
    std::vector<double> x1, y1, x2, y2;
    std::vector<double> inv_dy;     // 1/(y2-y1) for non-horizontal edges
    std::vector<uint8_t> active;    // 1 if edge should be considered (non-horizontal)
    double xmin, xmax, ymin, ymax;
};

static inline Vec2 operator+(const Vec2& a, const Vec2& b) { return {a.x + b.x, a.y + b.y}; }
static inline Vec2 operator-(const Vec2& a, const Vec2& b) { return {a.x - b.x, a.y - b.y}; }
static inline Vec2 operator*(const Vec2& a, double s) { return {a.x * s, a.y * s}; }

static inline Vec2 rotate(const Vec2& v, double ang) {
    double c = std::cos(ang), s = std::sin(ang);
    return {c*v.x - s*v.y, s*v.x + c*v.y};
}

// Build closed polygon vertices for Koch snowflake at given depth.
// Initial triangle: (0,0), (1,0), (1/2, sqrt(3)/2) (same as your Python).
static std::vector<Vec2> koch_polygon(int depth) {
    std::vector<Vec2> v = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.5, std::sqrt(3.0)/2.0},
        {0.0, 0.0} // close
    };
    const double ang = -M_PI/3.0;

    for (int d = 0; d < depth; ++d) {
        std::vector<Vec2> nv;
        nv.reserve((v.size()-1)*4 + 1);
        for (size_t i = 0; i + 1 < v.size(); ++i) {
            Vec2 p1 = v[i], p2 = v[i+1];
            Vec2 diff = (p2 - p1) * (1.0/3.0);
            Vec2 rotv = rotate(diff, ang);

            Vec2 a = p1;
            Vec2 b = p1 + diff;
            Vec2 c = b + rotv;
            Vec2 e = b + diff;

            nv.push_back(a);
            nv.push_back(b);
            nv.push_back(c);
            nv.push_back(e);
        }
        nv.push_back(nv.front());
        v.swap(nv);
    }
    return v;
}

static EdgeTable build_edge_table(const std::vector<Vec2>& poly) {
    EdgeTable E;
    const size_t n = poly.size();
    if (n < 3) throw std::runtime_error("Koch polygon too small");
    const size_t m = n - 1; // edges from i -> i+1 (poly closed)

    E.x1.resize(m); E.y1.resize(m);
    E.x2.resize(m); E.y2.resize(m);
    E.inv_dy.resize(m);
    E.active.resize(m);

    E.xmin = E.xmax = poly[0].x;
    E.ymin = E.ymax = poly[0].y;

    for (size_t i = 0; i < n; ++i) {
        E.xmin = std::min(E.xmin, poly[i].x);
        E.xmax = std::max(E.xmax, poly[i].x);
        E.ymin = std::min(E.ymin, poly[i].y);
        E.ymax = std::max(E.ymax, poly[i].y);
    }

    for (size_t i = 0; i < m; ++i) {
        double ax = poly[i].x, ay = poly[i].y;
        double bx = poly[i+1].x, by = poly[i+1].y;
        E.x1[i] = ax; E.y1[i] = ay;
        E.x2[i] = bx; E.y2[i] = by;

        double dy = by - ay;
        if (dy == 0.0) {
            E.active[i] = 0;   // horizontal edge: skip in crossing test
            E.inv_dy[i] = 0.0;
        } else {
            E.active[i] = 1;
            E.inv_dy[i] = 1.0 / dy;
        }
    }
    return E;
}

// Even-odd rule crossing test using precomputed edges.
// Uses the standard "straddle" test: (y1>y)!=(y2>y), then compute x-intersect.
static inline bool point_in_poly_fast(double x, double y, const EdgeTable& E) {
    bool inside = false;
    const size_t m = E.x1.size();
    for (size_t i = 0; i < m; ++i) {
        if (!E.active[i]) continue;
        double y1 = E.y1[i], y2 = E.y2[i];

        // straddle check
        bool c = (y1 > y) != (y2 > y);
        if (!c) continue;

        // x intersection (no divide: multiply by inv_dy)
        double x1 = E.x1[i], x2 = E.x2[i];
        double xint = x1 + (x2 - x1) * ((y - y1) * E.inv_dy[i]);
        if (xint >= x) inside = !inside;
    }
    return inside;
}

// Shoelace area (for optional volume sanity; polygon must be closed or not—works either way).
static double polygon_area(const std::vector<Vec2>& poly) {
    const size_t n = poly.size();
    if (n < 3) return 0.0;
    double a = 0.0;
    for (size_t i = 0; i + 1 < n; ++i) {
        a += poly[i].x * poly[i+1].y - poly[i+1].x * poly[i].y;
    }
    // if not explicitly closed, you'd also add last->first; here poly is closed
    return 0.5 * std::abs(a);
}

static inline void write_koch_polygon_as_npy_xy(int depth, const std::string& path) {
    const std::vector<Vec2> poly = koch_polygon(depth);     // includes closing vertex
    const size_t m = poly.size();

    std::vector<double> buf;
    buf.resize(m * 2);

    for (size_t i = 0; i < m; ++i) {
        buf[i * 2 + 0] = poly[i].x;
        buf[i * 2 + 1] = poly[i].y;
    }
    cnpy::npy_save(path, buf.data(), {m, size_t(2)}, "w");
}

// ---------------- Gr(n,k) volume computation ---------------- 
// log GammaOwn(k, a) = (k(k-1)/4) * log(pi) + sum_{i=1}^k lgamma(a - (i-1)/2)
static inline long double logGammaOwn(int k, long double a) {
    if (k <= 0) throw std::runtime_error("logGammaOwn: k must be > 0");

    const long double pi = acosl(-1.0L);
    const long double logpi = logl(pi);

    long double s = (long double)k * (k - 1) * 0.25L * logpi;

    for (int i = 1; i <= k; ++i) {
        long double arg = a - (long double)(i - 1) * 0.5L;
        s += lgammal(arg);
    }
    return s;
}

// log VolumeO(k) = k*log(2) + (k^2/2)*log(pi) - logGammaOwn(k, k/2)
static inline long double logVolumeO(int k) {
    if (k <= 0) throw std::runtime_error("logVolumeO: k must be > 0");

    const long double pi = acosl(-1.0L);
    const long double logpi = logl(pi);
    const long double log2  = logl(2.0L);

    long double kk = (long double)k;
    long double a  = kk * 0.5L;

    return kk * log2 + (kk * kk * 0.5L) * logpi - logGammaOwn(k, a);
}

// log GrVol(n,k) = logVolO(n) - logVolO(k) - logVolO(n-k)
static inline long double logGrassmannVolume(int n, int k) {
    if (n <= 0) throw std::runtime_error("logGrassmannVolume: n must be > 0");
    if (k < 0 || k > n) throw std::runtime_error("logGrassmannVolume: require 0 <= k <= n");
    if (k == 0 || k == n) return 0.0L;
    return logVolumeO(n) - logVolumeO(k) - logVolumeO(n - k);
}

static inline double grassmannVolume(int n, int k) {
    long double lv = logGrassmannVolume(n, k);

    // guard exp overflow/underflow for double
    const long double log_max = logl((long double)std::numeric_limits<double>::max());
    const long double log_min = logl((long double)std::numeric_limits<double>::min());

    if (lv > log_max) return std::numeric_limits<double>::infinity();
    if (lv < log_min) return 0.0;

    return (double)expl(lv); // std::exp overload works for long double too
}

// -------------------------Sphere surface computation ------------------------
// log surface area of unit n-sphere S^n in R^{n+1}
static inline long double compute_sphere_log_surface_area(int n) {
    if (n < 0) throw std::runtime_error("compute_sphere_log_surface_area: n must be >= 0");

    const long double pi = acosl(-1.0L);
    const long double logpi = logl(pi);
    const long double a = ((long double)n + 1.0L) * 0.5L;  // (n+1)/2
    // log( 2 * pi^a / Gamma(a) ) = log(2) + a*log(pi) - lgamma(a)
    return logl(2.0L) + a * logpi - lgammal(a);
}

// surface area of unit n-sphere S^n
static inline double compute_sphere_surface_area(int n) {
    const long double lv = compute_sphere_log_surface_area(n);

    const long double log_max = logl((long double)std::numeric_limits<double>::max());
    const long double log_min = logl((long double)std::numeric_limits<double>::min());

    if (lv > log_max) return std::numeric_limits<double>::infinity();
    if (lv < log_min) return 0.0;
    return (double)expl(lv); // if expl missing in your env, use exp(lv)
}

// ----------------------Volume of B_d(0,1)----------------------
// log volume of unit n-ball in R^n (radius = 1)
static inline long double compute_log_unit_ball_volume(int n) {
    if (n < 0) throw std::runtime_error("logUnitBallVolume: n must be >= 0");
    if (n == 0) return 0.0L; // volume of 0-ball is 1

    const long double pi = acosl(-1.0L);
    const long double logpi = logl(pi);

    const long double a = (long double)n * 0.5L;     // n/2
    // log( pi^(n/2) / Gamma(n/2 + 1) ) = (n/2)*log(pi) - lgamma(n/2 + 1)
    return a * logpi - lgammal(a + 1.0L);
}

static inline double compute_unit_ball_volume(int n) {
    long double lv = compute_log_unit_ball_volume(n);

    const long double log_max = logl((long double)std::numeric_limits<double>::max());
    const long double log_min = logl((long double)std::numeric_limits<double>::min());

    if (lv > log_max) return std::numeric_limits<double>::infinity();
    if (lv < log_min) return 0.0;
    return (double)expl(lv);
}

// ---------------- RNG: xoshiro256++ seeded once per thread ----------------
struct Xoshiro256pp {
    uint64_t s[4];

    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    inline uint64_t next_u64() {
        const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    // [0,1) double with 53 bits precision
    inline double next_double01() {
        return (next_u64() >> 11) * (1.0 / 9007199254740992.0); // 2^53
    }
};

// SplitMix64 for seeding
static inline uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline Xoshiro256pp& tls_rng() {
    static thread_local Xoshiro256pp rng;
    static thread_local bool seeded = false;
    if (!seeded) {
        std::random_device rd;

        // Mix multiple rd() draws; some implementations have limited entropy per call.
        uint64_t x = 0;
        for (int i = 0; i < 8; ++i) {
            x ^= (uint64_t(rd()) << 32) ^ uint64_t(rd());
            x = splitmix64(x);
        }
        rng.s[0] = splitmix64(x);
        rng.s[1] = splitmix64(x);
        rng.s[2] = splitmix64(x);
        rng.s[3] = splitmix64(x);
        seeded = true;
    }
    return rng;
}

/* generate uniform random numbers in [a,b] */
static inline double uniform_ab(Xoshiro256pp& r, double a, double b) {
    return a + (b - a) * r.next_double01();
}

/* generate std. Gaussian N(0,1) random numbers (avoiding log(0)) using Box-Muller */
static inline double normal01(Xoshiro256pp& rng) {
    double u1 = rng.next_double01();
    double u2 = rng.next_double01();
    u1 = std::max(u1, 1e-300);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

/* uniform sampling in [0,1]^d */
template<int dim, typename T, typename internal>
void fill_from_uniform_in_unitbox(T &pts, int num) {
    parlay::parallel_for(0, num, [&](int i) {
        auto& rng = tls_rng();
        for (int k = 0; k < dim; ++k) {
            pts[i].x[k] = static_cast<internal>(rng.next_double01()); // [0,1)
        }
    });
}

/* uniform sampling within d-dimensional unit ball B_d(0,1)*/
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

/* uniform sampling (via rejection) from 2D Koch snowflake*/
template<typename T>
void fill_from_koch_snowflake_2d(T& pts, int num, int depth) {
    if (num <= 0) return;

    std::vector<Vec2> poly = koch_polygon(depth);
    EdgeTable E = build_edge_table(poly);

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
            // generate candidates
            for (int i = 0; i < CAND; ++i) {
                xs[i] = uniform_ab(rng, E.xmin, E.xmax);
                ys[i] = uniform_ab(rng, E.ymin, E.ymax);
            }

            // filter
            for (int i = 0; i < CAND; ++i) {
                if (!point_in_poly_fast(xs[i], ys[i], E)) continue;
                ax[aCount] = xs[i];
                ay[aCount] = ys[i];
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

inline double sphere_geodesic_from_chord(double chord) noexcept {
    // chord should be in [0,2] for unit sphere; clamp for numerical safety
    double x = 0.5 * chord;
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return M_PI; // antipodal (or clamp)
    return 2.0 * std::asin(x);
}

inline double sphere_geodesic_from_chord_atan2(double chord) noexcept {
    double c = std::clamp(chord, 0.0, 2.0);
    double dot = 1.0 - 0.5 * c * c;   // cos(theta)
    double sin_theta = std::sqrt(std::max(0.0, 1.0 - dot * dot));
    return std::atan2(sin_theta, dot);
}

template<int dim, int p>
std::tuple<int, double> callEuclideanMstGrassmann(
    int num,
    int n,
    int k
) {  
    parlay::sequence<pargeo::point<dim>> pts(num);

    fill_from_grassmann_svd_projection<dim, parlay::sequence<pargeo::point<dim>>, double>(
        pts, num, n, k);

    auto I = euclideanMst<dim>(pts);
    auto S = pts.data();

    double sum = 0.0;
    double compensation = 0.0;
    for (auto e : I) {
        double dist = S[e.u].dist(S[e.v]);
        double pdist = std::pow(dist, p);
        double y = pdist - compensation;
        double t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    return std::make_tuple(num, sum);
}

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

template<int dim, int p>
std::tuple<int, double> callEuclideanMst(
    int num,
    std::string shape,
    bool use_geodesic_on_sphere=false,
    int koch_depth=5
) {
    parlay::sequence<pargeo::point<dim>> pts(num);

    if (shape == "cube") {
        fill_from_uniform_in_unitbox<dim, parlay::sequence<pargeo::point<dim>>, double>(pts, num);
    } else if (shape == "ball") {
        fill_from_uniform_in_unitball<dim, parlay::sequence<pargeo::point<dim>>, double>(pts, num);
    } else if (shape == "sphere") {
        fill_from_uniform_on_unitsphere<dim, parlay::sequence<pargeo::point<dim>>, double>(pts, num);
    } else if (shape == "koch") {
        fill_from_koch_snowflake_2d(pts, num, koch_depth);
        //write_pts_as_npy_xy(pts, num, "/tmp/koch_pts.npy");
    } else {
        throw std::runtime_error("Unknown shape (expected cube|ball|sphere|grassmann|koch)");
    }

    auto I = euclideanMst<dim>(pts);
    auto S = pts.data();

    double sum = 0.0;
    double c = 0.0; // Neumaier compensation

    for (auto e : I) {
        double d = S[e.u].dist(S[e.v]);
        if (use_geodesic_on_sphere && shape == "sphere") {
            d = sphere_geodesic_from_chord_atan2(d);
        }

        double pdist = std::pow(d, p);

        // Neumaier summation
        double t = sum + pdist;
        if (std::abs(sum) >= std::abs(pdist)) {
            c += (sum - t) + pdist;
        } else {
            c += (pdist - t) + sum;
        }
        sum = t;
    }

    sum += c;
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
        double pdist = pow(dist, p);
        double y = pdist - compensation;
        double t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    return std::make_tuple(rows, sum);
}

template <int dim, int p>
std::tuple<int, double> dispatchEuclideanMst(
    int numPoints,
    const std::string& shape,
    const std::string& inputFile,
    int gr_n,
    int gr_k,
    bool sphere_geodesic,
    int koch_depth
) {
    if (!inputFile.empty()) {
        return callEuclideanMstFromFile<dim,p>(inputFile);
    }
    if (shape == "grassmann") {
        return callEuclideanMstGrassmann<dim,p>(numPoints, gr_n, gr_k);
    }
    return callEuclideanMst<dim,p>(numPoints, shape, sphere_geodesic, koch_depth);
}

using DispatchFn = std::tuple<int, double>(*)(
    int, const std::string&, const std::string&, int, int, bool, int
);

template<int dim, int p>
std::tuple<int, double> dispatchHelper(
    int numPoints,
    const std::string& shape,
    const std::string& inputFile,
    int gr_n,
    int gr_k,
    bool sphere_geodesic,
    int koch_depth
)
{
    return dispatchEuclideanMst<dim, p>(
        numPoints, shape, inputFile, gr_n, gr_k, sphere_geodesic, koch_depth
    );
}

std::tuple<int, double> dispatch(
    int dim, 
    int p, 
    int numPoints,
    const std::string& shape, 
    const std::string& inputFile,
    int gr_n, 
    int gr_k, 
    bool sphere_geodesic,
    int koch_depth
) {
    if (inputFile.empty()) {
        if (shape.empty()) throw std::runtime_error("No shape given!");
        if (numPoints < 0) throw std::runtime_error("Number of points not >0!");
        if (dim < 2 || dim > 16) throw std::runtime_error("Dimension needs to be in {2,...,16}");
    }

    // Table of function pointers for p = 1 to 5 and dim = 2 to 16
    static const DispatchFn dispatchTable[15][5] = {
        {&dispatchHelper<2, 1>, &dispatchHelper<2, 2>, &dispatchHelper<2, 3>, &dispatchHelper<2, 4>, &dispatchHelper<2, 5>},
        {&dispatchHelper<3, 1>, &dispatchHelper<3, 2>, &dispatchHelper<3, 3>, &dispatchHelper<3, 4>, &dispatchHelper<3, 5>},
        {&dispatchHelper<4, 1>, &dispatchHelper<4, 2>, &dispatchHelper<4, 3>, &dispatchHelper<4, 4>, &dispatchHelper<4, 5>},
        {&dispatchHelper<5, 1>, &dispatchHelper<5, 2>, &dispatchHelper<5, 3>, &dispatchHelper<5, 4>, &dispatchHelper<5, 5>},
        {&dispatchHelper<6, 1>, &dispatchHelper<6, 2>, &dispatchHelper<6, 3>, &dispatchHelper<6, 4>, &dispatchHelper<6, 5>},
        {&dispatchHelper<7, 1>, &dispatchHelper<7, 2>, &dispatchHelper<7, 3>, &dispatchHelper<7, 4>, &dispatchHelper<7, 5>},
        {&dispatchHelper<8, 1>, &dispatchHelper<8, 2>, &dispatchHelper<8, 3>, &dispatchHelper<8, 4>, &dispatchHelper<8, 5>},
        {&dispatchHelper<9, 1>, &dispatchHelper<9, 2>, &dispatchHelper<9, 3>, &dispatchHelper<9, 4>, &dispatchHelper<9, 5>},
        {&dispatchHelper<10, 1>, &dispatchHelper<10, 2>, &dispatchHelper<10, 3>, &dispatchHelper<10, 4>, &dispatchHelper<10, 5>},
        {&dispatchHelper<11, 1>, &dispatchHelper<11, 2>, &dispatchHelper<11, 3>, &dispatchHelper<11, 4>, &dispatchHelper<11, 5>},
        {&dispatchHelper<12, 1>, &dispatchHelper<12, 2>, &dispatchHelper<12, 3>, &dispatchHelper<12, 4>, &dispatchHelper<12, 5>},
        {&dispatchHelper<13, 1>, &dispatchHelper<13, 2>, &dispatchHelper<13, 3>, &dispatchHelper<13, 4>, &dispatchHelper<13, 5>},
        {&dispatchHelper<14, 1>, &dispatchHelper<14, 2>, &dispatchHelper<14, 3>, &dispatchHelper<14, 4>, &dispatchHelper<14, 5>},
        {&dispatchHelper<15, 1>, &dispatchHelper<15, 2>, &dispatchHelper<15, 3>, &dispatchHelper<15, 4>, &dispatchHelper<15, 5>},
        {&dispatchHelper<16, 1>, &dispatchHelper<16, 2>, &dispatchHelper<16, 3>, &dispatchHelper<16, 4>, &dispatchHelper<16, 5>},
    };

    // Ensure p and dim are within valid ranges
    if (dim < 2 || dim > 16 || p < 1 || p > 5) {
        throw std::invalid_argument("Unsupported dimension or p value");
    }
    return dispatchTable[dim - 2][p - 1](
        numPoints, shape, inputFile, gr_n, gr_k, sphere_geodesic, koch_depth
    );
}

int main(int argc, char* argv[]) 
{
    // always required args by parser (other args are required based on context and checked later)
    std::string inputFile;      // input file (binary numpy tensor)
    std::string dbFile;         // SQLite3 database file
    std::string shape;          // sample from ball or cube
    int numPoints = -1;         // is set on cmdline
    int dim = -1;               // is set on cmdline
    int intdim = -1;            // will be computed
    int p = 1;                  // default p
    int gr_n = -1;              // ambient dim. of Gr(n,k)
    int gr_k = -1;              // subspace dim of Gr(n,k)
    int koch_depth = 5;         // default depth of Koch snowflake
    bool sphere_geodesic = false;
    double volume = std::numeric_limits<double>::quiet_NaN(); // default to NaN, will be computed
    
    try {
        po::options_description desc("Options");
        desc.add_options()
            ("help", "produce help message")
            ("dim", po::value<int>(&dim)->required(), "dimensionality of input vectors (R^d).")
            ("numPoints", po::value<int>(&numPoints)->required(), "Number of points to sample.")
            ("shape", po::value<std::string>(&shape)->required(), "Sample from cube|ball|sphere|grassmann|koch.")
            ("dbFile", po::value<std::string>(&dbFile)->required(), "SQLite3 database file.")
            ("p", po::value<int>(&p), "Power of edge lengths.")
            ("inputFile", po::value<std::string>(&inputFile), "Numpy matrix input file.")
            ("gr_n", po::value<int>(&gr_n), "Grassmann ambient dimension n (R^n).")
            ("gr_k", po::value<int>(&gr_k), "Grassmann subspace dimension k.")
            ("koch_depth", po::value<int>(&koch_depth)->default_value(5),
                "Recursion depth for Koch snowflake (only if shape=koch).")
            ("sphere_geodesic", po::bool_switch(&sphere_geodesic)->default_value(false),
                "If set and --shape sphere, convert Euclidean chord lengths to spherical geodesic lengths when summing MST.");

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

    // make sure we have enough precision for output stats
    std::cout << std::scientific << std::setprecision(12);

    try {
        // dispatch MST computation
        std::tuple<int, double> result = dispatch(
            dim, 
            p, 
            numPoints, 
            shape, 
            inputFile, 
            gr_n, 
            gr_k, 
            sphere_geodesic,
            koch_depth
        );
        
        numPoints = std::get<0>(result);
        double mst_length = std::get<1>(result);

        if (shape == "grassmann") {
            if (gr_n <= 0 || gr_k <= 0) throw std::runtime_error("--shape grassmann requires --gr_n and --gr_k to be set");
            if (gr_n <= gr_k) throw std::runtime_error("--shape grassmann requires gr_n > gr_k");
            if (!std::isfinite(volume)) {
                volume = grassmannVolume(gr_n, gr_k);
            }
            intdim = gr_k * (gr_n - gr_k);
        } else if (shape == "sphere") {
            if (dim < 2) throw std::runtime_error("Sphere requires dim >= 2");
                intdim = dim - 1;
            if (!std::isfinite(volume)) {
                volume = compute_sphere_surface_area(dim - 1); 
            }
        } else if (shape == "ball") {
            if (!std::isfinite(volume)) {
                volume = compute_unit_ball_volume(dim);
            }
            intdim = dim;
        } else if (shape == "koch") {
            if (dim != 2) throw std::runtime_error("--shape koch requires --dim 2");
            if (koch_depth < 0) throw std::runtime_error("--koch_depth must be >= 0");
            auto poly = koch_polygon(koch_depth);
            volume = polygon_area(poly);  
            intdim = dim;  
            //write_koch_polygon_as_npy_xy(koch_depth, "/tmp/koch_polygon.npy");
        } else if (shape == "cube") {
            if (!std::isfinite(volume)) {
                volume = 1.0; // unit cube
            }   
            intdim = dim;
        }
        
        double mst_length_normalized = stable_normalized_mst(mst_length, numPoints, p, intdim, volume);
        fmt::print(
            "| n={:>8d} | volume={:>20.16f} | intdim={:>2d} | L_n={:>22.12f} | L_n (norm)={:>22.16f}\n",
            numPoints, volume, intdim, mst_length, 
            fmt::styled(mst_length_normalized,fmt::fg(fmt::color::green) | fmt::emphasis::bold)
        );

        writeToDatabase(dbFile, numPoints, mst_length, mst_length_normalized);

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return -1;
    }
    return 0;
}
