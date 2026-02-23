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

#pragma region SQLite utilities

// Function to execute an SQL command
void execute_sql_statement(sqlite3* db, const std::string& sql) {
    char* errmsg;
    int rc = sqlite3_exec(db, sql.c_str(), 0, 0, &errmsg);

    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << errmsg << std::endl;
        sqlite3_free(errmsg);
    }
}

// Function to write data to the SQLite database
void write_to_database(const std::string& dbFilename, int num_points, double mst_length, double normalized_mst_length) {
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
    execute_sql_statement(db, createTableSQL);

    // Begin transaction to avoid locking issues during multiple operations
    execute_sql_statement(db, "BEGIN TRANSACTION;");

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
    execute_sql_statement(db, "COMMIT;");

    // Close the database connection
    sqlite3_close(db);
}

#pragma endregion

#pragma region RNG utilities

/**
 * @brief xoshiro256++ pseudorandom number generator.
 *
 * This struct implements the xoshiro256++ PRNG (64-bit output) with a 256-bit
 * internal state stored as 4x 64-bit words.
 *
 * Properties / notes:
 * - Fast, high-quality PRNG suitable for simulations and Monte Carlo.
 * - Not cryptographically secure; do not use for security-sensitive purposes.
 * - The state must be seeded with a non-zero state (all-zero is an invalid
 *   fixed point for xoshiro-family generators).
 *
 */
struct Xoshiro256pp {
    // Internal 256-bit state (4 * 64-bit words). Must not be all zeros.
    uint64_t s[4];

    /**
     * @brief Rotate-left (rotl) operation on a 64-bit unsigned integer.
     *
     * @param x Input value.
     * @param k Rotation amount in bits.
     * @return Value of x rotated left by k bits.
     *
     * @note Precondition: 0 <= k < 64. (For this implementation, callers only
     *       use fixed constants that satisfy this.)
     */
    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    /**
     * @brief Generate the next 64-bit pseudorandom integer and advance state.
     *
     * @return A 64-bit pseudorandom value.
     *
     * @details
     * This is the xoshiro256++ output function: it scrambles state to produce
     * output and updates the internal state using xorshift/rotation operations.
     *
     * @warning Not thread-safe when shared: concurrent calls require external
     *          synchronization or independent generator instances.
     */
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

    /**
     * @brief Generate a uniform double in [0, 1) with 53 bits of precision.
     *
     * @return A floating-point value uniformly distributed on [0, 1).
     *
     * @details
     * Uses the top 53 bits of a 64-bit output to populate the mantissa of an
     * IEEE-754 double (via scaling by 2^-53). This is a standard technique to
     * obtain evenly spaced representable doubles in [0, 1).
     *
     * @note The value 1.0 is never returned.
     */    
    inline double next_double01() {
        return (next_u64() >> 11) * (1.0 / 9007199254740992.0);
    }
};

/**
 * @brief SplitMix64 generator step.
 *
 * Advances the input state and returns a 64-bit pseudorandom value.
 *
 * @param x Reference to the 64-bit state. The state is incremented
 *          and scrambled in-place.
 *
 * @return A 64-bit pseudorandom value derived from the updated state.
 *
 * @details
 * SplitMix64 is a simple, fast generator with 64-bit state. It is commonly
 * used for:
 *
 *  - Seeding more sophisticated generators (e.g. xoshiro256++).
 *  - Hash-like scrambling of integer inputs.
 *
 * It has excellent equidistribution and bit-mixing properties for its size,
 * but should not be used directly for high-quality long-period simulations
 * where stronger generators are preferred.
 *
 * Algorithm:
 *  - Adds a fixed odd Weyl increment (2^64 / φ).
 *  - Applies two multiplicative mixing steps.
 *  - Finishes with a final xorshift.
 *
 * Reference:
 *  - Sebastiano Vigna, "An experimental exploration of Marsaglia's xorshift generators"
 *
 * Thread safety:
 *  - Not thread-safe if the same state variable is shared across threads.
 *
 * Precondition:
 *  - None (any 64-bit value is valid).
 *
 * Period:
 *  - 2^64 (full-period Weyl sequence).
 */
static inline uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/**
 * @brief Returns a thread-local instance of Xoshiro256pp.
 *
 * Provides one independent PRNG instance per thread. The generator is
 * lazily seeded on first use within each thread.
 *
 * @return Reference to the calling thread's Xoshiro256pp instance.
 *
 * @details
 * The generator is stored in thread-local storage (`thread_local`), ensuring:
 *
 *  - Each thread has its own independent RNG state.
 *  - No synchronization is required for concurrent use.
 *  - No false sharing between threads.
 *
 * Seeding strategy:
 *  - Uses std::random_device to obtain entropy.
 *  - Mixes multiple draws to guard against implementations that provide
 *    limited entropy per call.
 *  - Applies splitmix64 to expand entropy into four 64-bit state words.
 *
 * The seeding occurs exactly once per thread (lazy initialization).
 *
 * @note
 * - Not reproducible across runs because std::random_device is used.
 * - If deterministic behavior is required, provide an explicit seeding API.
 *
 * @warning
 * - The quality of std::random_device is implementation-dependent.
 *   On some platforms it may be deterministic.
 * - For reproducible simulations, do not rely on this function.
 *
 * Thread safety:
 *  - Safe for concurrent calls across threads.
 *  - Not safe to share the returned RNG across threads.
 */
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

/**
 * @brief Draws a uniform random double in the interval [a, b).
 *
 * @param r RNG instance (xoshiro256++).
 * @param a Lower bound.
 * @param b Upper bound.
 * @return A double sampled uniformly from [a, b).
 *
 * @details
 * This uses the generator's `next_double01()` which yields a uniform in [0, 1)
 * with 53 bits of precision, then applies an affine transformation:
 *
 *   a + (b - a) * U, where U ~ Uniform([0,1)).
 *
 * @note
 * - The upper bound b is never returned (up to floating-point rounding).
 * - If a == b the result is exactly a.
 *
 * @warning
 * - If b < a, the interval is effectively reversed (result still follows the
 *   same affine map), which is usually unintended. Consider enforcing b >= a.
 * - For very large |a| or |b|, floating-point rounding may reduce resolution.
 */
static inline double uniform_ab(Xoshiro256pp& r, double a, double b) {
    return a + (b - a) * r.next_double01();
}

/**
 * @brief Draws a standard normal N(0, 1) random variate.
 *
 * @param rng RNG instance (xoshiro256++).
 * @return A double approximately distributed as N(0, 1).
 *
 * @details
 * Implements the Box–Muller transform using two independent uniforms U1, U2:
 *
 *   Z = sqrt(-2 ln(U1)) * cos(2*pi*U2)
 *
 * where U1, U2 ~ Uniform([0,1)).
 *
 * Numerical stability:
 * - U1 must be strictly positive due to the logarithm. Since `next_double01()`
 *   can in principle return 0.0, we clamp:
 *
 *   U1 = max(U1, 1e-300)
 *
 * This avoids log(0) and yields a practically negligible bias at extreme tail
 * probabilities.
 *
 * Performance notes:
 * - This implementation returns one normal sample per two uniform draws.
 *   (A common optimization caches the "sine" branch to return a second sample
 *   on the next call.)
 */
static inline double normal01(Xoshiro256pp& rng) {
    double u1 = rng.next_double01();
    double u2 = rng.next_double01();
    u1 = std::max(u1, 1e-300);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}
#pragma endregion

#pragma region misc. uilities

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

/**
 * @brief Computes the log multivariate gamma function log Γ_k(a).
 *
 * @param k Dimension parameter (must be > 0).
 * @param a Real argument.
 * @return log Γ_k(a) as a long double.
 *
 * @details
 * Implements the multivariate gamma function (also called the generalized
 * gamma function) used in multivariate statistics (e.g., Wishart / Inverse-Wishart):
 *
 *   Γ_k(a) = π^{k(k-1)/4} * ∏_{i=1}^{k} Γ(a - (i-1)/2)
 *
 * Therefore:
 *
 *   log Γ_k(a) = (k(k-1)/4) * log π + ∑_{i=1}^{k} log Γ(a - (i-1)/2)
 *
 * This routine evaluates the expression in log-space using `lgammal` for
 * improved numerical stability.
 *
 * Domain / preconditions:
 *  - k > 0
 *  - a must satisfy: a > (k - 1)/2
 *    so that all arguments a - (i-1)/2 are positive and Γ(.) is finite.
 *
 * Exceptions:
 *  - Throws std::runtime_error if k <= 0.
 *
 * @note
 * - This function does not explicitly validate the a-domain constraint.
 *   If violated, `lgammal` may return INF/NaN and/or set errno depending on
 *   the standard library implementation.
 * - Uses long double math (`acosl`, `logl`, `lgammal`) to reduce rounding error.
 */
static inline long double log_mvgamma(int k, long double a) {
    if (k <= 0) throw std::runtime_error("log_mvgamma: k must be > 0");

    const long double pi = acosl(-1.0L);
    const long double logpi = logl(pi);

    long double s = (long double)k * (k - 1) * 0.25L * logpi;

    for (int i = 1; i <= k; ++i) {
        long double arg = a - (long double)(i - 1) * 0.5L;
        s += lgammal(arg);
    }
    return s;
}

#pragma endregion

#pragma region Koch snowflake utilities

/**
 * @brief Lightweight 2D vector with double precision components.
 *
 * This is a simple POD-like value type intended for geometry utilities.
 * No invariants are enforced.
 */
struct Vec2 { double x, y; };
/**
 * @brief Structure-of-arrays representation of polygon edges for scanline-like processing.
 *
 * Stores edges as parallel arrays (SoA) to improve cache locality and vectorization
 * in inner loops compared to an array-of-structs representation.
 *
 * Each edge i is represented by endpoints:
 *   (x1[i], y1[i]) -> (x2[i], y2[i])
 *
 * Additional per-edge metadata:
 *  - inv_dy[i]  = 1 / (y2[i] - y1[i]) for non-horizontal edges
 *  - active[i]  = 1 if the edge is non-horizontal and should be considered
 *
 * Bounding box:
 *  - xmin/xmax/ymin/ymax bound the geometry represented by this table.
 *
 * Invariants (expected by downstream code):
 *  - All per-edge vectors (x1,y1,x2,y2,inv_dy,active) have identical length.
 *  - If active[i] == 0, the corresponding edge is horizontal (y1 == y2) and
 *    inv_dy[i] is either unused or may contain an arbitrary value.
 */
struct EdgeTable {
    std::vector<double> x1, y1, x2, y2;   ///< Edge endpoints in SoA form.
    std::vector<double> inv_dy;           ///< 1/(y2-y1) for non-horizontal edges.
    std::vector<uint8_t> active;          ///< 1 if edge is non-horizontal; else 0.
    double xmin, xmax, ymin, ymax;        ///< Axis-aligned bounding box of all edges.
};

static inline Vec2 operator+(const Vec2& a, const Vec2& b) { return {a.x + b.x, a.y + b.y}; }
static inline Vec2 operator-(const Vec2& a, const Vec2& b) { return {a.x - b.x, a.y - b.y}; }
static inline Vec2 operator*(const Vec2& a, double s) { return {a.x * s, a.y * s}; }

/**
 * @brief Rotates a 2D vector by an angle about the origin.
 *
 * @param v Input vector.
 * @param ang Rotation angle in radians (counterclockwise for positive angles).
 * @return Rotated vector R(ang) * v.
 *
 * @note Uses std::sin/std::cos; for repeated rotations in tight loops, consider
 *       passing precomputed sin/cos or using a rotation matrix to avoid repeated
 *       transcendental evaluations.
 */
static inline Vec2 rotate(const Vec2& v, double ang) {
    double c = std::cos(ang), s = std::sin(ang);
    return {c*v.x - s*v.y, s*v.x + c*v.y};
}
#pragma endregion

#pragma region Koch snowflake (and snowflake band) construction

/**
 * @brief Constructs a closed polygonal approximation of the Koch snowflake boundary.
 *
 * Generates the Koch curve starting from an equilateral triangle and applies
 * `depth` refinement steps. The returned vertex list is a *closed* polyline:
 * the last vertex equals the first.
 *
 * @param depth Number of refinement iterations (must be >= 0).
 * @return Vertices of the closed polygonal chain representing the Koch boundary.
 *
 * @details
 * Initialization:
 *  - Starts with an equilateral triangle with vertices:
 *      (0,0), (1,0), (0.5, sqrt(3)/2)
 *  - The polyline is explicitly closed by repeating the first vertex at the end.
 *
 * Refinement rule (per edge p1 -> p2):
 *  - Split the edge into thirds: diff = (p2 - p1) / 3.
 *  - Construct points:
 *      a = p1
 *      b = p1 + diff
 *      e = p1 + 2*diff
 *  - Insert a “spike” point:
 *      c = b + rotate(diff, -pi/3)
 *
 * Each iteration replaces every segment by 4 segments, so the number of edges
 * grows by a factor of 4 per depth level.
 *
 * @note
 * - Angle units are radians.
 * - The sign of the rotation angle determines whether the spikes point inward
 *   or outward relative to the initial triangle orientation. This implementation
 *   uses -pi/3.
 */
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

/**
 * @brief Builds an edge table (structure-of-arrays) from a closed polygon.
 *
 * Converts a closed polygonal chain into an `EdgeTable` representation suitable
 * for scanline / ray-crossing style computations (e.g., point-in-polygon tests,
 * winding/crossing counts, rasterization-like passes).
 *
 * The input is expected to be *closed*, i.e. `poly.front() == poly.back()`,
 * so that edges are formed as:
 *
 *   edge i: poly[i] -> poly[i+1]  for i = 0 .. n-2
 *
 * Horizontal edges (dy == 0) are marked inactive and can be skipped by
 * downstream crossing tests to avoid degeneracies.
 *
 * @param poly Closed polygon vertex list.
 * @return Fully populated edge table, including bounding box.
 *
 * Preconditions:
 *  - poly.size() >= 3
 *  - poly is closed (poly[0] equals poly[n-1]) if you intend the last-to-first
 *    edge to be excluded. This function assumes closure and uses n-1 edges.
 *
 * Postconditions / invariants:
 *  - All per-edge arrays in the returned `EdgeTable` have the same length m = n-1.
 *  - For each edge i:
 *      (x1[i], y1[i]) = poly[i]
 *      (x2[i], y2[i]) = poly[i+1]
 *  - active[i] == 0 iff dy == 0 (horizontal edge), otherwise active[i] == 1.
 *  - inv_dy[i] == 1/dy for active edges; inv_dy[i] == 0 for inactive edges.
 *  - xmin/xmax/ymin/ymax bound all polygon vertices.
 *
 * Exceptions:
 *  - Throws std::runtime_error if the polygon has fewer than 3 vertices.
 *
 * @note
 * Using exact equality `dy == 0.0` treats only perfectly horizontal edges as
 * inactive. If vertices may contain floating-point noise, we should instead consider 
 * replacing this check with an epsilon comparison (e.g. |dy| < eps).
 */
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

/**
 * @brief Fast point-in-polygon test using a precomputed edge table.
 *
 * Tests whether the query point (x, y) lies inside a (simple) polygon using
 * the ray-crossing (even–odd) rule. This implementation casts a horizontal ray
 * to +infinity in x-direction and toggles the inside state on each edge
 * intersection.
 *
 * @param x Query x-coordinate.
 * @param y Query y-coordinate.
 * @param E Precomputed edge table (typically from `build_edge_table`).
 * @return True if the point is inside the polygon according to the even–odd rule,
 *         false otherwise.
 *
 * @details
 * For each non-horizontal edge i, we:
 *  1. Perform a straddle test to check whether the edge crosses the horizontal
 *     line y = query_y:
 *
 *       (y1 > y) != (y2 > y)
 *
 *     This convention excludes hits exactly at vertices in a consistent way
 *     and avoids double-counting shared endpoints when combined with skipping
 *     horizontal edges.
 *
 *  2. Compute the x-coordinate of the intersection of the edge with y = query_y:
 *
 *       x_int = x1 + (x2 - x1) * ((y - y1) / (y2 - y1))
 *
 *     Using `inv_dy[i] = 1/(y2 - y1)` avoids a division in the inner loop.
 *
 *  3. Toggle the parity if the intersection lies to the right of the query point
 *     (`x_int >= x`), which corresponds to the ray from (x, y) to +∞.
 *
 *
 * Preconditions:
 *  - E’s arrays have consistent lengths and represent edges (as produced by
 *    `build_edge_table`).
 *  - E.active[i] == 0 for horizontal edges, and E.inv_dy[i] is valid for active edges.
 *
 * @note Boundary behavior:
 * - This routine implements a strict even–odd crossing test. Points exactly on
 *   the polygon boundary (on an edge or vertex) are not handled explicitly and
 *   may return either true or false depending on floating-point rounding and
 *   the chosen inequality (`>=`).
 * - If you need a robust “inside-or-on-boundary” predicate, add an explicit
 *   segment-distance / colinearity check before the parity test.
 *
 * Numerical notes:
 * - Uses exact comparisons for the straddle test. For noisy inputs, consider
 *   epsilon handling for y comparisons.
 */
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

/**
 * @brief Computes the (unsigned) area of a closed polygon using the shoelace formula.
 *
 * @param poly Vertex list of a polygonal chain.
 * @return Non-negative area of the polygon.
 *
 * @details
 * Uses the classical shoelace (Gauss area) formula:
 *
 *   A = 1/2 * | Σ (x_i y_{i+1} − x_{i+1} y_i) |
 *
 * This implementation assumes the polygon is explicitly closed
 * (i.e. `poly.front() == poly.back()`), so edges are formed as:
 *
 *   (poly[i] -> poly[i+1]) for i = 0 .. n-2
 *
 * The absolute value is taken, so the result is orientation-independent.
 *
 * Preconditions:
 *  - If `poly` is closed, the last vertex equals the first.
 *  - If not closed, the last-to-first edge is not included in this implementation.
 *
 * @note
 * - If you require the signed area (positive for CCW, negative for CW),
 *   remove `std::abs`.
 * - For self-intersecting polygons, this computes the algebraic area,
 *   not necessarily the geometric union area.
 */
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

/**
 * @brief Computes the area centroid (center of mass) of a simple polygon.
 *
 * @param poly Vertex list of a polygon (closed or non-closed).
 * @return The centroid of the polygon’s interior (uniform density).
 *
 * @details
 * Uses the standard area-centroid formula derived from the shoelace identity:
 *
 *   A2 = Σ (x_i y_{i+1} − x_{i+1} y_i)          (twice signed area)
 *
 *   Cx = (1 / (6A)) Σ (x_i + x_{i+1}) * cross_i
 *   Cy = (1 / (6A)) Σ (y_i + y_{i+1}) * cross_i
 *
 * Since A2 = 2A, we use:
 *
 *   inv6A = 1 / (3 * A2)
 *
 * and compute:
 *
 *   Cx = Cx_sum * inv6A
 *   Cy = Cy_sum * inv6A
 *
 * The algorithm works for both closed polygons
 * (`poly.front() == poly.back()`) and open vertex lists.
 *
 * Numerical considerations:
 *  - Accumulation is performed in long double to reduce cancellation error.
 *  - The result is returned as double.
 *
 * Preconditions:
 *  - Polygon must have at least 3 distinct vertices.
 *  - Polygon must have non-zero signed area.
 *  - Polygon is assumed to be simple (non-self-intersecting).
 *
 * Exceptions:
 *  - Throws std::runtime_error if polygon is too small.
 *  - Throws std::runtime_error if the polygon is degenerate (zero area).
 *
 * @note
 * - The sign of A2 depends on vertex orientation (CCW positive, CW negative).
 *   The centroid formula remains correct because the same sign appears in
 *   numerator and denominator.
 * - For self-intersecting polygons, this computes the algebraic centroid,
 *   not necessarily the geometric centroid of the union region.
 */
static inline Vec2 polygon_centroid_area(const std::vector<Vec2>& poly) {
    const size_t n = poly.size();
    if (n < 3) throw std::runtime_error("polygon_centroid_area: poly too small");

    const size_t m = (n >= 2 && poly.front().x == poly.back().x && poly.front().y == poly.back().y) ? (n - 1) : n;
    if (m < 3) throw std::runtime_error("polygon_centroid_area: need at least 3 distinct vertices");

    long double A2 = 0.0L; // 2*Area signed
    long double Cx = 0.0L;
    long double Cy = 0.0L;

    for (size_t i = 0; i < m; ++i) {
        const Vec2& p = poly[i];
        const Vec2& q = poly[(i + 1) % m];
        const long double cross = (long double)p.x * (long double)q.y - (long double)q.x * (long double)p.y;
        A2 += cross;
        Cx += ((long double)p.x + (long double)q.x) * cross;
        Cy += ((long double)p.y + (long double)q.y) * cross;
    }

    if (A2 == 0.0L) throw std::runtime_error("polygon_centroid_area: degenerate polygon (zero area)");

    const long double inv6A = 1.0L / (3.0L * A2); // since centroid uses 1/(6A) and A2=2A => 1/(3*A2)
    Vec2 c;
    c.x = (double)(Cx * inv6A);
    c.y = (double)(Cy * inv6A);
    return c;
}

/**
 * @brief Uniformly scales a polygon about a specified center point.
 *
 * @param poly   Input polygon (closed or non-closed).
 * @param center Center of scaling.
 * @param s      Positive scaling factor (must be > 0).
 * @return Scaled polygon, preserving closure if the input was closed.
 *
 * @details
 * Each vertex p is mapped to:
 *
 *   p' = center + s * (p - center)
 *
 * which corresponds to a uniform dilation about `center`.
 *
 * If the input polygon is explicitly closed
 * (`poly.front() == poly.back()`), the returned polygon is also closed.
 * Otherwise, the returned vertex list remains open.
 *
 * Complexity:
 *  - Time:  O(n)
 *  - Space: O(n)
 *
 * Preconditions:
 *  - s > 0
 *  - poly may be empty or small; no minimum size is required.
 *
 * Exceptions:
 *  - Throws std::runtime_error if s <= 0.
 *
 * @note
 * - A scaling factor s > 1 enlarges the polygon.
 * - A scaling factor 0 < s < 1 shrinks the polygon.
 * - Orientation (CW/CCW) is preserved for s > 0.
 *
 * Numerical notes:
 * - Closure detection uses exact floating-point comparison.
 *   If vertices were produced via floating-point operations,
 *   consider an epsilon-based closure test for robustness.
 */
static inline std::vector<Vec2> scale_polygon_about(
    const std::vector<Vec2>& poly, Vec2 center, double s)
{
    if (!(s > 0.0)) throw std::runtime_error("scale_polygon_about: scale must be > 0");
    std::vector<Vec2> out;
    out.reserve(poly.size());

    const bool closed =
        poly.size() >= 2 &&
        poly.front().x == poly.back().x &&
        poly.front().y == poly.back().y;

    const size_t n = poly.size();
    const size_t m = closed ? (n - 1) : n;

    for (size_t i = 0; i < m; ++i) {
        const double dx = poly[i].x - center.x;
        const double dy = poly[i].y - center.y;
        out.push_back({ center.x + s * dx, center.y + s * dy });
    }
    // close it if the input was closed
    if (closed) out.push_back(out.front());
    return out;
}
#pragma endregion

#pragma region Volume/Area computation

/**
 * @brief Computes the log-volume of the orthogonal group O(k).
 *
 * @param k Dimension parameter (must be > 0).
 * @return log(vol(O(k))) as a long double.
 *
 * @details
 * The (Haar) volume of the orthogonal group O(k) under the standard
 * Riemannian metric / normalization commonly used in random matrix theory
 * can be expressed in closed form via the multivariate gamma function:
 *
 *   vol(O(k)) = 2^k * π^{k^2/2} / Γ_k(k/2)
 *
 * Taking logs yields:
 *
 *   log vol(O(k)) = k log 2 + (k^2/2) log π - log Γ_k(k/2)
 *
 * This routine evaluates the expression in log-space for numerical stability
 * using long double arithmetic.
 *
 * Dependencies:
 *  - Requires `log_mvgamma(k, a)` (log multivariate gamma Γ_k(a)).
 *
 * Preconditions:
 *  - k > 0
 *  - The call to log_mvgamma uses a = k/2, which satisfies the domain
 *    requirement a > (k-1)/2 for all k >= 1.
 *
 * Exceptions:
 *  - Throws std::runtime_error if k <= 0.
 *
 * @note
 * - π is computed as `acosl(-1)` to avoid relying on non-standard `M_PI`.
 * - The exact normalization of "volume" depends on convention; this formula
 *   matches the Γ_k-based expression above.
 */
static inline long double log_volume_O(int k) {
    if (k <= 0) throw std::runtime_error("log_volume_O: k must be > 0");

    const long double pi = acosl(-1.0L);
    const long double logpi = logl(pi);
    const long double log2  = logl(2.0L);

    long double kk = (long double)k;
    long double a  = kk * 0.5L;

    return kk * log2 + (kk * kk * 0.5L) * logpi - log_mvgamma(k, a);
}

/**
 * @brief Computes the log-volume of the Grassmann manifold Gr(k, n).
 *
 * @param n Ambient dimension (must be > 0).
 * @param k Subspace dimension (must satisfy 0 <= k <= n).
 * @return log(vol(Gr(k, n))) as a long double.
 *
 * @details
 * The Grassmann manifold Gr(k, n) is the space of k-dimensional linear
 * subspaces of R^n. With the standard homogeneous-space identification:
 *
 *   Gr(k, n) ≅ O(n) / ( O(k) × O(n-k) )
 *
 * the corresponding (Haar / induced Riemannian) volume satisfies:
 *
 *   vol(Gr(k, n)) = vol(O(n)) / ( vol(O(k)) * vol(O(n-k)) )
 *
 * Therefore:
 *
 *   log vol(Gr(k, n)) = log vol(O(n)) - log vol(O(k)) - log vol(O(n-k))
 *
 * This routine computes the result entirely in log-space for numerical
 * stability via `log_volume_O`.
 *
 * Preconditions:
 *  - n > 0
 *  - 0 <= k <= n
 *
 * Special cases:
 *  - For k = 0 or k = n, Gr(k, n) consists of a single point, hence volume 1
 *    and log-volume 0.
 *
 * Exceptions:
 *  - Throws std::runtime_error if n <= 0.
 *  - Throws std::runtime_error if k is outside [0, n].
 *
 * Dependencies:
 *  - `log_volume_O(int)` must implement a consistent volume normalization for O(m)
 *    across all m used here.
 *
 * @note
 * - The exact numerical value depends on the normalization convention used for
 *   vol(O(m)). This function is consistent as long as the same convention is used
 *   in `log_volume_O` for all arguments.
 */
static inline long double log_grassmann_volume(int n, int k) {
    if (n <= 0) throw std::runtime_error("log_grassmann_volume: n must be > 0");
    if (k < 0 || k > n) throw std::runtime_error("log_grassmann_volume: require 0 <= k <= n");
    if (k == 0 || k == n) return 0.0L;
    return log_volume_O(n) - log_volume_O(k) - log_volume_O(n - k);
}

/**
 * @brief Computes the volume of the Grassmann manifold Gr(k, n) as a double.
 *
 * @param n Ambient dimension (must be > 0).
 * @param k Subspace dimension (must satisfy 0 <= k <= n).
 * @return vol(Gr(k, n)) as a double. Returns +inf on overflow and 0 on underflow.
 *
 * @details
 * This is a convenience wrapper around `log_grassmann_volume(n, k)` that
 * exponentiates the log-volume while guarding against double overflow/underflow.
 *
 * Let:
 *   lv = log vol(Gr(k, n))
 *
 * Then:
 *   vol = exp(lv)
 *
 * Since vol(Gr(k, n)) can be extremely large or small, this routine compares
 * `lv` against the log of the double representable range:
 *
 *   log_max = log(DBL_MAX)
 *   log_min = log(DBL_MIN)   (smallest *positive normal* double)
 *
 * and returns:
 *  - +∞ if lv > log_max
 *  - 0  if lv < log_min   (underflow to subnormal/zero is treated as 0)
 *  - exp(lv) otherwise
 *
 * Complexity:
 *  - Dominated by `log_grassmann_volume`.
 *
 * Exceptions:
 *  - Propagates any exceptions thrown by `log_grassmann_volume`.
 *
 * @note
 * - Underflow threshold uses `std::numeric_limits<double>::min()`, which is the
 *   smallest positive *normal* value (not denorm_min). Values smaller than that
 *   may still be representable as subnormals, but are treated here as 0 for
 *   simplicity/stability.
 * - Uses long double math internally and converts to double at the end.
 */
static inline double grassmann_volume(int n, int k) {
    long double lv = log_grassmann_volume(n, k);

    const long double log_max = logl((long double)std::numeric_limits<double>::max());
    const long double log_min = logl((long double)std::numeric_limits<double>::min());

    if (lv > log_max) return std::numeric_limits<double>::infinity();
    if (lv < log_min) return 0.0;

    return (double)expl(lv);
}

/**
 * @brief Computes the log surface area of the unit n-sphere S^n.
 *
 * @param n Dimension of the sphere (must be >= 0).
 * @return log( surface_area(S^n) ) as a long double.
 *
 * @details
 * The surface area of the unit n-sphere embedded in R^{n+1} is:
 *
 *   |S^n| = 2 * π^{(n+1)/2} / Γ((n+1)/2)
 *
 * Therefore:
 *
 *   log |S^n| = log(2) + ((n+1)/2) * log(π) - log Γ((n+1)/2)
 *
 * This implementation evaluates the expression in log-space using
 * long double arithmetic and `lgammal` for improved numerical stability.
 *
 * Examples:
 *  - n = 0: |S^0| = 2        (two points)
 *  - n = 1: |S^1| = 2π       (unit circle circumference)
 *  - n = 2: |S^2| = 4π       (unit sphere surface area)
 *
 * Preconditions:
 *  - n >= 0
 *
 * Exceptions:
 *  - Throws std::runtime_error if n < 0.
 *
 * @note
 * - π is computed via `acosl(-1)` to avoid reliance on non-standard `M_PI`.
 * - Uses the standard convention where S^n ⊂ R^{n+1}.
 * - For large n, computing the log surface area is numerically stable,
 *   while the surface area itself may overflow in double precision.
 */
static inline long double sphere_log_surface_area(int n) {
    if (n < 0) throw std::runtime_error("compute_sphere_log_surface_area: n must be >= 0");

    const long double pi = acosl(-1.0L);
    const long double logpi = logl(pi);
    const long double a = ((long double)n + 1.0L) * 0.5L;  // (n+1)/2

    return logl(2.0L) + a * logpi - lgammal(a);
}

/**
 * @brief Computes the surface area of the unit n-sphere S^n as a double.
 *
 * @param n Dimension of the sphere (must be >= 0).
 * @return Surface area |S^n| as a double.
 *
 * @details
 * This is a convenience wrapper around `sphere_log_surface_area(n)`.
 * It exponentiates the log surface area while guarding against
 * overflow and underflow in double precision.
 *
 * Let:
 *   lv = log |S^n|
 *
 * Then:
 *   |S^n| = exp(lv)
 *
 * Because |S^n| grows and then decays super-exponentially in n,
 * direct exponentiation may overflow or underflow in double precision.
 * Therefore we compare `lv` against:
 *
 *   log_max = log(DBL_MAX)
 *   log_min = log(DBL_MIN)   (smallest positive normal double)
 *
 * and return:
 *  - +∞ if lv > log_max
 *  - 0  if lv < log_min
 *  - exp(lv) otherwise
 *
 * Numerical notes:
 *  - Uses long double internally for improved stability.
 *  - Underflow is clamped to 0 instead of producing subnormals.
 *  - If `expl` is unavailable, `exp(lv)` may be used (with slightly
 *    reduced precision before casting to double).
 *
 * Exceptions:
 *  - Propagates any exception thrown by `sphere_log_surface_area`.
 */
static inline double sphere_surface_area(int n) {
    const long double lv = sphere_log_surface_area(n);

    const long double log_max = logl((long double)std::numeric_limits<double>::max());
    const long double log_min = logl((long double)std::numeric_limits<double>::min());

    if (lv > log_max) return std::numeric_limits<double>::infinity();
    if (lv < log_min) return 0.0;
    return (double)expl(lv); // if expl missing in your env, use exp(lv)
}

static inline long double log_unit_ball_volume(int n) {
    if (n < 0) throw std::runtime_error("logUnitBallVolume: n must be >= 0");
    if (n == 0) return 0.0L; // volume of 0-ball is 1

    const long double pi = acosl(-1.0L);
    const long double logpi = logl(pi);

    const long double a = (long double)n * 0.5L;     // n/2
    // log( pi^(n/2) / Gamma(n/2 + 1) ) = (n/2)*log(pi) - lgamma(n/2 + 1)
    return a * logpi - lgammal(a + 1.0L);
}

static inline double unit_ball_volume(int n) {
    long double lv = log_unit_ball_volume(n);

    const long double log_max = logl((long double)std::numeric_limits<double>::max());
    const long double log_min = logl((long double)std::numeric_limits<double>::min());

    if (lv > log_max) return std::numeric_limits<double>::infinity();
    if (lv < log_min) return 0.0;
    return (double)expl(lv);
}

#pragma endregion

#pragma region Sampling

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

#pragma endregion

#pragma region MST computation

template<int dim, int p>
std::tuple<int, double> call_euclidean_mst_grassmann(
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

template<int dim, int p>
std::tuple<int, double> call_euclidean_mst(
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
    } else if (shape == "kochband") {
        fill_from_koch_snowflake_band_2d(pts, num, koch_depth, /*thickness=*/50e-2);
        write_pts_as_npy_xy(pts, num, "/tmp/kochband_pts.npy");
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
std::tuple<int, double> call_euclidean_mst_from_file(std::string inputFile) {
    int rows;
    int cols;
    
    Eigen::MatrixXd X = load_npy_to_eigen_row_major(inputFile);
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
std::tuple<int, double> dispatch_euclidean_mst(
    int numPoints,
    const std::string& shape,
    const std::string& inputFile,
    int gr_n,
    int gr_k,
    bool sphere_geodesic,
    int koch_depth
) {
    if (!inputFile.empty()) {
        return call_euclidean_mst_from_file<dim,p>(inputFile);
    }
    if (shape == "grassmann") {
        return call_euclidean_mst_grassmann<dim,p>(numPoints, gr_n, gr_k);
    }
    return call_euclidean_mst<dim,p>(numPoints, shape, sphere_geodesic, koch_depth);
}

using DispatchFn = std::tuple<int, double>(*)(
    int, const std::string&, const std::string&, int, int, bool, int
);

template<int dim, int p>
std::tuple<int, double> dispatch_helper(
    int numPoints,
    const std::string& shape,
    const std::string& inputFile,
    int gr_n,
    int gr_k,
    bool sphere_geodesic,
    int koch_depth
) {
    return dispatch_euclidean_mst<dim, p>(
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
        {&dispatch_helper<2, 1>, &dispatch_helper<2, 2>, &dispatch_helper<2, 3>, &dispatch_helper<2, 4>, &dispatch_helper<2, 5>},
        {&dispatch_helper<3, 1>, &dispatch_helper<3, 2>, &dispatch_helper<3, 3>, &dispatch_helper<3, 4>, &dispatch_helper<3, 5>},
        {&dispatch_helper<4, 1>, &dispatch_helper<4, 2>, &dispatch_helper<4, 3>, &dispatch_helper<4, 4>, &dispatch_helper<4, 5>},
        {&dispatch_helper<5, 1>, &dispatch_helper<5, 2>, &dispatch_helper<5, 3>, &dispatch_helper<5, 4>, &dispatch_helper<5, 5>},
        {&dispatch_helper<6, 1>, &dispatch_helper<6, 2>, &dispatch_helper<6, 3>, &dispatch_helper<6, 4>, &dispatch_helper<6, 5>},
        {&dispatch_helper<7, 1>, &dispatch_helper<7, 2>, &dispatch_helper<7, 3>, &dispatch_helper<7, 4>, &dispatch_helper<7, 5>},
        {&dispatch_helper<8, 1>, &dispatch_helper<8, 2>, &dispatch_helper<8, 3>, &dispatch_helper<8, 4>, &dispatch_helper<8, 5>},
        {&dispatch_helper<9, 1>, &dispatch_helper<9, 2>, &dispatch_helper<9, 3>, &dispatch_helper<9, 4>, &dispatch_helper<9, 5>},
        {&dispatch_helper<10, 1>, &dispatch_helper<10, 2>, &dispatch_helper<10, 3>, &dispatch_helper<10, 4>, &dispatch_helper<10, 5>},
        {&dispatch_helper<11, 1>, &dispatch_helper<11, 2>, &dispatch_helper<11, 3>, &dispatch_helper<11, 4>, &dispatch_helper<11, 5>},
        {&dispatch_helper<12, 1>, &dispatch_helper<12, 2>, &dispatch_helper<12, 3>, &dispatch_helper<12, 4>, &dispatch_helper<12, 5>},
        {&dispatch_helper<13, 1>, &dispatch_helper<13, 2>, &dispatch_helper<13, 3>, &dispatch_helper<13, 4>, &dispatch_helper<13, 5>},
        {&dispatch_helper<14, 1>, &dispatch_helper<14, 2>, &dispatch_helper<14, 3>, &dispatch_helper<14, 4>, &dispatch_helper<14, 5>},
        {&dispatch_helper<15, 1>, &dispatch_helper<15, 2>, &dispatch_helper<15, 3>, &dispatch_helper<15, 4>, &dispatch_helper<15, 5>},
        {&dispatch_helper<16, 1>, &dispatch_helper<16, 2>, &dispatch_helper<16, 3>, &dispatch_helper<16, 4>, &dispatch_helper<16, 5>},
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
                volume = grassmann_volume(gr_n, gr_k);
            }
            intdim = gr_k * (gr_n - gr_k);
        } else if (shape == "sphere") {
            if (dim < 2) throw std::runtime_error("Sphere requires dim >= 2");
                intdim = dim - 1;
            if (!std::isfinite(volume)) {
                volume = sphere_surface_area(dim - 1); 
            }
        } else if (shape == "ball") {
            if (!std::isfinite(volume)) {
                volume = unit_ball_volume(dim);
            }
            intdim = dim;
        } else if (shape == "koch") {
            if (dim != 2) throw std::runtime_error("--shape koch requires --dim 2");
            if (koch_depth < 0) throw std::runtime_error("--koch_depth must be >= 0");
            auto poly = koch_polygon(koch_depth);
            volume = polygon_area(poly);  
            intdim = dim;  
        } else if (shape == "kochband") {
            if (dim != 2) throw std::runtime_error("--shape kochband requires --dim 2");
            if (koch_depth < 0) throw std::runtime_error("--koch_depth must be >= 0");
            auto inner = koch_polygon(koch_depth);
            const Vec2 c = polygon_centroid_area(inner);
            const double s = 1.0 + 50e-2; // thickness parameter
            auto outer = scale_polygon_about(inner, c, s);
            volume = polygon_area(outer) - polygon_area(inner);
            intdim = dim;
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

        write_to_database(dbFile, numPoints, mst_length, mst_length_normalized);

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return -1;
    }
    return 0;
}
