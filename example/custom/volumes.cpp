#include <cmath>
#include <limits>
#include <stdexcept>

#include "euclideanMst/custom/volumes.h"

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
long double log_mvgamma(int k, long double a) {
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
long double log_volume_O(int k) {
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
long double log_grassmann_volume(int n, int k) {
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
double grassmann_volume(int n, int k) {
    long double lv = log_grassmann_volume(n, k);

    const long double log_max = logl((long double)std::numeric_limits<double>::max());
    const long double log_min = logl((long double)std::numeric_limits<double>::min());

    if (lv > log_max) return std::numeric_limits<double>::infinity();
    if (lv < log_min) return 0.0;

    return (double)expl(lv);
}

long double log_unit_ball_volume(int n) { 
    if (n < 0) throw std::runtime_error("logUnitBallVolume: n must be >= 0"); 
    if (n == 0) return 0.0L; // volume of 0-ball is 1 
    
    const long double pi = acosl(-1.0L); 
    const long double logpi = logl(pi); 
    const long double a = (long double)n * 0.5L; 
    
    // n/2 // log( pi^(n/2) / Gamma(n/2 + 1) ) = (n/2)*log(pi) - lgamma(n/2 + 1) 
    return a * logpi - lgammal(a + 1.0L); 
}

double unit_ball_volume(int n) {
    long double lv = log_unit_ball_volume(n);

    const long double log_max = logl((long double)std::numeric_limits<double>::max());
    const long double log_min = logl((long double)std::numeric_limits<double>::min());

    if (lv > log_max) return std::numeric_limits<double>::infinity();
    if (lv < log_min) return 0.0;
    return (double)expl(lv);
}
