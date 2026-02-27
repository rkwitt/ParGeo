#ifndef EUCLIDEAN_MST_CUSTOM_SPHERE_H
#define EUCLIDEAN_MST_CUSTOM_SPHERE_H

#include <algorithm>   
#include <cmath>       
#include <limits>      
#include <stdexcept>

namespace emstExtension::custom {

inline double sphere_geodesic_from_chord(double chord) noexcept {
    // chord should be in [0,2] for unit sphere; clamp for numerical safety
    if (!std::isfinite(chord)) return std::numeric_limits<double>::quiet_NaN();
    double x = 0.5 * chord;
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return M_PI; // antipodal (or clamp)
    return 2.0 * std::asin(x);
}

inline double sphere_geodesic_from_chord_atan2(double chord) noexcept {
    if (!std::isfinite(chord)) return std::numeric_limits<double>::quiet_NaN();
    double c = std::clamp(chord, 0.0, 2.0);
    double dot = 1.0 - 0.5 * c * c;   // cos(theta)
    double sin_theta = std::sqrt(std::max(0.0, 1.0 - dot * dot));
    return std::atan2(sin_theta, dot);
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
inline long double sphere_log_surface_area(int n) {
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
 *
 * Exceptions:
 *  - Propagates any exception thrown by `sphere_log_surface_area`.
 */
inline double sphere_surface_area(int n) {
    const long double lv = sphere_log_surface_area(n);

    const long double log_max = logl((long double)std::numeric_limits<double>::max());
    const long double log_min = logl((long double)std::numeric_limits<double>::min());

    if (lv > log_max) return std::numeric_limits<double>::infinity();
    if (lv < log_min) return 0.0;
    return (double)expl(lv);
}

}

#endif // EUCLIDEAN_MST_CUSTOM_SPHERE_H