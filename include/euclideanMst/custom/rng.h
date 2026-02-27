#ifndef EUCLIDEAN_MST_CUSTOM_RNG_H
#define EUCLIDEAN_MST_CUSTOM_RNG_H

#include <cstdint>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>

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

#endif