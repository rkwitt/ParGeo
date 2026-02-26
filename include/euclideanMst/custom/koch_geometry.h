#pragma once

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cmath>
#include "cnpy.h" 

using namespace std;

struct Vec2 { double x, y; };

struct EdgeTable {
    std::vector<double> x1, y1, x2, y2;   ///< Edge endpoints in SoA form.
    std::vector<double> inv_dy;           ///< 1/(y2-y1) for non-horizontal edges.
    std::vector<uint8_t> active;          ///< 1 if edge is non-horizontal; else 0.
    double xmin, xmax, ymin, ymax;        ///< Axis-aligned bounding box of all edges.
};


inline Vec2 operator+(const Vec2& a, const Vec2& b) { return {a.x + b.x, a.y + b.y}; }
inline Vec2 operator-(const Vec2& a, const Vec2& b) { return {a.x - b.x, a.y - b.y}; }
inline Vec2 operator*(const Vec2& a, double s) { return {a.x * s, a.y * s}; }

std::vector<Vec2> koch_polygon(int depth);
EdgeTable build_edge_table(const std::vector<Vec2>& poly);
bool point_in_poly_fast(double x, double y, const EdgeTable& E);
double polygon_area(const std::vector<Vec2>& poly);
void write_koch_polygon_as_npy_xy(int depth, const std::string& path);
std::vector<Vec2> scale_polygon_about(const std::vector<Vec2>& poly, Vec2 center, double s);
Vec2 polygon_centroid_area(const std::vector<Vec2>& poly);

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
inline Vec2 rotate(const Vec2& v, double ang) {
    double c = std::cos(ang), s = std::sin(ang);
    return {c*v.x - s*v.y, s*v.x + c*v.y};
}


