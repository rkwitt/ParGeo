#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>

#include "euclideanMst/custom/systematic_fractal_geometry.h"
#include "euclideanMst/custom/koch_geometry.h"

static inline Vec2 rot90(const Vec2& v) { return {-v.y, v.x}; }

// Matches the Python SystematicFractal.generator(p1,p2).
static std::vector<Vec2> systematic_edge(const Vec2& p1, const Vec2& p2, int degree) {
    if ((degree % 2) == 0) {
        throw std::runtime_error("systematic_edge: only odd degrees are currently supported (use degree=1,3,5,...).");
    }

    const int denom = (degree + 1 + (degree % 2)) * 2; // exactly like python: /(deg+1+deg%2)/2
    const Vec2 right = (p2 - p1) * (1.0 / (double)denom);
    const Vec2 left  = right * (-1.0);
    const Vec2 up    = rot90(right);
    const Vec2 down  = up * (-1.0);

    std::vector<Vec2> pts;
    pts.reserve(64);
    pts.push_back(p1);
    
    pts.push_back(pts.back() + right);
    pts.push_back(pts.back() + up);
    pts.push_back(pts.back() + right);
    pts.push_back(pts.back() + down);
    pts.push_back(pts.back() + right);

    for (int k = 1; k < degree; ++k) {
        if (k % 2 == 1) {
            for (int t = 0; t < k + 1; ++t) pts.push_back(pts.back() + up);
            for (int t = 0; t < k + 1; ++t) pts.push_back(pts.back() + left);
            pts.push_back(pts.back() + up);
            for (int t = 0; t < k + 2; ++t) pts.push_back(pts.back() + right);
            for (int t = 0; t < k + 2; ++t) pts.push_back(pts.back() + down);
            pts.push_back(pts.back() + right);
        }
    }

    pts.push_back(pts.back() + right);
    pts.push_back(pts.back() + down);
    pts.push_back(pts.back() + right);
    pts.push_back(pts.back() + up);
    pts.push_back(pts.back() + right);

    for (int k = 1; k < degree; ++k) {
        if (k % 2 == 1) {
            for (int t = 0; t < k + 1; ++t) pts.push_back(pts.back() + down);
            for (int t = 0; t < k + 1; ++t) pts.push_back(pts.back() + left);
            pts.push_back(pts.back() + down);
            for (int t = 0; t < k + 2; ++t) pts.push_back(pts.back() + right);
            for (int t = 0; t < k + 2; ++t) pts.push_back(pts.back() + up);
            pts.push_back(pts.back() + right);
        }
    }

    // Python: new_points.pop(-1) to avoid duplicating segment endpoints
    if (!pts.empty()) pts.pop_back();
    return pts;
}

std::vector<Vec2> systematic_polygon(int degree, int depth, bool closed) {
    if (degree < 1) throw std::runtime_error("systematic_polygon: degree must be >= 1");
    if ((degree % 2) == 0)
        throw std::runtime_error("systematic_polygon: only odd degrees are currently supported (use degree=1,3,5,...)");
    if (depth  < 1) throw std::runtime_error("systematic_polygon: depth must be >= 1");

    std::vector<Vec2> V = {{0,0},{1,0},{1,1},{0,1}};

    for (int d = 0; d < depth; ++d) {
        std::vector<Vec2> next;
        next.reserve(V.size() * 4);

        for (size_t i = 0; i < V.size(); ++i) {
            const Vec2& p1 = V[i];
            const Vec2& p2 = V[(i + 1) % V.size()];
            auto seg = systematic_edge(p1, p2, degree);
            next.insert(next.end(), seg.begin(), seg.end());
        }

        // remove consecutive duplicates (defensive)
        next.erase(std::unique(next.begin(), next.end(),
            [](const Vec2& a, const Vec2& b){
                return std::abs(a.x - b.x) < 1e-15 && std::abs(a.y - b.y) < 1e-15;
            }), next.end());

        if (next.size() < 3) throw std::runtime_error("systematic_polygon: degenerate polygon");
        V = std::move(next);
    }

    if (closed) {
        if (!(V.front().x == V.back().x && V.front().y == V.back().y)) {
            V.push_back(V.front());
        }
    }
    return V;
}
