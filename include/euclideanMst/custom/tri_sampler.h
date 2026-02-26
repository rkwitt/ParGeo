#pragma once

#include <vector>
#include <stdexcept>

// CGAL includes for triangulating the Koch snowflake
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_2.h>
#include <CGAL/Constrained_triangulation_face_base_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Triangulation_data_structure_2.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/enum.h> // for CGAL::ON_POSITIVE_SIDE etc. (optional)

#include "euclideanMst/custom/koch_geometry.h"

struct Tri2 {
    double ax, ay;
    double bx, by;
    double cx, cy;
    double area; // positive
};

struct FaceInfo2 {
    int nesting_level = -1; // -1 means "unvisited"
    bool in_domain() const { return nesting_level % 2 == 1; }
};

using K  = CGAL::Exact_predicates_inexact_constructions_kernel;
using P2 = K::Point_2;
using Vb  = CGAL::Triangulation_vertex_base_2<K>;
using Fbb = CGAL::Constrained_triangulation_face_base_2<K>;
using Fb  = CGAL::Triangulation_face_base_with_info_2<FaceInfo2, K, Fbb>;
using TDS = CGAL::Triangulation_data_structure_2<Vb, Fb>;
using CDT = CGAL::Constrained_Delaunay_triangulation_2<K, TDS>;

struct KochTriSampler {
    std::vector<Tri2> tris;
    std::vector<double> prefix; // prefix[i] = sum_{j<=i} area_j
    double total_area = 0.0;

    bool empty() const { return tris.empty() || total_area <= 0.0; }

    // Sample one point uniformly from the polygon.
    template <class RNG>
    inline void sample(RNG& rng, double& outx, double& outy) const {
        // Choose triangle proportional to area.
        const double u = rng.next_double01() * total_area;

        // binary search prefix
        auto it = std::lower_bound(prefix.begin(), prefix.end(), u);
        size_t idx = (it == prefix.end()) ? (prefix.size() - 1) : (size_t)(it - prefix.begin());
        const Tri2& T = tris[idx];

        // Uniform point in triangle via barycentric transform:
        // r1,r2 ~ U(0,1), sqrt trick.
        double r1 = rng.next_double01();
        double r2 = rng.next_double01();
        double sr1 = std::sqrt(r1);

        double wA = 1.0 - sr1;
        double wB = sr1 * (1.0 - r2);
        double wC = sr1 * r2;

        outx = wA * T.ax + wB * T.bx + wC * T.cx;
        outy = wA * T.ay + wB * T.by + wC * T.cy;
    }
};

void mark_domains(CDT& cdt);
void flood_fill_component(CDT& cdt, CDT::Face_handle start, int level, std::deque<CDT::Face_handle>& border);
double tri_area(double ax, double ay, double bx, double by, double cx, double cy);
KochTriSampler read_sampler_binary(const std::string& path);

static constexpr uint32_t KOCHTRI_MAGIC = 0x4B4F4348; // 'KOCH'
static constexpr uint32_t KOCHTRI_VER   = 1;

static KochTriSampler build_koch_trisampler(int depth) {
    // 1) Build polygon boundary vertices (your existing function)
    std::vector<Vec2> poly = koch_polygon(depth);
    if (poly.size() < 4) throw std::runtime_error("koch_polygon produced too few vertices");

    // Ensure it's closed (your koch_polygon already returns closed)
    if (!(poly.front().x == poly.back().x && poly.front().y == poly.back().y)) {
        poly.push_back(poly.front());
    }

    // 2) Build CDT + insert constraints
    CDT cdt;

    // Insert constraints for each boundary segment (i -> i+1)
    // IMPORTANT: CDT constraint insertion can handle repeated vertices, but it's
    // cleaner to avoid the duplicate closing point when iterating segments.
    const size_t n = poly.size();
    for (size_t i = 0; i + 1 < n; ++i) {
        const P2 a(poly[i].x,   poly[i].y);
        const P2 b(poly[i+1].x, poly[i+1].y);
        cdt.insert_constraint(a, b);
    }

    // 3) Mark inside faces
    mark_domains(cdt);

    // 4) Extract triangles (finite faces that are inside)
    KochTriSampler S;
    S.tris.reserve((size_t)cdt.number_of_faces()); // rough

    for (auto f = cdt.finite_faces_begin(); f != cdt.finite_faces_end(); ++f) {
        if (!f->info().in_domain()) continue;

        const P2 p0 = f->vertex(0)->point();
        const P2 p1 = f->vertex(1)->point();
        const P2 p2 = f->vertex(2)->point();

        Tri2 t;
        t.ax = CGAL::to_double(p0.x()); t.ay = CGAL::to_double(p0.y());
        t.bx = CGAL::to_double(p1.x()); t.by = CGAL::to_double(p1.y());
        t.cx = CGAL::to_double(p2.x()); t.cy = CGAL::to_double(p2.y());
        t.area = tri_area(t.ax, t.ay, t.bx, t.by, t.cx, t.cy);

        // Degenerate faces should not happen, but skip if area ~ 0
        if (t.area <= 0.0) continue;

        S.tris.push_back(t);
    }

    if (S.tris.empty()) {
        throw std::runtime_error("CDT produced no interior faces; check polygon validity / constraints.");
    }

    // 5) Prefix sums of areas
    S.prefix.resize(S.tris.size());
    double acc = 0.0;
    for (size_t i = 0; i < S.tris.size(); ++i) {
        acc += S.tris[i].area;
        S.prefix[i] = acc;
    }
    S.total_area = acc;

    return S;
}
