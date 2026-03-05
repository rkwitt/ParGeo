#ifndef TRI_SAMPLER_H
#define TRI_SAMPLER_H

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "parlay/parallel.h"

#include <CGAL/Constrained_triangulation_plus_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_2.h>
#include <CGAL/Constrained_triangulation_face_base_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Triangulation_data_structure_2.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/enum.h> // for CGAL::ON_POSITIVE_SIDE etc. (optional)

#include "euclideanMst/custom/rng.h"
#include "euclideanMst/custom/config.h"
#include "euclideanMst/custom/koch_geometry.h"
#include "euclideanMst/custom/systematic_fractal_geometry.h"

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

// using CDTBase = CGAL::Constrained_Delaunay_triangulation_2<K, TDS>;
// using CDT     = CGAL::Constrained_triangulation_plus_2<CDTBase>;

struct TriSampler {
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

static constexpr uint32_t TRI_MAGIC = 0x4B4F4348; // 'KOCH'
static constexpr uint32_t TRI_VER   = 1;

void mark_domains(CDT& cdt);
void flood_fill_component(CDT& cdt, CDT::Face_handle start, int level, std::deque<CDT::Face_handle>& border);
double tri_area(double ax, double ay, double bx, double by, double cx, double cy);
void write_sampler_binary(const TriSampler& S, const std::string& path);
TriSampler read_sampler_binary(const std::string& path);
bool file_exists(const std::string& path);
TriSampler build_trisampler(int depth);
TriSampler build_polygon_trisampler(const std::vector<Vec2>& poly_closed);

template<typename T>
void fill_from_koch_snowflake_2d_cdt(T& pts, const MstConfig& cfg) {
    if (cfg.num_points <= 0) return;
    if (cfg.koch_depth < 0) throw std::runtime_error("depth must be >= 0");

    static std::mutex mtx;
    static int cached_depth = -1;
    static std::string cached_path;
    static TriSampler sampler;

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
                sampler = build_trisampler(cfg.koch_depth);

                if (cfg.use_triangulation_file && !path.empty()) {
                    try {
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

template<typename T>
void fill_from_systematic_fractal_2d_cdt(T& pts, const MstConfig& cfg) {
    if (cfg.num_points <= 0) return;
    if (cfg.sys_depth < 1)  throw std::runtime_error("sys_depth must be >= 1");
    if (cfg.sys_degree < 1) throw std::runtime_error("sys_degree must be >= 1");
    if ((cfg.sys_degree % 2) == 0)
        throw std::runtime_error("systematic sampler: only odd degrees are currently supported.");

    static std::mutex mtx;
    static int cached_depth = -1;
    static int cached_degree = -1;
    static std::string cached_path;
    static TriSampler sampler;

    auto resolve_cache_path = [&](int degree, int depth) -> std::string {
        if (!cfg.use_triangulation_file) return {};
        if (!cfg.triangulation_file.empty()) return cfg.triangulation_file;
        return "/tmp/systematic_tris_deg_" + std::to_string(degree) + "_depth_" + std::to_string(depth) + ".bin";
    };

    const std::string path = resolve_cache_path(cfg.sys_degree, cfg.sys_depth);

    {
        std::lock_guard<std::mutex> lock(mtx);

        const bool need_rebuild =
            (cached_depth  != cfg.sys_depth)  ||
            (cached_degree != cfg.sys_degree) ||
            (cached_path   != path);

        if (need_rebuild) {
            bool loaded = false;

            if (cfg.use_triangulation_file && file_exists(path)) {
                sampler = read_sampler_binary(path);
                loaded = true;
            }

            if (!loaded) {
                auto poly = systematic_polygon(cfg.sys_degree, cfg.sys_depth, /*closed=*/true);
                sampler = build_polygon_trisampler(poly);

                if (cfg.use_triangulation_file && !path.empty()) {
                    try { write_sampler_binary(sampler, path); }
                    catch (const std::exception& e) {
                        std::cerr << "[systematic] warning: could not write triangulation cache: " << e.what() << "\n";
                    }
                }
            }

            cached_depth  = cfg.sys_depth;
            cached_degree = cfg.sys_degree;
            cached_path   = path;
        }
    }

    if (sampler.empty()) throw std::runtime_error("systematic sampler is empty");

    parlay::parallel_for(0, cfg.num_points, [&](int i) {
        auto& rng = tls_rng();
        double x, y;
        sampler.sample(rng, x, y);
        pts[i].x[0] = x;
        pts[i].x[1] = y;
    });
}

#endif // TRI_SAMPLER_H