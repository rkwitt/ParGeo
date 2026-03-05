#include <iostream>
#include <algorithm>
#include <filesystem>
#include <random>
#include <cassert>
#include <iomanip>
#include <tuple>
#include <atomic>
#include <vector>
#include <numbers>
#include <fmt/core.h>
#include <fmt/color.h>
#include <cmath>
#include <stdexcept>
#include <deque>
#include <mutex>

#include <boost/program_options.hpp>
#include <Eigen/Dense>

#include "euclideanMst/custom/koch_geometry.h"
#include "euclideanMst/custom/volumes.h"
#include "euclideanMst/custom/sphere.h"
#include "euclideanMst/custom/npy.h"
#include "euclideanMst/custom/sql.h"
#include "euclideanMst/custom/rng.h"
#include "euclideanMst/custom/config.h"
#include "euclideanMst/custom/tri_sampler.h"
#include "euclideanMst/custom/samplers.h"

#include "euclideanMst/euclideanMst.h"
#include "parlay/utilities.h"
#include "pargeo/point.h"


namespace po = boost::program_options;
using namespace std;
using namespace parlay;
using namespace pargeo;

using DispatchFn = std::tuple<int, double>(*)(const MstConfig&);

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


template<int dim, int p, typename PointSeq>
static inline double mst_p_sum(
    PointSeq& pts,
    const MstConfig& cfg
) {
    static_assert(p >= 0, "p must be >= 0");
    static_assert(dim >= 2, "dim must be >= 2");
    
    // call to Pargeo's MST implementation
    auto I = euclideanMst<dim>(pts);
    auto S = pts.data();

    double sum = 0.0;
    double c = 0.0;

    for (auto e : I) {
        double d = S[e.u].dist(S[e.v]);

        if (cfg.use_sphere_geodesic) {
            d = emstExtension::custom::sphere_geodesic_from_chord_atan2(d);
        }

        const double pd = std::pow(d, p);

        const double t = sum + pd;
        if (std::abs(sum) >= std::abs(pd)) c += (sum - t) + pd;
        else                               c += (pd - t) + sum;
        sum = t;
    }

    return sum + c;
}

template<int dim>
void fill_points_from_config(
    parlay::sequence<pargeo::point<dim>>& pts,
    const MstConfig& cfg
) {
    if (!cfg.input_file.empty()) {
        Eigen::MatrixXd X = load_npy_to_eigen_row_major(cfg.input_file);
        if (X.cols() != dim) throw std::runtime_error("Dimension mismatch: file has cols != dim");

        pts = parlay::sequence<pargeo::point<dim>>(static_cast<size_t>(X.rows()));

        for (int i = 0; i < X.rows(); ++i) {
            for (int k = 0; k < dim; ++k) {
                pts[i].x[k] = X(i, k);
            }
        }
        return;
    }

    if (cfg.num_points <= 0) {
        throw std::runtime_error("num_points must be > 0 (unless input_file is provided)");
    }

    pts = parlay::sequence<pargeo::point<dim>>(static_cast<size_t>(cfg.num_points));

    switch (cfg.shape) {
        case Shape::Cube:
            fill_from_uniform_in_unitbox<dim, decltype(pts), double>(pts, cfg.num_points);
            return;
        case Shape::Ball:
            fill_from_uniform_in_unitball<dim, decltype(pts), double>(pts, cfg.num_points);
            return;
        case Shape::Sphere:
            fill_from_uniform_on_unitsphere<dim, decltype(pts), double>(pts, cfg.num_points);
            return;
        case Shape::Koch:
            if (cfg.koch_depth < 0) throw std::runtime_error("koch_depth must be >= 0");
            fill_from_koch_snowflake_2d_cdt(pts, cfg);
            return;
        case Shape::Systematic:
            fill_from_systematic_fractal_2d_cdt(pts, cfg);
            return;
        case Shape::Grassmann:
            if (cfg.gr_n <= 0) throw std::runtime_error("shape=grassmann requires gr_n > 0");
            if (cfg.gr_k < 0 || cfg.gr_k > cfg.gr_n) throw std::runtime_error("shape=grassmann requires 0 <= gr_k <= gr_n");
            if (dim != cfg.gr_n * cfg.gr_n) throw std::runtime_error("For grassmann projection embedding require dim == gr_n*gr_n");
            fill_from_grassmann_svd_projection<dim, decltype(pts), double>(
                pts, cfg.num_points, cfg.gr_n, cfg.gr_k
            );
            return;
    }
    throw std::runtime_error("Unsupported shape in config");
}

template<int dim, int p>
static std::tuple<int, double> run_mst(const MstConfig& cfg) {
    parlay::sequence<pargeo::point<dim>> pts;
    fill_points_from_config<dim>(pts, cfg);
    const double sum = mst_p_sum<dim, p>(pts, cfg);
    return { static_cast<int>(pts.size()), sum };
}

template<int dim, int p>
static std::tuple<int, double> dispatch_helper(const MstConfig& cfg) {
    return run_mst<dim, p>(cfg);
}

static std::tuple<int, double> dispatch(int dim, int p, const MstConfig& cfg) {
    // Validate supported template instantiations
    if (dim < 2 || dim > 16 || p < 1 || p > 5) {
        throw std::invalid_argument("Unsupported dimension or p value");
    }

    static const DispatchFn dispatchTable[15][5] = {
        {&dispatch_helper<2, 1>,  &dispatch_helper<2, 2>,  &dispatch_helper<2, 3>,  &dispatch_helper<2, 4>,  &dispatch_helper<2, 5>},
        {&dispatch_helper<3, 1>,  &dispatch_helper<3, 2>,  &dispatch_helper<3, 3>,  &dispatch_helper<3, 4>,  &dispatch_helper<3, 5>},
        {&dispatch_helper<4, 1>,  &dispatch_helper<4, 2>,  &dispatch_helper<4, 3>,  &dispatch_helper<4, 4>,  &dispatch_helper<4, 5>},
        {&dispatch_helper<5, 1>,  &dispatch_helper<5, 2>,  &dispatch_helper<5, 3>,  &dispatch_helper<5, 4>,  &dispatch_helper<5, 5>},
        {&dispatch_helper<6, 1>,  &dispatch_helper<6, 2>,  &dispatch_helper<6, 3>,  &dispatch_helper<6, 4>,  &dispatch_helper<6, 5>},
        {&dispatch_helper<7, 1>,  &dispatch_helper<7, 2>,  &dispatch_helper<7, 3>,  &dispatch_helper<7, 4>,  &dispatch_helper<7, 5>},
        {&dispatch_helper<8, 1>,  &dispatch_helper<8, 2>,  &dispatch_helper<8, 3>,  &dispatch_helper<8, 4>,  &dispatch_helper<8, 5>},
        {&dispatch_helper<9, 1>,  &dispatch_helper<9, 2>,  &dispatch_helper<9, 3>,  &dispatch_helper<9, 4>,  &dispatch_helper<9, 5>},
        {&dispatch_helper<10, 1>, &dispatch_helper<10, 2>, &dispatch_helper<10, 3>, &dispatch_helper<10, 4>, &dispatch_helper<10, 5>},
        {&dispatch_helper<11, 1>, &dispatch_helper<11, 2>, &dispatch_helper<11, 3>, &dispatch_helper<11, 4>, &dispatch_helper<11, 5>},
        {&dispatch_helper<12, 1>, &dispatch_helper<12, 2>, &dispatch_helper<12, 3>, &dispatch_helper<12, 4>, &dispatch_helper<12, 5>},
        {&dispatch_helper<13, 1>, &dispatch_helper<13, 2>, &dispatch_helper<13, 3>, &dispatch_helper<13, 4>, &dispatch_helper<13, 5>},
        {&dispatch_helper<14, 1>, &dispatch_helper<14, 2>, &dispatch_helper<14, 3>, &dispatch_helper<14, 4>, &dispatch_helper<14, 5>},
        {&dispatch_helper<15, 1>, &dispatch_helper<15, 2>, &dispatch_helper<15, 3>, &dispatch_helper<15, 4>, &dispatch_helper<15, 5>},
        {&dispatch_helper<16, 1>, &dispatch_helper<16, 2>, &dispatch_helper<16, 3>, &dispatch_helper<16, 4>, &dispatch_helper<16, 5>}
    };

    return dispatchTable[dim - 2][p - 1](cfg);
}

static void print_colored_help(const po::options_description& all) {
    using fmt::print;
    using fmt::fg;
    using fmt::color;
    using fmt::emphasis;

    print(fg(color::blue) | emphasis::bold,
        "MST Computation Options\n");
    print(fg(color::dark_gray),
        "-------------------------------------------------------------------------------------\n\n");

    for (const auto& opt : all.options()) {
        std::string name = opt->long_name();
        std::string desc = opt->description();
        print(fg(color::green) | emphasis::bold,
            "  --{:20}", name);
        print(fg(color::white),
            " {}\n", desc);
    }
    print("\n");
}

static void print_shape_requirements() {
    std::cout
        << "Shape-specific requirements:\n"
        << "  cube/ball:    require --dim, and either --num_points or --input_file\n"
        << "  sphere:       require --dim>=2; optional --use_sphere_geodesic\n"
        << "  koch:         require --dim 2 and --koch_depth>=0\n"
        << "  systematic:   require --dim 2 and --sys_degree>=1 and --sys_depth>=1\n"
        << "  grassmann:    require --gr_n>0, 0<=--gr_k<=--gr_n, and --dim == gr_n*gr_n\n";
}

static MstConfig parse_cli(int argc, char* argv[], std::string& db_file, int& dim, int& p)
{
    std::string input_file;
    std::string shape_str;

    int num_points = -1;
    int gr_n = -1;
    int gr_k = -1;
    int koch_depth = 5;
    bool use_sphere_geodesic = false;
    bool use_triangulation_file = false;
    int sys_degree = -1;
    int sys_depth = -1;
    std::string triangulation_file;

    po::options_description base("Base options");
    base.add_options()
        ("help,h", "Produce help message")
        ("dim", po::value<int>(&dim)->required(), "Dimensionality of input vectors (R^d).")
        ("p", po::value<int>(&p)->default_value(1), "Power of edge lengths.")
        ("shape", po::value<std::string>(&shape_str)->required(), "cube|ball|sphere|grassmann|koch")
        ("db_file", po::value<std::string>(&db_file)->required(), "SQLite3 database file.")
        ("input_file", po::value<std::string>(&input_file)->default_value(""), "Numpy matrix input file.")
        ("num_points", po::value<int>(&num_points)->default_value(-1),
            "Number of points to sample (required if no --input_file).");

    po::options_description sphere_opts("Sphere options");
    sphere_opts.add_options()
        ("use_sphere_geodesic",
        po::bool_switch(&use_sphere_geodesic)->default_value(false),
        "For shape=sphere: convert chord length to geodesic distance.");

    po::options_description grassmann_opts("Grassmann options");
    grassmann_opts.add_options()
        ("gr_n", po::value<int>(&gr_n)->default_value(-1), "For shape=grassmann: ambient dimension n (R^n).")
        ("gr_k", po::value<int>(&gr_k)->default_value(-1), "For shape=grassmann: subspace dimension k.");

    po::options_description systematic_opts("Systematic fractal options");
    grassmann_opts.add_options()
        ("sys_degree", po::value<int>(&sys_degree)->default_value(1), "For shape=systematic: degree of the fractal.")
        ("sys_depth", po::value<int>(&sys_depth)->default_value(5), "For shape=systematic: recursion depth.");

    po::options_description koch_opts("Koch fractal options");
    koch_opts.add_options()
        ("koch_depth", po::value<int>(&koch_depth)->default_value(5), "For shape=koch: recursion depth.");

    po::options_description fractal_opts("General fractal options");
    koch_opts.add_options()
        ("use_triangulation_file",
        po::bool_switch(&use_triangulation_file)->default_value(false),
        "For shape=koch|systematic: enable disk cache for triangulation.")
        ("triangulation_file",
        po::value<std::string>(&triangulation_file)->default_value(""),
        "For shape=koch|systematic: triangulation cache path (default /tmp/...).");

    po::options_description all("Options");
    all.add(base).add(sphere_opts).add(grassmann_opts).add(systematic_opts).add(fractal_opts).add(koch_opts);

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, all), vm);

        if (vm.count("help")) {
            print_colored_help(all);            
            print_shape_requirements();
            std::cout << "\n";
            std::exit(0);
        }

        po::notify(vm);
    } catch (const po::error& ex) {
        std::cerr << "CLI error: " << ex.what() << "\n\n" << all << "\n";
        print_shape_requirements();
        std::cerr << "\n";
        std::exit(1);
    }
    auto require = [&](bool cond, const char* msg) {
        if (!cond) throw po::error(msg);
    };

    Shape sh;
    try {
        sh = parse_shape(shape_str);
    } catch (const std::exception& e) {
        throw po::error(e.what());
    }

    require(!input_file.empty() || num_points > 0,
            "Either --input_file must be provided or --num_points must be > 0.");

    if (sh == Shape::Sphere) {
        require(dim >= 2, "--shape sphere requires --dim >= 2.");
    }

    if (sh == Shape::Koch) {
        require(dim == 2, "--shape koch requires --dim 2.");
        require(koch_depth > 0, "--shape koch requires --koch_depth > 0.");
    }

    if (sh == Shape::Systematic) {
        require(dim == 2, "--shape systematic requires --dim 2.");
        require(sys_degree >= 1, "--shape systematic requires --sys_degree >= 1.");
        require(sys_depth  >= 1, "--shape systematic requires --sys_depth >= 1.");
        require((sys_degree % 2) == 1,
                "--shape systematic currently supports only odd degrees (use --sys_degree 1,3,5,...).");
    }

    if (sh == Shape::Grassmann) {
        require(gr_n > 2, "--shape grassmann requires --gr_n > 2.");
        require(gr_k > 0 && gr_k < gr_n, "--shape grassmann requires 0 < --gr_k < --gr_n.");
        require(dim == gr_n * gr_n, "--shape grassmann requires --dim == gr_n*gr_n.");
    }

    MstConfig cfg;
    cfg.num_points = num_points;
    cfg.input_file = input_file;
    cfg.shape = sh;
    cfg.gr_n = gr_n;
    cfg.gr_k = gr_k;
    cfg.use_sphere_geodesic = use_sphere_geodesic;
    cfg.koch_depth = koch_depth;
    cfg.use_triangulation_file = use_triangulation_file;
    cfg.triangulation_file = triangulation_file;
    cfg.sys_degree = sys_degree;
    cfg.sys_depth = sys_depth;

    return cfg;
}

int main(int argc, char* argv[]) 
{
    int dim = -1;
    int p = 1;
    std::string db_file;

    MstConfig cfg;
    try {
        cfg = parse_cli(argc, argv, db_file, dim, p);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << std::scientific << std::setprecision(12);

    try {
        const auto result = dispatch(dim, p, cfg);
        const int n_used = std::get<0>(result);
        const double mst_length = std::get<1>(result);

        int intdim = -1;
        double volume = std::numeric_limits<double>::quiet_NaN();
        if (cfg.shape == Shape::Grassmann) {
            if (!std::isfinite(volume)) volume = grassmann_volume(cfg.gr_n, cfg.gr_k);
            intdim = cfg.gr_k * (cfg.gr_n - cfg.gr_k);
        } else if (cfg.shape == Shape::Sphere) {
            if (!std::isfinite(volume)) volume = emstExtension::custom::sphere_surface_area(dim - 1);
            intdim = dim - 1;
        } else if (cfg.shape == Shape::Ball) {
            if (!std::isfinite(volume)) volume = unit_ball_volume(dim);
            intdim = dim;
        } else if (cfg.shape == Shape::Koch) {
            const auto poly = koch_polygon(cfg.koch_depth);
            volume = polygon_area(poly);
            intdim = 2;
        } else if (cfg.shape == Shape::Systematic) {
            const auto poly = systematic_polygon(cfg.sys_degree, cfg.sys_depth, /*closed=*/true);
            volume = polygon_area(poly);
            intdim = 2;
        } else if (cfg.shape == Shape::Cube) {
            if (!std::isfinite(volume)) volume = 1.0;
            intdim = dim;
        } else {
            throw std::runtime_error("Unsupported shape in config");
        }

        const double mst_length_normalized = stable_normalized_mst(mst_length, n_used, p, intdim, volume);
        fmt::print(
            "| n={:>8d} | volume={:>20.16f} | dim={:>2d} | intdim={:>2d} | L_n={:>22.12f} | L_n (norm)={:>22.16f}\n",
            n_used, volume, dim, intdim, mst_length, 
            fmt::styled(mst_length_normalized,fmt::fg(fmt::color::green) | fmt::emphasis::bold)
        );

        write_to_database(db_file, n_used, mst_length, mst_length_normalized);

    } catch (const std::exception& e) {
        std::cerr << "Runtime error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
