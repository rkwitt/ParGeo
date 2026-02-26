#pragma once

using namespace std;

enum class Shape {
    Cube,       // unit cube [0,1]^d
    Ball,       // unit ball B_d(0,1)
    Sphere,     // unit sphere S^{d-1}(0,1)
    Koch,       // Koch snowflake
    KochBand,   // Band between two Koch snowflakes
    Grassmann   // Grassmann manifold
};


struct MstConfig {
    int num_points = 0;
    Shape shape = Shape::Cube;
    std::string input_file;
    int gr_n = -1;
    int gr_k = -1;
    bool use_sphere_geodesic = false;
    int koch_depth = -1;
    double koch_band_thickness = -1.0;
    bool use_triangulation_file = false; 
    std::string triangulation_file;
};


static inline Shape parse_shape(const std::string& s) {
    if (s == "cube")      return Shape::Cube;
    if (s == "ball")      return Shape::Ball;
    if (s == "sphere")    return Shape::Sphere;
    if (s == "koch")      return Shape::Koch;
    if (s == "kochband")  return Shape::KochBand;
    if (s == "grassmann") return Shape::Grassmann;
    throw std::runtime_error("Unknown shape: " + s + " (expected cube|ball|sphere|grassmann|koch|kochband)");
}