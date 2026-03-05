#ifndef SYSTEMATIC_FRACTAL_GEOMETRY_H
#define SYSTEMATIC_FRACTAL_GEOMETRY_H

#include <vector>           
#include "euclideanMst/custom/koch_geometry.h"

std::vector<Vec2> systematic_polygon(int degree, int depth, bool closed = true);

#endif