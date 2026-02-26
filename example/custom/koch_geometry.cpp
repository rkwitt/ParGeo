#include "euclideanMst/custom/koch_geometry.h"

/**
 * @brief Builds an edge table (structure-of-arrays) from a closed polygon.
 *
 * Converts a closed polygonal chain into an `EdgeTable` representation suitable
 * for scanline / ray-crossing style computations (e.g., point-in-polygon tests,
 * winding/crossing counts, rasterization-like passes).
 *
 * The input is expected to be *closed*, i.e. `poly.front() == poly.back()`,
 * so that edges are formed as:
 *
 *   edge i: poly[i] -> poly[i+1]  for i = 0 .. n-2
 *
 * Horizontal edges (dy == 0) are marked inactive and can be skipped by
 * downstream crossing tests to avoid degeneracies.
 *
 * @param poly Closed polygon vertex list.
 * @return Fully populated edge table, including bounding box.
 *
 * Preconditions:
 *  - poly.size() >= 3
 *  - poly is closed (poly[0] equals poly[n-1]) if you intend the last-to-first
 *    edge to be excluded. This function assumes closure and uses n-1 edges.
 *
 * Postconditions / invariants:
 *  - All per-edge arrays in the returned `EdgeTable` have the same length m = n-1.
 *  - For each edge i:
 *      (x1[i], y1[i]) = poly[i]
 *      (x2[i], y2[i]) = poly[i+1]
 *  - active[i] == 0 iff dy == 0 (horizontal edge), otherwise active[i] == 1.
 *  - inv_dy[i] == 1/dy for active edges; inv_dy[i] == 0 for inactive edges.
 *  - xmin/xmax/ymin/ymax bound all polygon vertices.
 *
 * Exceptions:
 *  - Throws std::runtime_error if the polygon has fewer than 3 vertices.
 *
 * @note
 * Using exact equality `dy == 0.0` treats only perfectly horizontal edges as
 * inactive. If vertices may contain floating-point noise, we should instead consider 
 * replacing this check with an epsilon comparison (e.g. |dy| < eps).
 */
EdgeTable build_edge_table(const std::vector<Vec2>& poly) {
    EdgeTable E;
    const size_t n = poly.size();
    if (n < 3) throw std::runtime_error("Koch polygon too small");
    const size_t m = n - 1; // edges from i -> i+1 (poly closed)

    E.x1.resize(m); E.y1.resize(m);
    E.x2.resize(m); E.y2.resize(m);
    E.inv_dy.resize(m);
    E.active.resize(m);

    E.xmin = E.xmax = poly[0].x;
    E.ymin = E.ymax = poly[0].y;

    for (size_t i = 0; i < n; ++i) {
        E.xmin = std::min(E.xmin, poly[i].x);
        E.xmax = std::max(E.xmax, poly[i].x);
        E.ymin = std::min(E.ymin, poly[i].y);
        E.ymax = std::max(E.ymax, poly[i].y);
    }

    for (size_t i = 0; i < m; ++i) {
        double ax = poly[i].x, ay = poly[i].y;
        double bx = poly[i+1].x, by = poly[i+1].y;
        E.x1[i] = ax; E.y1[i] = ay;
        E.x2[i] = bx; E.y2[i] = by;

        double dy = by - ay;
        if (dy == 0.0) {
            E.active[i] = 0;   // horizontal edge: skip in crossing test
            E.inv_dy[i] = 0.0;
        } else {
            E.active[i] = 1;
            E.inv_dy[i] = 1.0 / dy;
        }
    }
    return E;
}

/**
 * @brief Fast point-in-polygon test using a precomputed edge table.
 *
 * Tests whether the query point (x, y) lies inside a (simple) polygon using
 * the ray-crossing (even–odd) rule. This implementation casts a horizontal ray
 * to +infinity in x-direction and toggles the inside state on each edge
 * intersection.
 *
 * @param x Query x-coordinate.
 * @param y Query y-coordinate.
 * @param E Precomputed edge table (typically from `build_edge_table`).
 * @return True if the point is inside the polygon according to the even–odd rule,
 *         false otherwise.
 *
 * @details
 * For each non-horizontal edge i, we:
 *  1. Perform a straddle test to check whether the edge crosses the horizontal
 *     line y = query_y:
 *
 *       (y1 > y) != (y2 > y)
 *
 *     This convention excludes hits exactly at vertices in a consistent way
 *     and avoids double-counting shared endpoints when combined with skipping
 *     horizontal edges.
 *
 *  2. Compute the x-coordinate of the intersection of the edge with y = query_y:
 *
 *       x_int = x1 + (x2 - x1) * ((y - y1) / (y2 - y1))
 *
 *     Using `inv_dy[i] = 1/(y2 - y1)` avoids a division in the inner loop.
 *
 *  3. Toggle the parity if the intersection lies to the right of the query point
 *     (`x_int >= x`), which corresponds to the ray from (x, y) to +∞.
 *
 *
 * Preconditions:
 *  - E’s arrays have consistent lengths and represent edges (as produced by
 *    `build_edge_table`).
 *  - E.active[i] == 0 for horizontal edges, and E.inv_dy[i] is valid for active edges.
 *
 * @note Boundary behavior:
 * - This routine implements a strict even–odd crossing test. Points exactly on
 *   the polygon boundary (on an edge or vertex) are not handled explicitly and
 *   may return either true or false depending on floating-point rounding and
 *   the chosen inequality (`>=`).
 * - If you need a robust “inside-or-on-boundary” predicate, add an explicit
 *   segment-distance / colinearity check before the parity test.
 *
 * Numerical notes:
 * - Uses exact comparisons for the straddle test. For noisy inputs, consider
 *   epsilon handling for y comparisons.
 */
bool point_in_poly_fast(double x, double y, const EdgeTable& E) {
    bool inside = false;
    const size_t m = E.x1.size();
    for (size_t i = 0; i < m; ++i) {
        if (!E.active[i]) continue;
        double y1 = E.y1[i], y2 = E.y2[i];

        // straddle check
        bool c = (y1 > y) != (y2 > y);
        if (!c) continue;

        // x intersection (no divide: multiply by inv_dy)
        double x1 = E.x1[i], x2 = E.x2[i];
        double xint = x1 + (x2 - x1) * ((y - y1) * E.inv_dy[i]);
        if (xint >= x) inside = !inside;
    }
    return inside;
}

/**
 * @brief Computes the (unsigned) area of a closed polygon using the shoelace formula.
 *
 * @param poly Vertex list of a polygonal chain.
 * @return Non-negative area of the polygon.
 *
 * @details
 * Uses the classical shoelace (Gauss area) formula:
 *
 *   A = 1/2 * | Σ (x_i y_{i+1} − x_{i+1} y_i) |
 *
 * This implementation assumes the polygon is explicitly closed
 * (i.e. `poly.front() == poly.back()`), so edges are formed as:
 *
 *   (poly[i] -> poly[i+1]) for i = 0 .. n-2
 *
 * The absolute value is taken, so the result is orientation-independent.
 *
 * Preconditions:
 *  - If `poly` is closed, the last vertex equals the first.
 *  - If not closed, the last-to-first edge is not included in this implementation.
 *
 * @note
 * - If you require the signed area (positive for CCW, negative for CW),
 *   remove `std::abs`.
 * - For self-intersecting polygons, this computes the algebraic area,
 *   not necessarily the geometric union area.
 */
double polygon_area(const std::vector<Vec2>& poly) {
    const size_t n = poly.size();
    if (n < 3) return 0.0;
    double a = 0.0;
    for (size_t i = 0; i + 1 < n; ++i) {
        a += poly[i].x * poly[i+1].y - poly[i+1].x * poly[i].y;
    }
    // if not explicitly closed, you'd also add last->first; here poly is closed
    return 0.5 * std::abs(a);
}

void write_koch_polygon_as_npy_xy(int depth, const std::string& path) {
    const std::vector<Vec2> poly = koch_polygon(depth);     // includes closing vertex
    const size_t m = poly.size();

    std::vector<double> buf;
    buf.resize(m * 2);

    for (size_t i = 0; i < m; ++i) {
        buf[i * 2 + 0] = poly[i].x;
        buf[i * 2 + 1] = poly[i].y;
    }
    cnpy::npy_save(path, buf.data(), {m, size_t(2)}, "w");
}

/**
 * @brief Computes the area centroid (center of mass) of a simple polygon.
 *
 * @param poly Vertex list of a polygon (closed or non-closed).
 * @return The centroid of the polygon’s interior (uniform density).
 *
 * @details
 * Uses the standard area-centroid formula derived from the shoelace identity:
 *
 *   A2 = Σ (x_i y_{i+1} − x_{i+1} y_i)          (twice signed area)
 *
 *   Cx = (1 / (6A)) Σ (x_i + x_{i+1}) * cross_i
 *   Cy = (1 / (6A)) Σ (y_i + y_{i+1}) * cross_i
 *
 * Since A2 = 2A, we use:
 *
 *   inv6A = 1 / (3 * A2)
 *
 * and compute:
 *
 *   Cx = Cx_sum * inv6A
 *   Cy = Cy_sum * inv6A
 *
 * The algorithm works for both closed polygons
 * (`poly.front() == poly.back()`) and open vertex lists.
 *
 * Numerical considerations:
 *  - Accumulation is performed in long double to reduce cancellation error.
 *  - The result is returned as double.
 *
 * Preconditions:
 *  - Polygon must have at least 3 distinct vertices.
 *  - Polygon must have non-zero signed area.
 *  - Polygon is assumed to be simple (non-self-intersecting).
 *
 * Exceptions:
 *  - Throws std::runtime_error if polygon is too small.
 *  - Throws std::runtime_error if the polygon is degenerate (zero area).
 *
 * @note
 * - The sign of A2 depends on vertex orientation (CCW positive, CW negative).
 *   The centroid formula remains correct because the same sign appears in
 *   numerator and denominator.
 * - For self-intersecting polygons, this computes the algebraic centroid,
 *   not necessarily the geometric centroid of the union region.
 */
Vec2 polygon_centroid_area(const std::vector<Vec2>& poly) {
    const size_t n = poly.size();
    if (n < 3) throw std::runtime_error("polygon_centroid_area: poly too small");

    const size_t m = (n >= 2 && poly.front().x == poly.back().x && poly.front().y == poly.back().y) ? (n - 1) : n;
    if (m < 3) throw std::runtime_error("polygon_centroid_area: need at least 3 distinct vertices");

    long double A2 = 0.0L; // 2*Area signed
    long double Cx = 0.0L;
    long double Cy = 0.0L;

    for (size_t i = 0; i < m; ++i) {
        const Vec2& p = poly[i];
        const Vec2& q = poly[(i + 1) % m];
        const long double cross = (long double)p.x * (long double)q.y - (long double)q.x * (long double)p.y;
        A2 += cross;
        Cx += ((long double)p.x + (long double)q.x) * cross;
        Cy += ((long double)p.y + (long double)q.y) * cross;
    }

    if (A2 == 0.0L) throw std::runtime_error("polygon_centroid_area: degenerate polygon (zero area)");

    const long double inv6A = 1.0L / (3.0L * A2); // since centroid uses 1/(6A) and A2=2A => 1/(3*A2)
    Vec2 c;
    c.x = (double)(Cx * inv6A);
    c.y = (double)(Cy * inv6A);
    return c;
}

/**
 * @brief Uniformly scales a polygon about a specified center point.
 *
 * @param poly   Input polygon (closed or non-closed).
 * @param center Center of scaling.
 * @param s      Positive scaling factor (must be > 0).
 * @return Scaled polygon, preserving closure if the input was closed.
 *
 * @details
 * Each vertex p is mapped to:
 *
 *   p' = center + s * (p - center)
 *
 * which corresponds to a uniform dilation about `center`.
 *
 * If the input polygon is explicitly closed
 * (`poly.front() == poly.back()`), the returned polygon is also closed.
 * Otherwise, the returned vertex list remains open.
 *
 * Complexity:
 *  - Time:  O(n)
 *  - Space: O(n)
 *
 * Preconditions:
 *  - s > 0
 *  - poly may be empty or small; no minimum size is required.
 *
 * Exceptions:
 *  - Throws std::runtime_error if s <= 0.
 *
 * @note
 * - A scaling factor s > 1 enlarges the polygon.
 * - A scaling factor 0 < s < 1 shrinks the polygon.
 * - Orientation (CW/CCW) is preserved for s > 0.
 *
 * Numerical notes:
 * - Closure detection uses exact floating-point comparison.
 *   If vertices were produced via floating-point operations,
 *   consider an epsilon-based closure test for robustness.
 */
std::vector<Vec2> scale_polygon_about(
    const std::vector<Vec2>& poly, Vec2 center, double s)
{
    if (!(s > 0.0)) throw std::runtime_error("scale_polygon_about: scale must be > 0");
    std::vector<Vec2> out;
    out.reserve(poly.size());

    const bool closed =
        poly.size() >= 2 &&
        poly.front().x == poly.back().x &&
        poly.front().y == poly.back().y;

    const size_t n = poly.size();
    const size_t m = closed ? (n - 1) : n;

    for (size_t i = 0; i < m; ++i) {
        const double dx = poly[i].x - center.x;
        const double dy = poly[i].y - center.y;
        out.push_back({ center.x + s * dx, center.y + s * dy });
    }
    // close it if the input was closed
    if (closed) out.push_back(out.front());
    return out;
}

/**
 * @brief Constructs a closed polygonal approximation of the Koch snowflake boundary.
 *
 * Generates the Koch curve starting from an equilateral triangle and applies
 * `depth` refinement steps. The returned vertex list is a *closed* polyline:
 * the last vertex equals the first.
 *
 * @param depth Number of refinement iterations (must be >= 0).
 * @return Vertices of the closed polygonal chain representing the Koch boundary.
 *
 * @details
 * Initialization:
 *  - Starts with an equilateral triangle with vertices:
 *      (0,0), (1,0), (0.5, sqrt(3)/2)
 *  - The polyline is explicitly closed by repeating the first vertex at the end.
 *
 * Refinement rule (per edge p1 -> p2):
 *  - Split the edge into thirds: diff = (p2 - p1) / 3.
 *  - Construct points:
 *      a = p1
 *      b = p1 + diff
 *      e = p1 + 2*diff
 *  - Insert a “spike” point:
 *      c = b + rotate(diff, -pi/3)
 *
 * Each iteration replaces every segment by 4 segments, so the number of edges
 * grows by a factor of 4 per depth level.
 *
 * @note
 * - Angle units are radians.
 * - The sign of the rotation angle determines whether the spikes point inward
 *   or outward relative to the initial triangle orientation. This implementation
 *   uses -pi/3.
 */
std::vector<Vec2> koch_polygon(int depth) {
    std::vector<Vec2> v = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.5, std::sqrt(3.0)/2.0},
        {0.0, 0.0} // close
    };
    const double ang = -M_PI/3.0;

    for (int d = 0; d < depth; ++d) {
        std::vector<Vec2> nv;
        nv.reserve((v.size()-1)*4 + 1);
        for (size_t i = 0; i + 1 < v.size(); ++i) {
            Vec2 p1 = v[i], p2 = v[i+1];
            Vec2 diff = (p2 - p1) * (1.0/3.0);
            Vec2 rotv = rotate(diff, ang);

            Vec2 a = p1;
            Vec2 b = p1 + diff;
            Vec2 c = b + rotv;
            Vec2 e = b + diff;

            nv.push_back(a);
            nv.push_back(b);
            nv.push_back(c);
            nv.push_back(e);
        }
        nv.push_back(nv.front());
        v.swap(nv);
    }
    return v;
}