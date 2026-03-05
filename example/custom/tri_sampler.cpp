#include "euclideanMst/custom/tri_sampler.h"

#include <filesystem>

bool file_exists(const std::string& path) {
    return !path.empty() && std::filesystem::exists(path);
}

void mark_domains(CDT& cdt) {
    // Reset
    for (auto f = cdt.all_faces_begin(); f != cdt.all_faces_end(); ++f) {
        f->info().nesting_level = -1;
    }

    std::deque<CDT::Face_handle> border;
    flood_fill_component(cdt, cdt.infinite_face(), 0, border);

    // Process border components; each time we cross a constraint, nesting level increments.
    while (!border.empty()) {
        CDT::Face_handle fh = border.front();
        border.pop_front();

        if (fh->info().nesting_level != -1) continue;

        // Determine nesting level from already-visited neighbors across constrained edges
        // We can just set it to 1 + min visited neighbor nesting level, but the typical
        // CGAL example does: level = neighbor_level + 1 across constraint.
        // Here, we find any adjacent visited face across a constrained edge:
        int level = 1;
        bool found = false;

        for (int i = 0; i < 3 && !found; ++i) {
            CDT::Face_handle nb = fh->neighbor(i);
            if (nb->info().nesting_level == -1) continue;
            // if the shared edge is constrained, we crossed a boundary
            if (cdt.is_constrained(CDT::Edge(fh, i))) {
                level = nb->info().nesting_level + 1;
                found = true;
            }
        }

        // If not found (rare), fall back to level 1
        flood_fill_component(cdt, fh, level, border);
    }
}

void flood_fill_component(CDT& cdt,
    CDT::Face_handle start,
    int level,
    std::deque<CDT::Face_handle>& border)
{
    std::deque<CDT::Face_handle> q;
    q.push_back(start);

    while (!q.empty()) {
        CDT::Face_handle fh = q.front();
        q.pop_front();

        if (fh->info().nesting_level != -1) continue; // already visited
        fh->info().nesting_level = level;

        // Explore neighbors
        for (int i = 0; i < 3; ++i) {
            CDT::Face_handle nb = fh->neighbor(i);
            if (nb->info().nesting_level != -1) continue;

            // If edge (fh,i) is constrained, we don't cross it in this component;
            // instead, add neighbor to border list to process at higher nesting level.
            if (cdt.is_constrained(CDT::Edge(fh, i))) {
                border.push_back(nb);
            } else {
                q.push_back(nb);
            }
        }
    }
}

double tri_area(double ax, double ay, double bx, double by, double cx, double cy) {
    // 0.5 * |cross(b-a, c-a)|
    const double abx = bx - ax, aby = by - ay;
    const double acx = cx - ax, acy = cy - ay;
    const double cross = abx * acy - aby * acx;
    return 0.5 * std::abs(cross);
}

void write_sampler_binary(const TriSampler& S, const std::string& path) {
    if (path.empty()) throw std::runtime_error("write_sampler_binary: empty path");
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    // write to temp then rename (avoid partial files)
    std::string tmp = path + ".tmp." + std::to_string(::getpid());

    std::ofstream out(tmp, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open file for write: " + tmp);

    uint32_t magic = TRI_MAGIC;
    uint32_t ver   = TRI_VER;
    uint64_t ntri  = (uint64_t)S.tris.size();

    out.write((char*)&magic, sizeof(magic));
    out.write((char*)&ver,   sizeof(ver));
    out.write((char*)&ntri,  sizeof(ntri));
    out.write((char*)&S.total_area, sizeof(S.total_area));

    // store Tri2 as raw doubles (portable enough across same arch; see note below)
    for (const auto& t : S.tris) {
        out.write((char*)&t.ax, sizeof(double));
        out.write((char*)&t.ay, sizeof(double));
        out.write((char*)&t.bx, sizeof(double));
        out.write((char*)&t.by, sizeof(double));
        out.write((char*)&t.cx, sizeof(double));
        out.write((char*)&t.cy, sizeof(double));
        out.write((char*)&t.area, sizeof(double));
    }

    out.close();
    if (!out) throw std::runtime_error("Write failed: " + tmp);

    std::filesystem::rename(tmp, path);
}

TriSampler read_sampler_binary(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file for read: " + path);

    uint32_t magic = 0, ver = 0;
    uint64_t ntri = 0;
    double total_area = 0.0;

    in.read((char*)&magic, sizeof(magic));
    in.read((char*)&ver,   sizeof(ver));
    in.read((char*)&ntri,  sizeof(ntri));
    in.read((char*)&total_area, sizeof(total_area));

    if (!in) throw std::runtime_error("Corrupt header: " + path);
    if (magic != TRI_MAGIC) throw std::runtime_error("Bad magic (not a koch tri file): " + path);
    if (ver != TRI_VER) throw std::runtime_error("Unsupported version: " + std::to_string(ver));

    TriSampler S;
    S.tris.resize((size_t)ntri);
    S.total_area = total_area;

    for (size_t i = 0; i < (size_t)ntri; ++i) {
        Tri2 t;
        in.read((char*)&t.ax, sizeof(double));
        in.read((char*)&t.ay, sizeof(double));
        in.read((char*)&t.bx, sizeof(double));
        in.read((char*)&t.by, sizeof(double));
        in.read((char*)&t.cx, sizeof(double));
        in.read((char*)&t.cy, sizeof(double));
        in.read((char*)&t.area, sizeof(double));
        if (!in) throw std::runtime_error("Corrupt tri payload: " + path);
        S.tris[i] = t;
    }

    // rebuild prefix
    S.prefix.resize(S.tris.size());
    double acc = 0.0;
    for (size_t i = 0; i < S.tris.size(); ++i) {
        acc += S.tris[i].area;
        S.prefix[i] = acc;
    }
    // trust-but-verify total_area
    S.total_area = acc;

    if (S.tris.empty() || !(S.total_area > 0.0))
        throw std::runtime_error("Loaded empty/invalid sampler: " + path);

    return S;
}

TriSampler build_trisampler(int depth) {
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
    TriSampler S;
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

TriSampler build_polygon_trisampler(const std::vector<Vec2>& poly_in) {
    std::vector<Vec2> poly = poly_in;
    if (poly.size() < 4) throw std::runtime_error("build_polygon_trisampler: polygon too small");

    // Ensure closed
    if (!(poly.front().x == poly.back().x && poly.front().y == poly.back().y)) {
        poly.push_back(poly.front());
    }

    CGAL::Polygon_2<K> P;

    for (size_t i = 0; i + 1 < poly.size(); ++i) {  // skip duplicate last vertex
        P.push_back(P2(poly[i].x, poly[i].y));
    }

    if (!P.is_simple()) {
        throw std::runtime_error(
            "Polygon is not simple (self-intersects). "
            "Try smaller depth/degree or enable intersection handling.");
    }

    CDT cdt;

    // Insert constraints
    for (size_t i = 0; i + 1 < poly.size(); ++i) {
        const P2 a(poly[i].x,   poly[i].y);
        const P2 b(poly[i+1].x, poly[i+1].y);
        cdt.insert_constraint(a, b);
    }

    // Mark interior faces
    mark_domains(cdt);

    TriSampler S;
    S.tris.reserve((size_t)cdt.number_of_faces());

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
        if (t.area <= 0.0) continue;
        S.tris.push_back(t);
    }

    if (S.tris.empty()) {
        throw std::runtime_error("CDT produced no interior faces; polygon may be invalid/self-intersecting.");
    }

    // Prefix sums
    S.prefix.resize(S.tris.size());
    double acc = 0.0;
    for (size_t i = 0; i < S.tris.size(); ++i) {
        acc += S.tris[i].area;
        S.prefix[i] = acc;
    }
    S.total_area = acc;
    return S;
}
