#include "euclideanMst/custom/tri_sampler.h"

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

KochTriSampler read_sampler_binary(const std::string& path) {
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
    if (magic != KOCHTRI_MAGIC) throw std::runtime_error("Bad magic (not a koch tri file): " + path);
    if (ver != KOCHTRI_VER) throw std::runtime_error("Unsupported version: " + std::to_string(ver));

    KochTriSampler S;
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