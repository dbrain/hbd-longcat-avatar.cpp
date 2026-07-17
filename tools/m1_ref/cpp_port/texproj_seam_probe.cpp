// texproj_seam_probe — MEASURE the front<->back seam, on the CPU, with no atlas bake and no GPU.
//
// Why this exists: the owner's remaining texture complaint is "the crossover front to back has
// 'defects' (colours dont blend perfectly etc.) ... be nice if it didn't feel like there was a seam".
// Before changing the blend, measure WHAT the mismatch is: a global exposure/WB offset, a gradient,
// or genuinely different content. That decides the fix.
//
//   ./texproj_seam_probe <mesh.glb> <front.png> <back.png> [out_prefix]
//
// e.g. ./texproj_seam_probe $AS/_shootout_out/inline_soldier1536/refined.glb \
//                           $AS/_shootout_out/soldier_matte.png \
//                           $AS/_shootout_out/proj_v1/back_v1_rgba.png  /tmp/seam/s
//
// Uses the MESH VERTICES as texel proxies: project_onto's colour decision is a function of a 3D point +
// its normal ONLY (the atlas layout never enters it), so a per-vertex probe measures exactly the same
// thing an atlas texel would, at 165k samples, without a 5.6s bake. Runs the REAL texproj code — same
// Cam, raster_depth, visible(), subject_mask, fit_similarity, erode_mask, sample_bilinear, PointGrid,
// knn_gated — so it cannot drift from what project_onto will do.
//
// Writes <out_prefix>_verts.csv and <out_prefix>_pairs.csv; all statistics are done downstream.
// Build: ./build.sh texproj_seam_probe
#include "tex_project.hpp"
#include "glb_reader.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace texproj;

// Per-view sample of one 3D point, exactly as project_onto would take it.
struct Probe {
    bool  vis = false;      // passed the z-test (frontmost surface for this camera)
    bool  in_mask = false;  // sample lands inside the ERODED subject mask
    float col[3] = {0, 0, 0};   // sRGB [0,1] bilinear sample
    float facing = 0.f;
};

static Probe sample_view(const View& v, const float* P, const float* N, float eps) {
    Probe pr;
    float u, vv;
    const float d = v.cam.project(P, u, vv);
    pr.facing = v.cam.facing(N);
    pr.vis = visible(v.z, u, vv, d, eps);
    if (!pr.vis) return pr;
    float su = u, sv = vv;
    if (v.fit.fitted) {
        float ix, iy; v.fit.apply(u * v.img.w, vv * v.img.h, ix, iy);
        su = ix / (float)v.img.w; sv = iy / (float)v.img.h;
    }
    const int ix = (int)std::floor(su * (float)v.img.w);
    const int iy = (int)std::floor(sv * (float)v.img.h);
    pr.in_mask = !(ix < 0 || iy < 0 || ix >= v.img.w || iy >= v.img.h ||
                   (!v.subj.empty() && !v.subj[(size_t)iy * v.img.w + ix]));
    sample_bilinear(v.img, su, sv, pr.col);
    return pr;
}

// ---------------------------------------------------------------------------------------------
// SEAM_REAL_ATLAS=1 — drive the REAL texproj::project_onto on the CPU.
//
// Everything else in this file re-implements project_onto's decisions to measure them, which is fine for
// measuring but is NOT a test of the shipped function. The only thing project_onto needs that we lack is
// a UV atlas, and the atlas layout provably cannot change its output (a texel's colour is a function of
// its 3D position + normal alone). So: synthesize one. Give every triangle its own square cell in a grid
// — which is exactly what the precluster atlas already is (2999-4403 charts, median 4-7 verts: confetti).
//
// This runs the real projection, the real 3D fill, the real seam metric and the real debug dumps, on the
// real mesh, with no GPU and no bake. It is the only CPU-side proof that project_onto works before the
// owner spends a 3060 slot on it.
static void build_synthetic_atlas(const glb::Mesh& m, const std::vector<uint32_t>& faces,
                                  texatlas::BakedTexture& bt, int texsize) {
    const size_t F = faces.size() / 3;
    const int cells = std::max(1, (int)std::ceil(std::sqrt((double)F)));
    const float cw = 1.f / (float)cells;          // cell size in NORMALIZED uv (bt.uvs are normalized)
    bt.tw = texsize; bt.th = texsize;
    bt.verts.resize(F * 9); bt.normals.resize(F * 9); bt.uvs.resize(F * 6); bt.faces.resize(F * 3);
    for (size_t t = 0; t < F; t++) {
        const int cx = (int)(t % (size_t)cells), cy = (int)(t / (size_t)cells);
        // a right triangle inset inside the cell, so neighbouring cells never share a texel
        const float u0 = (cx + 0.15f) * cw, v0 = (cy + 0.15f) * cw, s = 0.70f * cw;
        const float uv[3][2] = {{u0, v0}, {u0 + s, v0}, {u0, v0 + s}};
        for (int k = 0; k < 3; k++) {
            const uint32_t src = faces[t * 3 + k];
            const size_t d = t * 3 + k;
            for (int c = 0; c < 3; c++) {
                bt.verts[d * 3 + c]   = m.verts[(size_t)src * 3 + c];
                bt.normals[d * 3 + c] = m.normals[(size_t)src * 3 + c];
            }
            bt.uvs[d * 2 + 0] = uv[k][0]; bt.uvs[d * 2 + 1] = uv[k][1];
            bt.faces[d] = (uint32_t)d;
        }
    }
    bt.base_color.assign((size_t)bt.tw * bt.th * 4, 0);
    bt.metal_rough.assign((size_t)bt.tw * bt.th * 4, 0);
    for (size_t i = 0; i < (size_t)bt.tw * bt.th; i++) bt.base_color[i * 4 + 3] = 255;
    std::printf("synthetic atlas: %dx%d, %zu tris -> %d x %d cells (%.1f px each), %zu verts\n",
                bt.tw, bt.th, F, cells, cells, cw * texsize, bt.verts.size() / 3);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <mesh.glb> <front.png> <back.png> [out_prefix]\n", argv[0]);
        return 2;
    }
    const std::string mesh_path = argv[1], front_path = argv[2], back_path = argv[3];
    const std::string outp = argc > 4 ? argv[4] : "";

    const float eps   = envf("TEXPROJ_DEPTH_EPS", 0.004f);
    const float nz_lo = envf("TEXPROJ_NZ_LO", 0.05f);
    const float nz_hi = envf("TEXPROJ_NZ_HI", 0.35f);
    const int   erode = std::max(0, envi("TEXPROJ_BG_ERODE", 2));
    const float front_thr = envf("TEXPROJ_FRONT_BG_THRESH", 1.f / 255.f);
    const float back_thr  = envf("TEXPROJ_BACK_BG_THRESH", 0.05f);
    const char* bg_env = std::getenv("TEXPROJ_BACK_BG");
    const std::string back_mode = bg_env ? bg_env : "black";
    // seam pairing knobs
    const float pair_maxd  = envf("SEAM_PAIR_MAXDIST", 0.01f);   // mesh units; mesh spans [-0.5,0.5]
    const float pair_mindot= envf("SEAM_PAIR_MINDOT", 0.3f);     // = TEXPROJ_FILL_MINDOT
    const int   pair_k     = std::max(1, std::min(32, envi("SEAM_PAIR_K", 4)));

    glb::Mesh m;
    if (!glb::read_glb(mesh_path.c_str(), m)) return 1;
    std::vector<uint32_t> faces(m.faces.begin(), m.faces.end());
    const size_t V = m.verts.size() / 3;
    std::printf("mesh   : %s  %zu verts  %zu tris  normals=%d\n", mesh_path.c_str(), V, faces.size() / 3,
                (int)(m.normals.size() == m.verts.size()));
    if (m.normals.size() != m.verts.size()) {
        std::fprintf(stderr, "mesh has no NORMAL attribute — this probe needs one\n"); return 1;
    }
    {
        float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
        for (size_t i = 0; i + 2 < m.verts.size(); i += 3)
            for (int c = 0; c < 3; c++) { mn[c] = std::min(mn[c], m.verts[i + c]); mx[c] = std::max(mx[c], m.verts[i + c]); }
        std::printf("         bbox x[%.4f,%.4f] y[%.4f,%.4f] z[%.4f,%.4f]%s\n", mn[0], mx[0], mn[1], mx[1], mn[2], mx[2],
                    (mx[0] > 1.1f || mn[0] < -1.1f) ? "   [WARN: not the pixal [-0.5,0.5] frame!]" : "");
    }

    // ---- SEAM_REAL_ATLAS=1: hand the REAL project_onto a synthetic atlas and let it run ----
    if (envi("SEAM_REAL_ATLAS", 0)) {
        texatlas::BakedTexture bt;
        build_synthetic_atlas(m, faces, bt, envi("SEAM_ATLAS_SIZE", 4096));
        Cfg pc;
        pc.front_img = front_path;
        pc.back_img  = back_path;
        pc.verbose   = true;
        const char* dd = std::getenv("SEAM_RENDER_DIR");
        if (dd) { pc.debug_dir = dd; mkdir(dd, 0755); }
        Stats ps;
        const bool ok = project_onto(bt, pc, &ps);
        std::printf("\n=== REAL project_onto returned %d ===\n", (int)ok);
        if (!ok) return 1;
        std::printf("  covered=%d front=%.2f%% back=%.2f%% holes=%.2f%%  fill3d=%d telea=%d\n",
                    ps.covered, ps.front_pct, ps.back_pct, ps.hole_pct, ps.n_fill3d, ps.n_telea);
        std::printf("  SEAM: n=%d  |d|=%.2f/255 (R%.2f G%.2f B%.2f)  bias=(%+.2f,%+.2f,%+.2f)  blend=%d\n",
                    ps.n_seam, ps.seam_mean_absdiff, ps.seam_absdiff[0], ps.seam_absdiff[1], ps.seam_absdiff[2],
                    ps.seam_bias[0], ps.seam_bias[1], ps.seam_bias[2], (int)ps.seam_blend_on);
        return 0;
    }

    // ---- build the two views exactly as project_onto does (front = yaw 0, back = yaw 180) ----
    std::vector<View> views;
    {
        View f(Cam(0.7332379387484828f, 1.3021559715270996f, 1.0f, 0.f));
        f.is_front = true; f.path = front_path;
        if (!load_rgba01(f.path, f.img, f.alpha, f.has_alpha)) return 1;
        views.push_back(std::move(f));
        View b(Cam(0.7332379387484828f, 1.3021559715270996f, 1.0f, 180.f));
        b.path = back_path;
        if (!load_rgba01(b.path, b.img, b.alpha, b.has_alpha)) return 1;
        views.push_back(std::move(b));
    }
    for (int i = 0; i < 2; i++)
        std::printf("view %d : %s  %dx%d  has_alpha=%d  yaw=%.0f\n", i, views[i].path.c_str(),
                    views[i].img.w, views[i].img.h, (int)views[i].has_alpha, views[i].cam.yaw_deg);

    for (int i = 0; i < 2; i++)
        views[i].z = raster_depth(m.verts, faces, views[i].cam, views[i].img.w, views[i].img.h);

    // Measure the normal convention with the REAL detector. It only reads bt.verts/bt.normals, so a
    // BakedTexture carrying just those is a faithful call, not a reimplementation.
    double sp = 0, sn = 0;
    float nsign;
    {
        texatlas::BakedTexture probe_bt;
        probe_bt.verts = m.verts; probe_bt.normals = m.normals;
        nsign = detect_normal_sign(probe_bt, views[0].z, views[0].cam, eps, true, &sp, &sn);
    }
    for (View& v : views) v.cam.nsign = nsign;

    for (int i = 0; i < 2; i++) {
        View& v = views[i];
        // mirror project_onto's resolution of TEXPROJ_BACK_BG=auto (alpha when the file has one)
        const std::string mode = v.is_front ? std::string("black")
                               : (back_mode == "auto" ? (v.has_alpha ? std::string("alpha") : std::string("black"))
                                                      : back_mode);
        const float thr = v.is_front ? front_thr : back_thr;
        subject_mask(v.img, v.alpha, v.has_alpha, mode, thr, true, v.subj, &v.ms);
        v.sil = silhouette_bbox(v.z);
        v.sub = bbox_of_mask(v.subj, v.img.w, v.img.h);
        v.fit = fit_similarity(v.sil, v.sub, false);
        std::printf("view %d : mask(%s thr=%.4f) subject %d px  fit scale=%.4f t=(%+.2f,%+.2f) fitted=%d\n",
                    i, mode.c_str(), thr, v.ms.n_subject, v.fit.sx, v.fit.tx, v.fit.ty, (int)v.fit.fitted);
        erode_mask(v.subj, v.img.w, v.img.h, erode);
    }

    // ---- REGISTRATION: does the view IMAGE's subject actually land on the mesh's silhouette? ----
    // The bbox fit only matches the two BOUNDING BOXES. If the generated back view's subject SHAPE differs
    // from the mesh's back silhouette by more than a similarity transform, the fit cannot fix it and the
    // back's content lands on the wrong surface near the seam. IoU of (fitted subject mask) vs (silhouette)
    // is the number that says so. The FRONT is the control: the mesh was built from that matte, so its IoU
    // is the best this pipeline can possibly do.
    std::printf("\n=== REGISTRATION: fitted subject mask vs mesh silhouette ===\n");
    for (int i = 0; i < 2; i++) {
        const View& v = views[i];
        // Rebuild the UN-eroded subject mask: erosion would bias IoU down for reasons unrelated to fit.
        std::vector<uint8_t> subj; MaskStats ms;
        subject_mask(v.img, v.alpha, v.has_alpha, v.is_front ? std::string("black") : back_mode,
                     v.is_front ? front_thr : back_thr, true, subj, &ms);
        long inter = 0, uni = 0, sil_only = 0, sub_only = 0;
        for (int y = 0; y < v.z.h; y++) for (int x = 0; x < v.z.w; x++) {
            const bool in_sil = std::isfinite(v.z.at(x, y));   // mesh silhouette pixel
            // map this silhouette pixel through the fit into view-image space, as project_onto samples
            float ix = (float)x + 0.5f, iy = (float)y + 0.5f;
            if (v.fit.fitted) v.fit.apply((float)x + 0.5f, (float)y + 0.5f, ix, iy);
            const int sx = (int)std::floor(ix), sy = (int)std::floor(iy);
            const bool in_sub = (sx >= 0 && sy >= 0 && sx < v.img.w && sy < v.img.h &&
                                 subj[(size_t)sy * v.img.w + sx]);
            if (in_sil && in_sub) inter++;
            if (in_sil || in_sub) uni++;
            if (in_sil && !in_sub) sil_only++;
            if (!in_sil && in_sub) sub_only++;
        }
        std::printf("view %d (yaw %3.0f): IoU=%.4f  intersect=%ld  mesh-only=%ld (%.2f%% of mesh silhouette)"
                    "  image-only=%ld\n", i, v.cam.yaw_deg, uni ? (double)inter / uni : 0.0, inter,
                    sil_only, inter + sil_only ? 100.0 * sil_only / (inter + sil_only) : 0.0, sub_only);
    }

    // ---- per-vertex: sample BOTH views, with and without the facing gate ----
    std::vector<Probe> pf(V), pb(V);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)V; i++) {
        const float* P = &m.verts[(size_t)i * 3];
        const float* N = &m.normals[(size_t)i * 3];
        pf[i] = sample_view(views[0], P, N, eps);
        pb[i] = sample_view(views[1], P, N, eps);
    }

    // Production "painted by view i" = facing > 0 AND visible AND smoothstep(nz_lo,nz_hi,facing) > 0
    // AND the sample is inside the eroded subject mask.
    auto painted = [&](const Probe& p) {
        return p.facing > 0.f && p.vis && smoothstep(nz_lo, nz_hi, p.facing) > 0.f && p.in_mask;
    };
    long n_pf = 0, n_pb = 0, n_both_painted = 0, n_both_sample = 0, n_neither = 0;
    for (size_t i = 0; i < V; i++) {
        const bool a = painted(pf[i]), b = painted(pb[i]);
        n_pf += a; n_pb += b; n_both_painted += (a && b);
        // "sample valid" IGNORING the facing sign: does each camera have an unoccluded, in-subject
        // sample of this exact 3D point? THIS is the honest geometric overlap.
        n_both_sample += (pf[i].vis && pf[i].in_mask && pb[i].vis && pb[i].in_mask);
        n_neither += (!a && !b);
    }
    std::printf("\n=== COVERAGE over %zu verts ===\n", V);
    std::printf("  front painted        : %8ld (%5.2f%%)\n", n_pf, 100.0 * n_pf / V);
    std::printf("  back  painted        : %8ld (%5.2f%%)\n", n_pb, 100.0 * n_pb / V);
    std::printf("  BOTH painted (c>0)   : %8ld (%5.2f%%)   <<< the 'overlap band' the blend averages\n",
                n_both_painted, 100.0 * n_both_painted / V);
    std::printf("  both have a valid sample IGNORING the facing sign: %8ld (%5.2f%%)\n",
                n_both_sample, 100.0 * n_both_sample / V);
    std::printf("  neither painted (hole): %8ld (%5.2f%%)\n", n_neither, 100.0 * n_neither / V);

    // ---- SEAM PAIRING: for each BACK-painted vertex, find its nearest FRONT-painted vertex in 3D ----
    // Same estimator the 3D hole fill uses (PointGrid + knn_gated + the normal gate), so the pairs are
    // exactly the neighbours the fill would blend across the seam.
    std::vector<int> src;
    for (size_t i = 0; i < V; i++) if (painted(pf[i])) src.push_back((int)i);
    std::vector<int> pairs_a, pairs_b; std::vector<float> pairs_d;
    if (!src.empty()) {
        PointGrid grid(src, m.verts, m.normals, 128, 0);
        const int ring = std::max(1, (int)std::ceil(pair_maxd / std::max(grid.cell, 1e-9f)));
        std::printf("\nseam pair grid: %dx%dx%d cell=%.5f  %zu front sources  maxdist=%.4f -> %d rings  k=%d mindot=%.2f\n",
                    grid.nx, grid.ny, grid.nz, grid.cell, grid.ids.size(), pair_maxd, ring, pair_k, pair_mindot);
        for (size_t i = 0; i < V; i++) {
            if (!painted(pb[i])) continue;
            const float* P = &m.verts[i * 3];
            const float* N = &m.normals[i * 3];
            const float nl = std::sqrt(N[0]*N[0] + N[1]*N[1] + N[2]*N[2]);
            if (!(nl > 1e-12f)) continue;
            const float nh[3] = {N[0]/nl, N[1]/nl, N[2]/nl};
            float bd[32]; int bi[32];
            const int cnt = knn_gated(grid, m.verts, m.normals, P, nh, pair_k, pair_mindot, ring, bi, bd);
            for (int j = 0; j < cnt; j++) {
                if (bd[j] > pair_maxd * pair_maxd) continue;
                pairs_a.push_back(bi[j]); pairs_b.push_back((int)i); pairs_d.push_back(std::sqrt(bd[j]));
            }
        }
    }
    std::printf("seam pairs: %zu (back-painted vert <-> nearest front-painted vert within %.4f)\n",
                pairs_b.size(), pair_maxd);

    // ---- SIMULATE project_onto per-vertex, then SPLAT-RENDER it from any yaw (CPU, no GPU) ----
    // The whole point: turn the statistics into something you can LOOK at. Vertices carry the same colour
    // decision as atlas texels, so splatting them through raster_depth's own z-buffer shows the seam.
    // SEAM_RENDER_DIR=<dir> enables it; SEAM_RENDER_YAWS="0,45,90,135,180" picks the cameras.
    const char* rdir_env = std::getenv("SEAM_RENDER_DIR");
    if (rdir_env && *rdir_env) {
        const std::string rdir = rdir_env;
        mkdir(rdir.c_str(), 0755);
        const int RW = std::max(64, envi("SEAM_RENDER_W", 480));
        // 0 = the shipped 3D fill. 1 = seam-aware cross-fade (the fix under test).
        const bool seam_fix   = envi("TEXPROJ_SEAM_BLEND", 0) != 0;
        const float seam_band = envf("TEXPROJ_SEAM_BAND", 0.35f);

        std::vector<float> rgb_lin((size_t)V * 3, 0.f);
        std::vector<uint8_t> valid(V, 0), src_of(V, 0);   // 1=front 2=back 3=fill
        for (size_t i = 0; i < V; i++) {
            if (painted(pf[i])) {
                for (int c = 0; c < 3; c++) rgb_lin[i * 3 + c] = srgb_to_linear(pf[i].col[c]);
                valid[i] = 1; src_of[i] = 1;
            } else if (painted(pb[i])) {
                for (int c = 0; c < 3; c++) rgb_lin[i * 3 + c] = srgb_to_linear(pb[i].col[c]);
                valid[i] = 1; src_of[i] = 2;
            }
        }
        // the shipped 3D fill, verbatim in structure (PointGrid + knn_gated + inverse-distance Shepard)
        const float fill_dot = envf("TEXPROJ_FILL_MINDOT", 0.3f);
        const int   fill_k   = std::max(1, std::min(32, envi("TEXPROJ_FILL_K", 6)));
        const float fill_dist= envf("TEXPROJ_FILL_MAXDIST", 0.08f);
        std::vector<int> painted_ids, hole_ids;
        for (size_t i = 0; i < V; i++) {
            if (!(pf[i].vis || pb[i].vis)) continue;          // inner wall of the shell: not a surface texel
            if (valid[i]) painted_ids.push_back((int)i); else hole_ids.push_back((int)i);
        }
        // THE SEAM METRIC. The front/back "overlap band" is empty by construction (facing_back ==
        // -facing_front, so no texel can ever have both confidences > 0). The ONLY place the two views
        // meet is a 3D-fill hole that has BOTH front-painted and back-painted neighbours in its k-NN.
        // |front_estimate - back_estimate| over THAT set is the honest "how far apart are the two images
        // where they meet" number, and it is what a colour match would have to shrink.
        long n_seam = 0, n_fillsrc = 0;
        double seam_d[3] = {0, 0, 0}, seam_dl[3] = {0, 0, 0}, delta_fix = 0;
        double seam_s[3] = {0, 0, 0}, seam_sl[3] = {0, 0, 0};   // SIGNED (front - back): the exposure test
        if (!painted_ids.empty()) {
            PointGrid grid(painted_ids, m.verts, m.normals, 128, envi("TEXPROJ_FILL_CELLCAP", 12));
            const int ring = std::max(1, (int)std::ceil(fill_dist / std::max(grid.cell, 1e-9f)));
            std::vector<float> frgb((size_t)hole_ids.size() * 3, 0.f);
            std::vector<uint8_t> got(hole_ids.size(), 0);
            const int NH = (int)hole_ids.size();
            #pragma omp parallel for schedule(dynamic, 512) \
                reduction(+:n_seam,delta_fix) reduction(+:seam_d[:3]) reduction(+:seam_dl[:3]) \
                reduction(+:seam_s[:3]) reduction(+:seam_sl[:3])
            for (int hi = 0; hi < NH; hi++) {
                const int p = hole_ids[hi];
                const float* P = &m.verts[(size_t)p * 3];
                const float* N = &m.normals[(size_t)p * 3];
                const float nl = std::sqrt(N[0]*N[0] + N[1]*N[1] + N[2]*N[2]);
                if (!(nl > 1e-12f)) continue;
                const float nh[3] = {N[0]/nl, N[1]/nl, N[2]/nl};
                float bd[32]; int bi[32];
                const int cnt = knn_gated(grid, m.verts, m.normals, P, nh, fill_k, fill_dot, ring, bi, bd);
                if (!cnt) continue;
                // Split the SAME k-NN set by provenance. wf/wb are the inverse-distance weights the
                // shipped fill already computes; separating them costs nothing and is what makes both the
                // seam metric and the cross-fade possible.
                float af[3] = {0,0,0}, ab[3] = {0,0,0}, wf = 0.f, wb = 0.f;
                for (int j = 0; j < cnt; j++) {
                    const float w = 1.f / (bd[j] + 1e-12f);
                    const uint8_t s = src_of[(size_t)bi[j]];
                    float* a = (s == 1) ? af : ab;
                    for (int c = 0; c < 3; c++) a[c] += w * rgb_lin[(size_t)bi[j] * 3 + c];
                    (s == 1 ? wf : wb) += w;
                }
                if (!(wf > 0.f) && !(wb > 0.f)) continue;
                float cf[3], cb[3];
                for (int c = 0; c < 3; c++) {
                    cf[c] = wf > 0.f ? af[c] / wf : 0.f;
                    cb[c] = wb > 0.f ? ab[c] / wb : 0.f;
                }
                // shipped fill = one inverse-distance mean over the whole k-NN set
                float ship[3];
                for (int c = 0; c < 3; c++) ship[c] = (af[c] + ab[c]) / (wf + wb);
                // seam-aware = cross-fade the two estimates by the hole's own facing
                float tt = smoothstep(-seam_band, seam_band, pf[p].facing);   // 1 = front side
                if (!(wf > 0.f)) tt = 0.f; else if (!(wb > 0.f)) tt = 1.f;
                float fix[3];
                for (int c = 0; c < 3; c++) fix[c] = tt * cf[c] + (1.f - tt) * cb[c];
                if (wf > 0.f && wb > 0.f) {   // THIS hole is on the seam: both views reach it
                    n_seam++;
                    for (int c = 0; c < 3; c++) {
                        seam_dl[c] += std::fabs(cf[c] - cb[c]);
                        seam_d[c]  += 255.0 * std::fabs(linear_to_srgb(cf[c]) - linear_to_srgb(cb[c]));
                        seam_sl[c] += cf[c] - cb[c];
                        seam_s[c]  += 255.0 * (linear_to_srgb(cf[c]) - linear_to_srgb(cb[c]));
                    }
                    for (int c = 0; c < 3; c++)
                        delta_fix += 255.0 * std::fabs(linear_to_srgb(fix[c]) - linear_to_srgb(ship[c])) / 3.0;
                }
                for (int c = 0; c < 3; c++) frgb[(size_t)hi * 3 + c] = seam_fix ? fix[c] : ship[c];
                got[hi] = 1;
            }
            for (int hi = 0; hi < NH; hi++) {
                if (!got[hi]) continue;
                const int p = hole_ids[hi];
                for (int c = 0; c < 3; c++) rgb_lin[(size_t)p * 3 + c] = frgb[(size_t)hi * 3 + c];
                valid[p] = 1; src_of[p] = 3; n_fillsrc++;
            }
        }
        std::printf("\nsim: %zu painted / %zu holes on the outer wall -> %ld 3D-filled  [seam_fix=%d band=%.2f]\n",
                    painted_ids.size(), hole_ids.size(), n_fillsrc, (int)seam_fix, seam_band);
        std::printf("SEAM BAND: %ld fill texels have BOTH a front- and a back-painted neighbour (%.2f%% of fills)\n",
                    n_seam, n_fillsrc ? 100.0 * n_seam / n_fillsrc : 0.0);
        if (n_seam) {
            std::printf("  |front_est - back_est| over the seam band: sRGB R=%.2f G=%.2f B=%.2f (mean %.2f/255)"
                        "   linear R=%.5f G=%.5f B=%.5f\n",
                        seam_d[0]/n_seam, seam_d[1]/n_seam, seam_d[2]/n_seam,
                        (seam_d[0]+seam_d[1]+seam_d[2])/(3.0*n_seam),
                        seam_dl[0]/n_seam, seam_dl[1]/n_seam, seam_dl[2]/n_seam);
            std::printf("  SIGNED (front_est - back_est), the exposure/WB test: sRGB R=%+.2f G=%+.2f B=%+.2f"
                        "   linear R=%+.5f G=%+.5f B=%+.5f\n",
                        seam_s[0]/n_seam, seam_s[1]/n_seam, seam_s[2]/n_seam,
                        seam_sl[0]/n_seam, seam_sl[1]/n_seam, seam_sl[2]/n_seam);
            std::printf("  => bias/scatter ratio R=%.3f G=%.3f B=%.3f  (a global colour match can only remove"
                        " the BIAS; <<1 means there is nothing for it to remove)\n",
                        std::fabs(seam_s[0])/std::max(seam_d[0],1e-9), std::fabs(seam_s[1])/std::max(seam_d[1],1e-9),
                        std::fabs(seam_s[2])/std::max(seam_d[2],1e-9));
            std::printf("  cross-fade vs shipped fill differs by %.2f/255 mean over the seam band\n",
                        delta_fix / n_seam);
        }

        // splat render: project every vertex through a yaw camera, z-test against that camera's own
        // z-buffer, write the nearest vertex's colour. src_of drives the provenance map.
        const char* yaws_env = std::getenv("SEAM_RENDER_YAWS");
        std::vector<float> yaws;
        {
            std::string s = yaws_env ? yaws_env : "0,45,90,135,180";
            size_t b = 0;
            while (b <= s.size()) { size_t e = s.find(',', b); if (e == std::string::npos) e = s.size();
                if (e > b) yaws.push_back((float)std::atof(s.substr(b, e - b).c_str())); b = e + 1; }
        }
        std::vector<uint8_t> strip((size_t)RW * yaws.size() * RW * 3, 0), pstrip((size_t)RW * yaws.size() * RW * 3, 0);
        const int SW = RW * (int)yaws.size();
        for (size_t yi = 0; yi < yaws.size(); yi++) {
            Cam rc(0.7332379387484828f, 1.3021559715270996f, 1.0f, yaws[yi]);
            rc.nsign = nsign;
            ZBuf rz = raster_depth(m.verts, faces, rc, RW, RW);
            std::vector<float> best((size_t)RW * RW, 1e30f);
            for (size_t i = 0; i < V; i++) {
                if (!valid[i]) continue;
                if (!(rc.facing(&m.normals[i * 3]) > 0.f)) continue;
                float u, vv; const float d = rc.project(&m.verts[i * 3], u, vv);
                if (!visible(rz, u, vv, d, eps)) continue;
                const int x = (int)std::floor(u * RW), y = (int)std::floor(vv * RW);
                if (x < 0 || y < 0 || x >= RW || y >= RW) continue;
                if (d >= best[(size_t)y * RW + x]) continue;
                best[(size_t)y * RW + x] = d;
                const size_t o = ((size_t)y * SW + (size_t)yi * RW + x) * 3;
                for (int c = 0; c < 3; c++) strip[o + c] = u8f(linear_to_srgb(rgb_lin[i * 3 + c]));
                const uint8_t s = src_of[i];
                pstrip[o + 0] = s == 1 ? 255 : 0;                       // RED   = front-projected
                pstrip[o + 1] = s == 2 ? 255 : 0;                       // GREEN = back-projected
                pstrip[o + 2] = s == 3 ? 255 : 0;                       // BLUE  = 3D-filled (INVENTED)
            }
        }
        dump_png(rdir + "/seam_render.png", strip.data(), SW, RW, 3);
        dump_png(rdir + "/seam_source.png", pstrip.data(), SW, RW, 3);
        std::printf("sim: wrote %s/seam_render.png + seam_source.png (R=front G=back B=3D-filled), yaws=%zu\n",
                    rdir.c_str(), yaws.size());
    }

    if (!outp.empty()) {
        {
            FILE* f = std::fopen((outp + "_verts.csv").c_str(), "w");
            if (!f) { std::fprintf(stderr, "cannot write %s_verts.csv\n", outp.c_str()); return 1; }
            std::fprintf(f, "i,x,y,z,nx,ny,nz,facing_f,f_vis,f_mask,f_paint,fr,fg,fb,b_vis,b_mask,b_paint,br,bg,bb\n");
            for (size_t i = 0; i < V; i++) {
                const float* P = &m.verts[i * 3]; const float* N = &m.normals[i * 3];
                std::fprintf(f, "%zu,%.6f,%.6f,%.6f,%.5f,%.5f,%.5f,%.5f,%d,%d,%d,%.6f,%.6f,%.6f,%d,%d,%d,%.6f,%.6f,%.6f\n",
                             i, P[0], P[1], P[2], N[0], N[1], N[2], pf[i].facing,
                             (int)pf[i].vis, (int)pf[i].in_mask, (int)painted(pf[i]),
                             pf[i].col[0], pf[i].col[1], pf[i].col[2],
                             (int)pb[i].vis, (int)pb[i].in_mask, (int)painted(pb[i]),
                             pb[i].col[0], pb[i].col[1], pb[i].col[2]);
            }
            std::fclose(f);
        }
        {
            FILE* f = std::fopen((outp + "_pairs.csv").c_str(), "w");
            if (!f) { std::fprintf(stderr, "cannot write %s_pairs.csv\n", outp.c_str()); return 1; }
            std::fprintf(f, "dist,x,y,z,facing_f,facing_b,fr,fg,fb,br,bg,bb\n");
            for (size_t p = 0; p < pairs_b.size(); p++) {
                const int ia = pairs_a[p], ib = pairs_b[p];
                const float* P = &m.verts[(size_t)ib * 3];
                std::fprintf(f, "%.6f,%.6f,%.6f,%.6f,%.5f,%.5f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                             pairs_d[p], P[0], P[1], P[2], pf[ia].facing, pb[ib].facing,
                             pf[ia].col[0], pf[ia].col[1], pf[ia].col[2],
                             pb[ib].col[0], pb[ib].col[1], pb[ib].col[2]);
            }
            std::fclose(f);
        }
        std::printf("wrote %s_verts.csv + %s_pairs.csv\n", outp.c_str(), outp.c_str());
    }
    return 0;
}
