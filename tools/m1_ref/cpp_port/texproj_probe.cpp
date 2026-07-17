// texproj_probe — measure the FRONT silhouette fit + subject mask WITHOUT a GPU run.
//
// Why this exists: tex_project.hpp's BUG 1 fix (auto-align the front) and BUG 2 fix (reject background
// samples) both hinge on numbers that were previously ASSERTED rather than measured — "the front IS the
// matte, pixel-perfect" was asserted and is false. This runs the REAL texproj code (the same Cam,
// raster_depth, silhouette_bbox, subject_mask, fit_similarity, erode_mask) against a mesh GLB + a view
// image and prints what project_onto will compute, in seconds, on the CPU, with no atlas bake and no GPU.
//
//   ./texproj_probe <mesh.glb> <view.png> [yaw_deg] [thresh] [erode]
//
// e.g. ./texproj_probe $AS/_shootout_out/inline_soldier1536/refined.glb
//                      $AS/_shootout_out/soldier_matte.png 0 0.0039 2
//
// Reads nothing else and writes nothing. Build: ./build.sh texproj_probe
#include "tex_project.hpp"
#include "glb_reader.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <mesh.glb> <view.png> [yaw_deg=0] [bg_thresh=0.0039] [erode=2]\n", argv[0]);
        return 2;
    }
    const std::string mesh_path = argv[1], img_path = argv[2];
    const float yaw    = argc > 3 ? (float)std::atof(argv[3]) : 0.f;
    const float thresh = argc > 4 ? (float)std::atof(argv[4]) : 1.f / 255.f;
    const int   erode  = argc > 5 ? std::atoi(argv[5]) : 2;

    // DEF_CAM / DEF_DIST / DEF_MS (image_to_rig.cpp:49) — the same camera project_onto defaults to.
    texproj::Cam cam(0.7332379387484828f, 1.3021559715270996f, 1.0f, yaw);

    glb::Mesh m;
    if (!glb::read_glb(mesh_path.c_str(), m)) return 1;
    std::vector<uint32_t> faces(m.faces.begin(), m.faces.end());
    std::printf("mesh   : %s  %zu verts  %zu tris\n", mesh_path.c_str(), m.verts.size() / 3, faces.size() / 3);
    {   // the mesh must already be in the pixal [-0.5,0.5] frame (image_to_rig bbox_canon_onto's output)
        float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
        for (size_t i = 0; i + 2 < m.verts.size(); i += 3)
            for (int c = 0; c < 3; c++) { mn[c] = std::min(mn[c], m.verts[i + c]); mx[c] = std::max(mx[c], m.verts[i + c]); }
        std::printf("         bbox x[%.4f,%.4f] y[%.4f,%.4f] z[%.4f,%.4f]%s\n", mn[0], mx[0], mn[1], mx[1], mn[2], mx[2],
                    (mx[0] > 1.1f || mn[0] < -1.1f) ? "   [WARN: not in the pixal [-0.5,0.5] frame!]" : "");
    }

    imgio::Image img; std::vector<float> alpha; bool has_alpha = false;
    if (!texproj::load_rgba01(img_path, img, alpha, has_alpha)) return 1;
    std::printf("image  : %s  %dx%d  has_alpha=%d\n", img_path.c_str(), img.w, img.h, (int)has_alpha);

    // --- the mesh's silhouette from this yaw = the covered pixels of its z-buffer (what the fit targets)
    texproj::ZBuf z = texproj::raster_depth(m.verts, faces, cam, img.w, img.h);
    texproj::BBox sil = texproj::silhouette_bbox(z);
    std::printf("\nsilhouette (z-buffer, yaw=%.1f): [%d,%d..%d,%d]  %.0fx%.0f px  "
                "u[%.4f,%.4f] w=%.4f  v[%.4f,%.4f] h=%.4f\n", yaw,
                sil.x0, sil.y0, sil.x1, sil.y1, sil.w(), sil.h(),
                sil.x0 / (float)img.w, (sil.x1 + 1) / (float)img.w, sil.w() / (float)img.w,
                sil.y0 / (float)img.h, (sil.y1 + 1) / (float)img.h, sil.h() / (float)img.h);

    // --- the image's subject, exactly as project_onto builds it
    for (int holes = 1; holes >= 0; holes--) {
        std::vector<uint8_t> subj; texproj::MaskStats ms;
        texproj::subject_mask(img, alpha, has_alpha, "black", thresh, holes != 0, subj, &ms);
        texproj::BBox sub = texproj::bbox_of_mask(subj, img.w, img.h);
        std::printf("\nsubject (black thresh=%.4f holefill=%d): %d px (%.2f%% of image)  nonzero=%d  "
                    "holefill-reclaimed=%d  DISCARDED-nonzero=%d (%.2f%%)\n",
                    thresh, holes, ms.n_subject, 100.0 * ms.n_subject / (double)(img.w * img.h),
                    ms.n_nonzero, ms.n_holefill, ms.n_lost,
                    ms.n_nonzero ? 100.0 * ms.n_lost / ms.n_nonzero : 0.0);
        std::printf("        bbox [%d,%d..%d,%d]  %.0fx%.0f px  u[%.4f,%.4f] w=%.4f  v[%.4f,%.4f] h=%.4f\n",
                    sub.x0, sub.y0, sub.x1, sub.y1, sub.w(), sub.h(),
                    sub.x0 / (float)img.w, (sub.x1 + 1) / (float)img.w, sub.w() / (float)img.w,
                    sub.y0 / (float)img.h, (sub.y1 + 1) / (float)img.h, sub.h() / (float)img.h);
        if (holes) {
            for (int aniso = 0; aniso <= 1; aniso++) {
                texproj::Fit f = texproj::fit_similarity(sil, sub, aniso != 0);
                std::printf("        FIT %-5s: fitted=%d  scale=%.4f/%.4f  (w-ratio %.4f  h-ratio %.4f)  "
                            "translate=(%+.2f, %+.2f) px\n", aniso ? "aniso" : "min", (int)f.fitted,
                            f.sx, f.sy, sub.w() / sil.w(), sub.h() / sil.h(), f.tx, f.ty);
                // how far does the fit actually MOVE a sample? that is the whole point of BUG 1.
                if (!f.fitted) continue;
                const float probes[4][2] = {{0.5f, 0.5f}, {0.5f, 0.38f}, {0.30f, 0.55f}, {0.5f, 0.92f}};
                const char* names[4] = {"image centre", "chest/buttons", "hand (left)", "boot sole"};
                for (int k = 0; k < 4; k++) {
                    const float px = probes[k][0] * img.w, py = probes[k][1] * img.h;
                    float ox, oy; f.apply(px, py, ox, oy);
                    std::printf("            shift @ %-14s (%.0f,%.0f) -> (%.1f,%.1f)  = (%+.2f, %+.2f) px\n",
                                names[k], px, py, ox, oy, ox - px, oy - py);
                }
            }
            std::vector<uint8_t> er = subj;
            for (int r = 1; r <= 3; r++) {
                er = subj; texproj::erode_mask(er, img.w, img.h, r);
                int n = 0; for (uint8_t q : er) n += q;
                std::printf("        erode r=%d: %d -> %d subject px (rim %d, %.2f%%)%s\n",
                            r, ms.n_subject, n, ms.n_subject - n, 100.0 * (ms.n_subject - n) / ms.n_subject,
                            r == erode ? "   <- default TEXPROJ_BG_ERODE" : "");
            }
        }
    }
    return 0;
}
