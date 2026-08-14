// smpl_rest_joints.hpp — SMPL-H 22 REST ("zero-pose") joint positions, GENERATED.
//
// PROVENANCE
//   file   /mnt/hdd/3d/avatar-shootout/EMAGE/emage_evaltools/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz
//   quantity  J = J_regressor @ v_template, first 22 rows, pelvis translated to the
//             origin.  float32 on disk, widened to float64 before the product — exactly
//             `d['J_regressor'].astype(np.float64) @ d['v_template'].astype(np.float64)`.
//   emitted with %.17g, i.e. the shortest form that round-trips to the same double.
//
// WHY IT IS BAKED.  The retarget's only use of the 167 MB npz was these 528 bytes, so a
// delivery box was carrying an entire SMPL-X weights tree to compute a constant.  The
// numbers are a property of the MODEL, not of a run: nothing here is a tuned value.
//
// REGENERATE / FALSIFY (both need the npz; neither is on the delivery path)
//   ./motion_retarget --bake-smpl-rest > smpl_rest_joints.hpp
//   ./motion_retarget --check-smpl-rest      # bit-for-bit vs the npz, exit 0 = identical
// The checker is the acceptance test: `identical 66/66` means the baked path and the
// npz path cannot diverge, so the retarget stays bit-exact.
#pragma once

namespace mret {

// [22][3], metres, pelvis at the origin.  Order = SMPL-H 22 (rig::kSmplNames).
inline const double (*smpl_rest_joints_baked())[3] {
    static const double J[22][3] = {
        {0, 0, 0},   //  0 pelvis
        {0.058189405117024431, -0.092763464376070526, -0.026001186732432735},   //  1 L_hip
        {-0.063267473520169246, -0.10390805428733857, -0.021250374024965997},   //  2 R_hip
        {-0.0027626926538583787, 0.10989058440597618, -0.027617633683067702},   //  3 spine1
        {0.11288485355066444, -0.47151688284414983, -0.035397252013969552},   //  4 L_knee
        {-0.10747742822434031, -0.46628806623270913, -0.038074253457721322},   //  5 R_knee
        {0.0066850049574832286, 0.2417438284330245, -0.033557616577079605},   //  6 spine2
        {0.069431408528923821, -0.8745763311334821, -0.067273195158720184},   //  7 L_ankle
        {-0.092060630453805914, -0.87701585964686191, -0.05826652072817972},   //  8 R_ankle
        {-0.0046454126639227178, 0.29397898714226878, -0.0051107188665761322},   //  9 spine3
        {0.11668871835321823, -0.93257373421321421, 0.050943123602820786},   // 10 L_foot
        {-0.13087301456903785, -0.9353443120618391, 0.060782473976941309},   // 11 R_foot
        {-0.016809871213276942, 0.45914605428917599, -0.036726060352586118},   // 12 neck
        {0.041718745681698546, 0.37892271327084154, -0.012331201139150903},   // 13 L_collar
        {-0.052340340612652393, 0.37831766422361113, -0.018510618092476939},   // 14 R_collar
        {0.007973617080612512, 0.61959783710888483, -0.015988797635041831},   // 15 head
        {0.16095776506026258, 0.43665073872180316, -0.0277921404875415},   // 16 L_shoulder
        {-0.15491807184503126, 0.43184210406069312, -0.031179147019015332},   // 17 R_shoulder
        {0.41508062601372225, 0.36450022102131391, -0.070250992024212089},   // 18 L_elbow
        {-0.42606763822131899, 0.39534963088203162, -0.057646236704712174},   // 19 R_elbow
        {0.6670673797981973, 0.38772144918663781, -0.072723074952288913},   // 20 L_wrist
        {-0.67533508621453442, 0.39081708030770373, -0.072971419294757728},   // 21 R_wrist
    };
    return J;
}

// The shape retarget_delta / decide_lr_swap want.
inline std::vector<V3> smpl_rest_joints_baked_v3() {
    const double (*J)[3] = smpl_rest_joints_baked();
    std::vector<V3> out(22);
    for (int i = 0; i < 22; i++) out[(size_t)i] = V3{J[i][0], J[i][1], J[i][2]};
    return out;
}

}  // namespace mret
