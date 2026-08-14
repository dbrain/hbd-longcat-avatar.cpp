// rig_stage_report.hpp — what the rig stage decided, as plain data.
//
// image_to_rig_main() returns an int, and a service cannot learn from an int whether the rig it just
// got was ACCEPTED by the selector or merely the least-bad of N rejected draws. Scraping stdout is
// not an answer (the log is not the contract), and reading it back off the GLB cannot recover a
// gate verdict that was computed before the write.
//
// So the rig stage fills one of these and `image_to_rig_last_rig_report()` hands it over. It is a
// side channel by design: the engine is documented one-request-at-a-time on one CUDA context
// (avatar_pipeline.hpp), which is exactly the condition under which a last-run report is well
// defined. Anything that ever runs two rig stages concurrently must plumb this through the call
// instead — and will have bigger problems than this struct.
#pragma once
#include <string>

namespace rig {

struct StageReport {
    bool   valid = false;        // a rig stage ran in this process (false after a --rig-cache hit)
    int    draws = 0;            // how many conditioning-cloud draws were decoded (1 = no re-draw)
    int    accepted_draw = -1;   // which draw the selector accepted; -1 = none passed, best-ranked shipped
    bool   accepted = false;

    // The three predicate terms, for the SHIPPED draw.
    bool   skin_ok = false;
    bool   humanoid_gate_ok = false;
    bool   generic_gate_ok = false;
    int    named_core = 0;          // of the SMPL-22 core slots
    int    falsifier_fails = -1;    // -1 = naming failed outright / not run

    // The native pose gate (rig_pose_gate.hpp, --generic-all-influential audit).
    bool   pose_gate_ran = false;
    bool   pose_gate_pass = false;
    double pose_gate_worst = 0;     // max(all-influential-worst, component-worst); gate limit 6.0
    double pose_gate_moved = 0;

    int    J = 0;
    std::string summary;            // one human-readable line, already formatted
};

}  // namespace rig

// Defined in image_to_rig.cpp. Reports whichever rig stage ran last IN THIS PROCESS; `valid` is
// false before the first one and after a --rig-cache hit (no draw was decoded).
const rig::StageReport& image_to_rig_last_rig_report();
