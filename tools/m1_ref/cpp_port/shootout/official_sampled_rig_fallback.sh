#!/usr/bin/env bash
# Retired deliberately: no production rig may be created by the upstream
# Python SkinTokens path.  Native R1/R3/R4 and the real GLB pose gate are the
# only publishable route; failures are retained by native_image_to_rig.sh.
set -euo pipefail

echo "official_sampled_rig_fallback.sh is retired: Python-assisted rig publication is disabled." >&2
echo "Use rig_texture_chain.sh (or native_image_to_rig.sh); a native failure remains rejected with its log." >&2
exit 64
