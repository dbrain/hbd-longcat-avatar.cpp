# visioncpp (vendored)

The inference library from [vision.cpp](https://github.com/Acly/vision.cpp), vendored so the
RMBG-2.0 background matte runs **in-process** in this repo — no `vision-server`, no
`docker run … vision-cli`, no runtime model download.

* Source: `/home/dbrain/dev/vision.cpp` @ `324c84dbcc3c431c9a51812871c732bc8c31e2bd`
  (`project(vision.cpp VERSION 0.3.1)`).
* Copied verbatim: `include/visp/*.h`, `src/visp/**` (minus `c-api.cpp`), `src/util/*.h`.
* Not copied: the HTTP server, the CLI, siglip2, sam3, tests, its `depend/` (its own ggml, stb, fmt).
* Local additions: `src/visp_stb_resize.cpp` (stb resize implementation only).

**Nothing here is modified.** Refreshing is a straight copy of those four paths plus a rebuild;
if a future upstream bump needs a local patch, add it as a separate file and say so here.

## Why vendored rather than linked

vision.cpp is a CMake project with a clean public API, so linking it was the obvious alternative.
It was rejected because its build brings **a second ggml** (`depend/ggml`) into a binary that
already links this repo's ggml — two copies of every `ggml_*` symbol, two CUDA contexts, and a
matte that cannot share the process (and therefore the 3060 lock) with the rig. Vendoring follows
the precedent already set here by `cumesh_native`, `xatlas`, `meshoptimizer` and `eigen_reference`,
and `build.sh` compiles vendored sources directly.

The sources compile **unmodified** against this repo's ggml: vision.cpp's `depend/ggml` is the
same `dbrain/ggml` fork, pinned at `d20b816d`, which is an ancestor of ours.

## Build

`tools/m1_ref/cpp_port/build_visioncpp.sh` compiles this tree plus `matte_native.cpp` into
`build/libvisioncpp.a` (C++20, g++-15 — visp needs `<format>`/`<span>`; callers stay C++17 and see
only `matte_native.hpp`). A **static archive** on purpose: the linker pulls a member only to
resolve an undefined symbol, so visp's `stbi_*` references bind to whatever the calling binary
already instantiated via `image_io.hpp`, and the stb-resize member is pulled in only when nothing
else provides it — no duplicate-symbol failures in either direction.

## The one trap

`VISP_FLASH_ATTENTION=0` is **required**. The Swin backbone runs head_dim=32, for which ggml's
CUDA flash-attention has no kernel instance: it does not fall back, it aborts inside `fattn.cu`.
`matte_native.cpp` sets it (with `overwrite=0`, so an explicit environment setting still wins).
Anything else calling `birefnet_*` directly must do the same.
