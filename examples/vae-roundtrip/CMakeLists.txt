set(TARGET sd-vae-roundtrip)

# Stale VAE-oracle harness: main.cpp references a removed API (ggml_backend_cpu_init,
# WAN::WanVAERunner — now WanRunner) and does not compile against current master.
# EXCLUDE_FROM_ALL so `cmake --build` (all) stays green; build explicitly to revive it.
add_executable(${TARGET} EXCLUDE_FROM_ALL main.cpp)
target_include_directories(${TARGET} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/src/core"
    "${PROJECT_SOURCE_DIR}/src/model/diffusion"
)
target_link_libraries(${TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)
