set(TARGET nava)

add_executable(${TARGET}
    main.cpp
    ../common/media_io.cpp
    ../common/log.cpp
)
target_include_directories(${TARGET} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${PROJECT_SOURCE_DIR}/src"
)
target_link_libraries(${TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
if(SD_CUDA)
    target_compile_definitions(${TARGET} PRIVATE GGML_USE_CUDA)
endif()
if(SD_WEBP)
    target_compile_definitions(${TARGET} PRIVATE SD_USE_WEBP)
    target_link_libraries(${TARGET} PRIVATE webp libwebpmux)
endif()
if(SD_WEBM)
    target_compile_definitions(${TARGET} PRIVATE SD_USE_WEBM)
    target_link_libraries(${TARGET} PRIVATE webm)
    # The nava render muxes the LTX audio-VAE waveform into the webm. Browsers only
    # decode Vorbis/Opus in WebM (raw PCM plays silent), so encode Opus when libopus
    # is available — mirrors examples/cli + examples/server.
    if(SD_OPUS_FOUND)
        target_compile_definitions(${TARGET} PRIVATE SD_USE_OPUS)
        target_link_libraries(${TARGET} PRIVATE opus)
        # media_io.cpp includes <opus/opus.h>; pkg-config only exposes
        # -I/usr/include/opus, so the parent (Opus_INCLUDEDIR=/usr/include) is needed.
        # A cross sysroot toolchain doesn't search the host /usr/include, and CMake
        # strips /usr/include from target_include_directories as "implicit" — so pass
        # it as a raw compile option, which CMake does not strip. Harmless (redundant)
        # for a native compiler where /usr/include is already implicit.
        if(Opus_INCLUDEDIR)
            target_compile_options(${TARGET} PRIVATE "-I${Opus_INCLUDEDIR}")
        endif()
    endif()
endif()
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)
