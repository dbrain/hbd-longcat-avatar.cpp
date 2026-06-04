set(TARGET sd-cli)

add_executable(${TARGET}
    ../common/common.cpp
    ../common/log.cpp
    ../common/media_io.cpp
    image_metadata.cpp
    main.cpp
)
if(APPLE)
    sd_set_macos_rpaths(${TARGET})
endif()
target_include_directories(${TARGET} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${PROJECT_SOURCE_DIR}/src"
)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE stable-diffusion zip ${CMAKE_THREAD_LIBS_INIT})
if(SD_WEBP)
    target_compile_definitions(${TARGET} PRIVATE SD_USE_WEBP)
    target_link_libraries(${TARGET} PRIVATE webp libwebpmux)
endif()
if(SD_WEBM)
    target_compile_definitions(${TARGET} PRIVATE SD_USE_WEBM)
    target_link_libraries(${TARGET} PRIVATE webm)
    if(SD_OPUS_FOUND)
        target_compile_definitions(${TARGET} PRIVATE SD_USE_OPUS)
        target_link_libraries(${TARGET} PRIVATE opus)
        # media_io.cpp includes <opus/opus.h>; a cross sysroot toolchain doesn't search
        # the host /usr/include and CMake strips it as "implicit", so pass the opus parent
        # dir as a raw compile option (mirrors examples/nava).
        if(Opus_INCLUDEDIR)
            target_compile_options(${TARGET} PRIVATE "-I${Opus_INCLUDEDIR}")
        endif()
    endif()
endif()
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)
