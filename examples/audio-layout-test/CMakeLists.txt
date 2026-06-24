set(TARGET sd-audio-layout-test)

add_executable(${TARGET}
    ../common/log.cpp
    ../common/media_io.cpp
    main.cpp
)
target_include_directories(${TARGET} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${PROJECT_SOURCE_DIR}/src"
)
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
        if(Opus_INCLUDEDIR)
            target_compile_options(${TARGET} PRIVATE "-I${Opus_INCLUDEDIR}")
        endif()
    endif()
endif()
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)
