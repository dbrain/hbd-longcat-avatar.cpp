set(TARGET sd-delight)

add_executable(${TARGET} main.cpp)
if(APPLE)
    sd_set_macos_rpaths(${TARGET})
endif()
target_include_directories(${TARGET} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/thirdparty"
)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)
