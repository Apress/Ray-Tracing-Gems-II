cmake_minimum_required (VERSION 3.19)
project (pathtrace)
set(SOURCES
    program.cpp
    balance_heuristic.h
    functions.h
    direct_cos.h
    direct_mat.h
    direct_light.h
    direct_mis.h
    sample_lights.h
    sample_lights_pdf.h
    material_lambert.h
    material_rough.h
    pathtrace.h
    pathtrace_mis.h
    trace_wrappers.h
    program_defs.h
    externals/stb_image_write.h
    externals/pcg-cpp-0.98/include/pcg_random.hpp
)
add_executable (pathtrace ${SOURCES})
target_include_directories(pathtrace PUBLIC "externals")
target_include_directories(pathtrace PUBLIC "externals/pcg-cpp-0.98/include")
set_property(TARGET pathtrace PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(pathtrace PUBLIC OpenMP::OpenMP_CXX)
endif()
