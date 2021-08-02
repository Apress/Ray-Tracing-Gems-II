# Copyright 2021 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.5)
set(PROJNAME "vk_ray_tracing_gems_2_ao")
project(${PROJNAME})

#####################################################################################
# look for shared_sources 1) as a sub-folder 2) at some other locations
# this cannot be put anywhere else since we still didn't find setup.cmake yet
# This is intended to handle both new and old shared_sources setups.
#
set(ADD_SUBDIR_BELOW 1)
if(NOT BASE_DIRECTORY)
  find_path(BASE_DIRECTORY
    NAMES shared_sources/cmake/setup.cmake shared_sources/CMakeLists_include.txt
    PATHS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../.. 
    REQUIRED
    DOC "Directory containing shared_sources"
    )
endif()
if(EXISTS ${BASE_DIRECTORY}/shared_sources/CMakeLists_include.txt)
  include(${BASE_DIRECTORY}/shared_sources/CMakeLists_include.txt)
elseif(EXISTS ${BASE_DIRECTORY}/shared_sources/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/shared_sources/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/shared_sources/cmake/utilities.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing shared_sources")
endif()

#####################################################################################
# Add sample packages
#
_add_package_VulkanSDK()
_add_shared_sources_lib()

#####################################################################################
_add_project_definitions(${PROJNAME})

#####################################################################################
# Source files for this project
#
file(GLOB SOURCE_FILES *.cpp *.hpp *.inl *.h *.c)

#####################################################################################
# GLSL to SPIR-V custom build
set(VULKAN_TARGET_ENV vulkan1.2)
unset(GLSL_SOURCES)
unset(SPV_OUTPUT)
file(GLOB_RECURSE GLSL_HEADER_FILES "shaders/*.h")
file(GLOB_RECURSE GLSL_SOURCE_FILES "shaders/*.rgen" "shaders/*.rmiss" "shaders/*.rchit" "shaders/*.frag" "shaders/*.vert")
foreach(GLSL ${GLSL_SOURCE_FILES})
  get_filename_component(FILE_NAME ${GLSL} NAME)
  _compile_GLSL(${GLSL} "autogen/${FILE_NAME}.spv" GLSL_SOURCES SPV_OUTPUT)
endforeach()

list(APPEND GLSL_SOURCES ${GLSL_HEADER_FILES})
source_group("Shader Files" FILES ${GLSL_SOURCES})

#####################################################################################
# Executable
#
add_executable(${PROJNAME} ${SOURCE_FILES} ${GLSL_SOURCES})

#####################################################################################
# Source code group
#
source_group("Source Files" FILES ${SOURCE_FILES})

#####################################################################################
# Linkage
#
target_link_libraries(${PROJNAME} ${PLATFORM_LIBRARIES} shared_sources)

foreach(DEBUGLIB ${LIBRARIES_DEBUG})
  target_link_libraries(${PROJNAME} debug ${DEBUGLIB})
endforeach(DEBUGLIB)

foreach(RELEASELIB ${LIBRARIES_OPTIMIZED})
  target_link_libraries(${PROJNAME} optimized ${RELEASELIB})
endforeach(RELEASELIB)

#####################################################################################
# copies binaries that need to be put next to the exe files (ZLib, etc.)
#
_copy_binaries_to_target( ${PROJNAME} )

install(FILES ${SPV_OUTPUT} CONFIGURATIONS Release DESTINATION "bin_${ARCH}/${PROJNAME}/shaders")
install(FILES ${SPV_OUTPUT} CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug/${PROJNAME}/shaders")
install(DIRECTORY "scenes" CONFIGURATIONS Release DESTINATION "bin_${ARCH}/${PROJNAME}")
install(DIRECTORY "scenes" CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug/${PROJNAME}")