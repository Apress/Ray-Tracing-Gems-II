cmake_minimum_required(VERSION 3.02)
#####################################################################################
# offer the choice of having shared_sources as a sub-folder... good for packaging a sample
#
# if BASE_DIRECTORY not defined, it means this cmake file was called as the first entry point and not included
if(NOT BASE_DIRECTORY) # if not defined, it means this cmake file was called as the first entry point and not included
  Message(FATAL_ERROR "cannot be directly invoked")
else()
  Message(STATUS "-------------------------------")
  Message(STATUS "Processing shared_sources files")
endif()


SET(NVPWINDOW_SOURCE
  ${BASE_DIRECTORY}/shared_sources/nvpwindow.cpp
  ${BASE_DIRECTORY}/shared_sources/nvpwindow.hpp
  ${BASE_DIRECTORY}/shared_sources/nvpsystem.cpp
  ${BASE_DIRECTORY}/shared_sources/nvpsystem.hpp
)

if(UNIX) 
  LIST(APPEND NVPWINDOW_SOURCE ${BASE_DIRECTORY}/shared_sources/nvpsystem_linux.cpp )
else()
  LIST(APPEND NVPWINDOW_SOURCE ${BASE_DIRECTORY}/shared_sources/nvpsystem_win32.cpp )
endif()

if(NOT (CMAKE_SIZEOF_VOID_P EQUAL 8))
  message( FATAL_ERROR "64-bit builds are mandatory for this framework" )
endif()

# CSF filefomat uses zlib
_add_package_ZLIB()
add_definitions(-DCSF_ZIP_SUPPORT=1)

# add packages that may be needed by some of the samples
_optional_package_VulkanSDK()
_optional_package_ShaderC()
_optional_package_OpenGL()
_optional_package_DirectX11()
_optional_package_DirectX12()
_optional_package_AntTweakBar()
_optional_package_Freeimage()
_optional_package_OculusSDK()
_optional_package_OpenVRSDK()
_optional_package_Optix()
_optional_package_Ftizb()
_optional_package_Cuda()
_optional_package_OpenCL()
_optional_package_NSight()
_optional_package_ZLIB()
_optional_package_Sockets()
_optional_package_ImGUI()

# process the rest of some cmake code that needs to be done after the packages
_process_shared_cmake_code()

file(GLOB FILEFORMATS_SOURCE fileformats/*.cpp fileformats/*.hpp fileformats/*.inl fileformats/*.h)
file(GLOB NVMATH_SOURCE nvmath/*.cpp nvmath/*.hpp nvmath/*.inl nvmath/*.h)
file(GLOB NVHELPERS_SOURCE nvh/*.cpp nvh/*.hpp nvh/*.inl nvh/*.h)
file(GLOB NOISE_SOURCE noise/*.cpp noise/*.hpp noise/*.inl noise/*.h)

set(library_name "shared_sources")

if(USING_OPENGL)
  file(GLOB NVHELPERSGL_SOURCE nvgl/*.cpp nvgl/*.hpp nvgl/*.inl nvgl/*.h)
  source_group("nvgl" FILES ${NVHELPERSGL_SOURCE})
  
  string(CONCAT library_name ${library_name} "_gl")
endif()

if(USING_VULKANSDK)
  file(GLOB NVHELPERSVK_SOURCE nvvk/*.cpp nvvk/*.hpp nvvk/*.inl nvvk/*.h)
  source_group("nvvk" FILES ${NVHELPERSVK_SOURCE})

  file(GLOB NVHELPERSVKPP_SOURCE nvvkpp/*.cpp nvvkpp/*.hpp nvvkpp/*.inl nvvkpp/*.h)
  source_group("nvvkpp" FILES ${NVHELPERSVKPP_SOURCE})

  string(CONCAT library_name ${library_name} "_vk")
  
  add_definitions(-DVULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
  #GLFW needs to see the vulkan header, so it can provide glfwCreateWindowSurface
  add_definitions(-DGLFW_INCLUDE_VULKAN)

  if(WIN32)
    add_definitions(-DVK_USE_PLATFORM_WIN32_KHR)
  endif(WIN32)
  if(UNIX)
    add_definitions(-DVK_USE_PLATFORM_XCB_KHR)
    add_definitions(-DVK_USE_PLATFORM_XLIB_KHR)
  endif(UNIX)
endif()

if (USING_DIRECTX11)
  file(GLOB NVHELPERSDX11_SOURCE nvdx11/*.cpp nvdx11/*.hpp nvdx11/*.inl nvdx11/*.h)
  source_group("nvdx11" FILES ${NVHELPERSDX11_SOURCE})

  string(CONCAT library_name ${library_name} "_dx11")
endif()

if (USING_DIRECTX12)
  file(GLOB NVHELPERSDX12_SOURCE nvdx12/*.cpp nvdx12/*.hpp nvdx12/*.inl nvdx12/*.h)
  source_group("nvdx12" FILES ${NVHELPERSDX12_SOURCE})

  string(CONCAT library_name ${library_name} "_dx12")
endif()


if (USING_OPTIX)
  Message(STATUS "Note: adding Optix utility files")
  file(GLOB NVHELPERSOPTIX_FILES nvoptix/*.cpp nvoptix/*.h)
  source_group("nvoptix" FILES ${NVHELPERSOPTIX_FILES})
endif()



source_group("fileformats" FILES ${FILEFORMATS_SOURCE})
source_group("noise" FILES ${NOISE_SOURCE})
source_group("nvmath" FILES ${NVMATH_SOURCE})
source_group("nvh" FILES ${NVHELPERS_SOURCE})
source_group("nvp" FILES ${NVPWINDOW_SOURCE})


if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(MSVC)

if(SUPPORT_NVTOOLSEXT)
  include_directories(
      ${BASE_DIRECTORY}/shared_external/NSight
  )
  LIST(APPEND PACKAGE_SOURCE_FILES 
    ${BASE_DIRECTORY}/shared_sources/nvh/nsightevents.h
    ${BASE_DIRECTORY}/shared_external/NSight/nvToolsExt.h
  )
  add_definitions(-DSUPPORT_NVTOOLSEXT)
endif()

if(USING_OPENGL)
  SET(OPENGL_FILES
    ${has_GL_FILES} 
  )
endif()

add_library(shared_sources STATIC 
  ${NVHELPERS_SOURCE}
  ${NVMATH_SOURCE}
  ${NOISE_SOURCE}
  ${FILEFORMATS_SOURCE}
  ${NVPWINDOW_SOURCE}
  ${OPENGL_FILES}
  ${NVHELPERSVK_SOURCE}
  ${NVHELPERSVKPP_SOURCE}
  ${NVHELPERSGL_SOURCE}
  ${NVHELPERSDX11_SOURCE}
  ${NVHELPERSDX12_SOURCE}
  ${PACKAGE_SOURCE_FILES}
  ${NVHELPERSOPTIX_FILES}
)

# Create precompiled header (added in CMake 3.16)
if((${CMAKE_VERSION} VERSION_GREATER "3.16.0") OR (${CMAKE_VERSION} VERSION_EQUAL "3.16.0"))
  target_precompile_headers(shared_sources PRIVATE
    "<algorithm>"
    ${VULKAN_HEADERS}
  )
endif()

if (MSVC)
  if (MSVC_VERSION GREATER_EQUAL 1920)
    set(glfw3 ${BASE_DIRECTORY}/shared_external/glfw3/lib-vc2019/glfw3.lib)
  elseif (MSVC_VERSION GREATER_EQUAL 1910)
    set(glfw3 ${BASE_DIRECTORY}/shared_external/glfw3/lib-vc2017/glfw3.lib)
  elseif (MSVC_VERSION GREATER_EQUAL 1900)
    set(glfw3 ${BASE_DIRECTORY}/shared_external/glfw3/lib-vc2015/glfw3.lib)
  else()
    message( FATAL_ERROR "incompatible MSVC version with glfw3 binaries" )
  endif()
  message( STATUS "using glfw3 " ${glfw3} )
else()
  set(glfw3 glfw)
endif()

target_link_libraries(shared_sources
  ${PLATFORM_LIBRARIES} ${glfw3} 
 )

set_target_properties(shared_sources PROPERTIES OUTPUT_NAME ${library_name})
message(STATUS "shared_sources library name: ${library_name}")
