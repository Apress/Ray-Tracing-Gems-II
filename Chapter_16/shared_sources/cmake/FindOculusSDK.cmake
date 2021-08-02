# Try to find the Oculus SDK
#
include(FindPackageHandleStandardArgs)

unset(OCULUSSDK_LIBS CACHE)
unset(OCULUSSDK_LIBS_DEBUG CACHE)
unset(OCULUSSDK_INCLUDE_DIRS CACHE)
unset(OCULUSSDK_FOUND CACHE)
unset(OCULUSSDK_SRC CACHE)
unset(OCULUSSDK_LOCATION CACHE)
unset(OCULUSSDK_KERNEL_LIB_DEBUG CACHE)
unset(OCULUSSDK_KERNEL_LIB CACHE)
unset(OCULUSSDK_PLATFORM_LIB CACHE)
unset(OCULUSSDK_AVATAR_LIB CACHE)

find_path( OCULUSSDK_LOCATION OculusSDK/LibOVR/Include/OVR_CAPI.h
  $ENV{OCULUSSDK_LOCATION}
  ${PROJECT_SOURCE_DIR}/../../shared_external_vr/OculusVR
  ${PROJECT_SOURCE_DIR}/../shared_external_vr/OculusVR
  ${PROJECT_SOURCE_DIR}/shared_external_vr/OculusVR
)
message( STATUS "Using OculusVR SDKs from " ${OCULUSSDK_LOCATION} )

if(OCULUSSDK_LOCATION)

  # put together the include dirs
  list(APPEND OCULUSSDK_INCLUDE_DIRS ${OCULUSSDK_LOCATION}/OculusSDK/LibOVR/Include)
  list(APPEND OCULUSSDK_INCLUDE_DIRS ${OCULUSSDK_LOCATION}/OculusSDK/LibOVRKernel/Src)
  list(APPEND OCULUSSDK_INCLUDE_DIRS ${OCULUSSDK_LOCATION}/OVRPlatformSDK/Include)
  list(APPEND OCULUSSDK_INCLUDE_DIRS ${OCULUSSDK_LOCATION}/OVRAvatarSDK/Include)
  mark_as_advanced(OCULUSSDK_INCLUDE_DIRS)

  # set the src files
  file(TO_CMAKE_PATH ${OCULUSSDK_LOCATION} OCULUSSDK_LOCATION_CMAKE)
  #set( OCULUSSDK_SRC ${OCULUSSDK_LOCATION_CMAKE}/OVRPlatformSDK/Windows/OVR_PlatformLoader.cpp)
  
  # find the Oculus VR lib (libOVR)
  # TODO: Linux handling
  if(MSVC)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(_OVR_ARCH "x64")
    else()
      set(_OVR_ARCH "Win32")
    endif()
  else()
    message(ERROR "OVR Linux support not yet implemented.")
  endif()
  mark_as_advanced(_OVR_ARCH)

  if(MSVC)
    if(MSVC11)
      set(_OVR_MSVC "VS2012")
    elseif(MSVC12)
      set(_OVR_MSVC "VS2013")
    elseif(MSVC14)
      set(_OVR_MSVC "VS2015")
    else()
      message(ERROR "FindOculusSDK: unsupported MSVC version.")
    endif()
    mark_as_advanced(_OVR_MSVC)
  endif()


  # LibOVR.lib: rendering and tracking, needed:
  find_library(OCULUSSDK_LIB LibOVR.lib HINTS 
                ${OCULUSSDK_LOCATION}/OculusSDK/LibOVR/Lib/Windows/${_OVR_ARCH}/Release/${_OVR_MSVC}
              )
  mark_as_advanced(OCULUSSDK_LIB)
  find_library(OCULUSSDK_LIB_DEBUG LibOVR.lib HINTS 
                ${OCULUSSDK_LOCATION}/OculusSDK/LibOVR/Lib/Windows/${_OVR_ARCH}/Debug/${_OVR_MSVC}
              )  
  mark_as_advanced(OCULUSSDK_LIB_DEBUG)
  
  # LibOVRKernel.lib: optional in older SDK versions. Support could get removed soon:
  find_library(OCULUSSDK_KERNEL_LIB LibOVRKernel.lib HINTS 
                ${OCULUSSDK_LOCATION}/OculusSDK/LibOVRKernel/Lib/Windows/${_OVR_ARCH}/Release/${_OVR_MSVC}
              )  
  mark_as_advanced(OCULUSSDK_KERNEL_LIB)
  find_library(OCULUSSDK_KERNEL_LIB_DEBUG LibOVRKernel.lib HINTS 
                ${OCULUSSDK_LOCATION}/OculusSDK/LibOVRKernel/Lib/Windows/${_OVR_ARCH}/Debug/${_OVR_MSVC}
              )  
  mark_as_advanced(OCULUSSDK_KERNEL_LIB_DEBUG)
  
  # LibOVRPlatform64_1.lib: needed for the avatar lib:
  find_library(OCULUSSDK_PLATFORM_LIB LibOVRPlatform64_1.lib HINTS 
                ${OCULUSSDK_LOCATION}/OVRPlatformSDK/Windows
              )
  mark_as_advanced(OCULUSSDK_PLATFORM_LIB)
  # libovravatar.lib: features like rendering of the Touch controllers:
  find_library(OCULUSSDK_AVATAR_LIB libovravatar.lib HINTS 
                ${OCULUSSDK_LOCATION}/OVRAvatarSDK/Windows
              )
  mark_as_advanced(OCULUSSDK_AVATAR_LIB)
             
  list(APPEND OCULUSSDK_LIBS ${OCULUSSDK_LIB})
  if (OCULUSSDK_KERNEL_LIB)
    # newer SDK versions don't have this lib anymore
    list(APPEND OCULUSSDK_LIBS ${OCULUSSDK_KERNEL_LIB})
  endif (OCULUSSDK_KERNEL_LIB)
  mark_as_advanced(OCULUSSDK_LIBS)

  list(APPEND OCULUSSDK_LIBS_DEBUG ${OCULUSSDK_LIB_DEBUG})
  if (OCULUSSDK_KERNEL_LIB_DEBUG)
    # newer SDK versions don't have this lib anymore
    list(APPEND OCULUSSDK_LIBS_DEBUG ${OCULUSSDK_KERNEL_LIB_DEBUG})
  endif (OCULUSSDK_KERNEL_LIB_DEBUG)
  mark_as_advanced(OCULUSSDK_LIBS_DEBUG)

  if(OCULUSSDK_PLATFORM_LIB)
    list(APPEND OCULUSSDK_LIBS       ${OCULUSSDK_PLATFORM_LIB})
    list(APPEND OCULUSSDK_LIBS_DEBUG ${OCULUSSDK_PLATFORM_LIB})
  endif(OCULUSSDK_PLATFORM_LIB)
  if(OCULUSSDK_AVATAR_LIB)
    list(APPEND OCULUSSDK_LIBS       ${OCULUSSDK_AVATAR_LIB})
    list(APPEND OCULUSSDK_LIBS_DEBUG ${OCULUSSDK_AVATAR_LIB})
    set(OCULUS_AVATAR_SUPPORT "YES")
  endif(OCULUSSDK_AVATAR_LIB)
  

  if(OCULUSSDK_LIB)
    set( OCULUSSDK_FOUND "YES" )
  endif(OCULUSSDK_LIB)

else(OCULUSSDK_LOCATION)

  message( "
      OculusSDK not found. 
      The OCULUSSDK folder you would specify with the OCULUSSDK_LOCATION env var and should contain
      LibOVR and LibOVRKernel folders in the structure the Oculus SDK is delivered."
  )

endif(OCULUSSDK_LOCATION)

find_package_handle_standard_args(OCULUSSDK DEFAULT_MSG
  OCULUSSDK_LOCATION
  OCULUSSDK_INCLUDE_DIRS
  OCULUSSDK_LIBS
  OCULUSSDK_LIBS_DEBUG
  OCULUSSDK_INCLUDE_DIRS
)

mark_as_advanced( OCULUSSDK_FOUND )

