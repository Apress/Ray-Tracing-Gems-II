# Try to find OpenVR
#
include(FindPackageHandleStandardArgs)

unset(OPENVRSDK_LIBS CACHE)
unset(OPENVRSDK_LIBS_DEBUG CACHE)
unset(OPENVRSDK_INCLUDE_DIRS CACHE)
unset(OPENVRSDK_FOUND CACHE)
unset(OPENVRSDK_LOCATION CACHE)

find_path( OPENVRSDK_LOCATION headers/openvr.h
  $ENV{OPENVRSDK_LOCATION}
  ${PROJECT_SOURCE_DIR}/../../shared_external_vr/OpenVR
  ${PROJECT_SOURCE_DIR}/../shared_external_vr/OpenVR
  ${PROJECT_SOURCE_DIR}/shared_external_vr/OpenVR
)
message( STATUS "Using OpenVR SDK from " ${OPENVRSDK_LOCATION} )

if(OPENVRSDK_LOCATION)

  # put together the include dirs
  list(APPEND OPENVRSDK_INCLUDE_DIRS ${OPENVRSDK_LOCATION}/headers)
  mark_as_advanced(OPENVRSDK_INCLUDE_DIRS)

  # find the OpenVR SDK lib (openvr_api.lib)
  # TODO: Linux handling
  if(MSVC)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(_OPENVR_ARCH "win64")
    else()
      set(_OPENVR_ARCH "win32")
    endif()
  else()
    message(ERROR "OpenVR Linux support not yet implemented.")
  endif()
  mark_as_advanced(_OPENVR_ARCH)

  if(MSVC)
    if(MSVC11)
      set(_OPENVR_MSVC "VS2012")
    elseif(MSVC12)
      set(_OPENVR_MSVC "VS2013")
    elseif(MSVC14)
      set(_OPENVR_MSVC "VS2015")
    else()
      message(ERROR "FindOpenVRSDK: unsupported MSVC version.")
    endif()
    mark_as_advanced(_OPENVR_MSVC)
  endif()


  # TODO: Linux handling
  find_library(OPENVRSDK_LIB openvr_api.lib HINTS 
                ${OPENVRSDK_LOCATION}/lib/${_OPENVR_ARCH}/
              )  
  mark_as_advanced(OPENVRSDK_LIB)
             
  list(APPEND OPENVRSDK_LIBS ${OPENVRSDK_LIB})
  mark_as_advanced(OPENVRSDK_LIBS)

  if(OPENVRSDK_LIB)
    set( OPENVRSDK_FOUND "YES" )
  endif(OPENVRSDK_LIB)

else(OPENVRSDK_LOCATION)

  message( "
      OPENVRSDK not found. 
      The OPENVRSDK folder you would specify with the OPENVRSDK_LOCATION env var and should contain
      \"lib\" and \"headers\" folders in the structure the OpenVR SDK is delivered."
  )

endif(OPENVRSDK_LOCATION)

find_package_handle_standard_args(OPENVRSDK DEFAULT_MSG
  OPENVRSDK_LOCATION
  OPENVRSDK_INCLUDE_DIRS
  OPENVRSDK_LIBS
)

mark_as_advanced( OPENVRSDK_FOUND )

