# Try to find FreeImage project dll and include file
#
unset(FREEIMAGE_DLL CACHE)
unset(FREEIMAGE_INCLUDE_DIR CACHE)
unset(FREEIMAGE_FOUND CACHE)

find_path( FREEIMAGE_INCLUDE_DIR FreeImage.h
  if(UNIX)
    /usr/include
  endif(UNIX)
  ${PROJECT_SOURCE_DIR}/shared_external/Freeimage/include
  ${PROJECT_SOURCE_DIR}/../shared_external/Freeimage/include
  ${PROJECT_SOURCE_DIR}/../../shared_external/Freeimage/include
  ${PROJECT_SOURCE_DIR}/../../../shared_external/Freeimage/include
  ${PROJECT_SOURCE_DIR}/../../../../shared_external/Freeimage/include
  ${FREEIMAGE_LOCATION}/include
  $ENV{FREEIMAGE_LOCATION}/include
)

if(FREEIMAGE_INCLUDE_DIR)

  if(UNIX)
    find_library( FREEIMAGE_DLL "libfreeimage.so")
    # don't assume a static link library on Linux'
    set(FREEIMAGE_LIB ${FREEIMAGE_DLL})
  else(UNIX)
    file(GLOB FREEIMAGE_DLL "${FREEIMAGE_INCLUDE_DIR}/../x64/FREEIMAGE*.dll")
    file(GLOB FREEIMAGE_LIB "${FREEIMAGE_INCLUDE_DIR}/../x64/FREEIMAGE*.lib")
  endif(UNIX)

  if(FREEIMAGE_DLL)
    set( FREEIMAGE_FOUND "YES" )
    file( GLOB FREEIMAGE_HEADERS "${FREEIMAGE_INCLUDE_DIR}/FreeImage*.h")
  endif(FREEIMAGE_DLL)

else(FREEIMAGE_INCLUDE_DIR)
  message(WARNING "
      FreeImage not found. 
      The FreeImage folder you would specify with FREEIMAGE_LOCATION should contain:
      - folder: containing the FreeImage[64_]*.dll
      - include folder: containing the include files
      OR this folder could directly contain the dll and headers, put together
      For now, samples will run without additional UI. But that's okay ;-)"
  )
endif(FREEIMAGE_INCLUDE_DIR)
include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(FreeImage DEFAULT_MSG
    FREEIMAGE_INCLUDE_DIR
    FREEIMAGE_DLL
)


# Do we have to rewrite the variable here...
#SET(FREEIMAGE_DLL ${FREEIMAGE_DLL} CACHE PATH "path")

mark_as_advanced( FREEIMAGE_FOUND )
