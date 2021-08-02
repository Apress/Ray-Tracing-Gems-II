# Try to find SvcMFCUI project dll and include file
#
unset(ANTTWEAKBAR_DLL CACHE)
unset(ANTTWEAKBAR_INCLUDE_DIR CACHE)
unset(ANTTWEAKBAR_FOUND CACHE)

find_path( ANTTWEAKBAR_INCLUDE_DIR AntTweakBar.h
  ${PROJECT_SOURCE_DIR}/shared_external/AntTweakBar/include
  ${PROJECT_SOURCE_DIR}/../shared_external/AntTweakBar/include
  ${PROJECT_SOURCE_DIR}/../../shared_external/AntTweakBar/include
  ${PROJECT_SOURCE_DIR}/../../../shared_external/AntTweakBar/include
  ${PROJECT_SOURCE_DIR}/../../../../shared_external/AntTweakBar/include
  ${ANTTWEAKBAR_LOCATION}/include
  $ENV{ANTTWEAKBAR_LOCATION}/include
)

macro(_find_dll targetVar dllName dllName64 folder)
  if(ARCH STREQUAL "x86")
      file(GLOB ANTTWEAKBAR_DLLS "${ANTTWEAKBAR_INCLUDE_DIR}/../lib/${folder}${dllName}")
      list(LENGTH ANTTWEAKBAR_DLLS NUMDLLS)
      if(NUMDLLS EQUAL 0)
        file(GLOB ANTTWEAKBAR_DLLS "${ANTTWEAKBAR_INCLUDE_DIR}/${folder}${dllName}")
      endif()
  else()
      file(GLOB ANTTWEAKBAR_DLLS "${ANTTWEAKBAR_INCLUDE_DIR}/../lib/${folder}${dllName64}")
      list(LENGTH ANTTWEAKBAR_DLLS NUMDLLS)
      if(NUMDLLS EQUAL 0)
        file(GLOB ANTTWEAKBAR_DLLS "${ANTTWEAKBAR_INCLUDE_DIR}/${folder}${dllName64}")
      endif()
  endif()

  list(LENGTH ANTTWEAKBAR_DLLS NUMDLLS)
  if(NUMDLLS EQUAL 0)
    message(STATUS "dll for the User Interface not found (${folder}${dllName}, ${folder}${dllName64})" )
    set (${targetVar} "NOTFOUND")
  else()
    list(GET ANTTWEAKBAR_DLLS 0 ${targetVar})
  endif()
endmacro()

macro(_find_so targetVar soName)
  file(GLOB ANTTWEAKBAR_SO "${ANTTWEAKBAR_INCLUDE_DIR}/../lib/${folder}${soName}")
  list(LENGTH ANTTWEAKBAR_SO NUMSO)
  if(NUMSO EQUAL 0)
    message(STATUS "so for the User Interface not found (${folder}${soName})" )
    set (${targetVar} "NOTFOUND")
  else()
    list(GET ANTTWEAKBAR_SO 0 ${targetVar})
  endif()
endmacro()

if(ANTTWEAKBAR_INCLUDE_DIR)

  if(UNIX)
    _find_so( ANTTWEAKBAR_DLL "libAntTweakBar.so")
    _find_so( ANTTWEAKBAR_LIB "libAntTweakBar.a")
  else(UNIX)
    _find_dll( ANTTWEAKBAR_DLL "AntTweakBar.dll" "AntTweakBar64.dll" "")
    _find_dll( ANTTWEAKBARD_DLL "AntTweakBar.dll" "AntTweakBar64.dll" "debug/")
    if(NOT ANTTWEAKBARD_DLL)
      message(STATUS "setting ANTTWEAKBARD_DLL to ${ANTTWEAKBAR_DLL}" )
    endif()
    _find_dll( ANTTWEAKBAR_LIB "AntTweakBar.lib" "AntTweakBar64.lib" "")
    _find_dll( ANTTWEAKBARD_LIB "AntTweakBar.lib" "AntTweakBar64.lib" "debug/")
    if(NOT ANTTWEAKBARD_DLL)
      message(STATUS "setting ANTTWEAKBARD_LIB to ${ANTTWEAKBAR_LIB}" )
    endif()
  endif(UNIX)
    if(ANTTWEAKBAR_DLL)
      set( ANTTWEAKBAR_FOUND "YES" )
      set( ANTTWEAKBAR_HEADERS "${ANTTWEAKBAR_INCLUDE_DIR}/AntTweakBar.h")
    endif(ANTTWEAKBAR_DLL)
else(ANTTWEAKBAR_INCLUDE_DIR)
  message(WARNING "
      AntTweakBar not found. 
      The ANTTWEAKBAR folder you would specify with ANTTWEAKBAR_LOCATION should contain:
      - lib folder: containing the ANTTWEAKBAR[64_]*.dll
      - include folder: containing the include files
      OR this folder could directly contain the dll and headers, put together
      For now, samples will run without additional UI. But that's okay ;-)"
  )
endif(ANTTWEAKBAR_INCLUDE_DIR)
include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(AntTweakBar DEFAULT_MSG
    ANTTWEAKBAR_INCLUDE_DIR
    ANTTWEAKBAR_DLL
)
# I duno why I have to rewrite the variable here...
SET(ANTTWEAKBAR_DLL ${ANTTWEAKBAR_DLL} CACHE PATH "path")

mark_as_advanced( ANTTWEAKBAR_FOUND )
