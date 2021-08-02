# Try to find DirectX 11 SDK.
# Once done this will define
#
# DX11SDK_FOUND
# DX11SDK_INCLUDE_DIR
# DX11SDK_LIBRARY_DIR
# DX11SDK_D3D_LIBRARIES



if (WIN32)

    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
        set (ARCH x64)
    else ()
        set (ARCH x86)
    endif ()
    
    
    find_path(DX11SDK_INCLUDE_DIR
        NAMES
            D3D11.h D3Dcompiler.h
        PATHS
            ${DXSDK_LOCATION}/Include
            $ENV{DXSDK_LOCATION}/Include
            ${DXSDK_ROOT}/Include
            $ENV{DXSDK_ROOT}/Include
            "C:/Program Files (x86)/Windows Kits/8.1/Include/um"
            "C:/Program Files (x86)/Microsoft DirectX SDK*/Include"
            "C:/Program Files/Microsoft DirectX SDK*/Include"
    )

    find_path(DX11SDK_LIBRARY_DIR
            d3d11.lib 
        PATHS
            ${DXSDK_LOCATION}/Lib/${ARCH}
            $ENV{DXSDK_LOCATION}/Lib/${ARCH}
            ${DXSDK_ROOT}/Lib/${ARCH}
            $ENV{DXSDK_ROOT}/Lib/${ARCH}
            "C:/Program Files (x86)/Windows Kits/8.1/Lib/winv6.3/um/${ARCH}"
            "C:/Program Files (x86)/Microsoft DirectX SDK*/Lib/${ARCH}"
            "C:/Program Files/Microsoft DirectX SDK*/Lib/${ARCH}"
    )


    foreach(DX_LIB d3d11 d3dcompiler)

        find_library(DX11SDK_${DX_LIB}_LIBRARY
            NAMES 
                ${DX_LIB}.lib
            PATHS
                ${DX11SDK_LIBRARY_DIR}
        )

        list(APPEND DX11SDK_D3D_LIBRARIES ${DX11SDK_${DX_LIB}_LIBRARY})


    endforeach(DX_LIB)

endif ()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(DX11SDK DEFAULT_MSG
    DX11SDK_INCLUDE_DIR
    DX11SDK_LIBRARY_DIR
    DX11SDK_D3D_LIBRARIES
)

mark_as_advanced(
    DX11SDK_INCLUDE_DIR
    DX11SDK_LIBRARY_DIR
    DX11SDK_D3D_LIBRARIES
)