# Try to find DirectX SDK.
# Once done this will define
#
# DX12SDK_FOUND
# DX12SDK_INCLUDE_DIR
# DX12SDK_LIBRARY_DIR
# DX12SDK_D3D_LIBRARIES


if (WIN32)

    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
        set (ARCH x64)
    else ()
        set (ARCH x86)
    endif ()
    
    
    find_path(DX12SDK_INCLUDE_DIR
        NAMES
            D3D12.h D3Dcompiler.h
        PATHS
            "C:/Program Files (x86)/Windows Kits/10/Include/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/um"
    )

    find_path(DX12SDK_LIBRARY_DIR
            d3d12.lib 
        PATHS
            "C:/Program Files (x86)/Windows Kits/10/Lib/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/um/x64"
    )


    foreach(DX_LIB d3d12 d3dcompiler dxgi)

        find_library(DX12SDK_${DX_LIB}_LIBRARY
            NAMES 
                ${DX_LIB}.lib
            PATHS
                ${DX12SDK_LIBRARY_DIR}
        )

        list(APPEND DX12SDK_D3D_LIBRARIES ${DX12SDK_${DX_LIB}_LIBRARY})


    endforeach(DX_LIB)

endif ()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(DX12SDK DEFAULT_MSG
    DX12SDK_INCLUDE_DIR
    DX12SDK_LIBRARY_DIR
    DX12SDK_D3D_LIBRARIES
)

mark_as_advanced(
    DX12SDK_INCLUDE_DIR
    DX12SDK_LIBRARY_DIR
    DX12SDK_D3D_LIBRARIES
)