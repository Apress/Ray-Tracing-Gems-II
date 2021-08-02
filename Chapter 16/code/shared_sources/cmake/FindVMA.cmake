# Try to find Vulkan Memeory Allocator
#
unset(VMA_INCLUDE_DIR CACHE)
unset(VMA_FOUND CACHE)

find_path( VMA_INCLUDE_DIR vk_mem_alloc.h
  ${PROJECT_SOURCE_DIR}/shared_external/VMA/src
  ${PROJECT_SOURCE_DIR}/../shared_external/VMA/src
  ${PROJECT_SOURCE_DIR}/../../shared_external/VMA/src
  ${PROJECT_SOURCE_DIR}/../../../shared_external/VMA/src
  ${PROJECT_SOURCE_DIR}/../../../../shared_external/VMA/src
  ${VMA_LOCATION}/src
  $ENV{VMA_LOCATION}/src
)

if(VMA_INCLUDE_DIR)
  set( VMA_FOUND "YES" )
  set( VMA_HEADERS "${VMA_INCLUDE_DIR}/vk_mem_alloc.h")
else(VMA_INCLUDE_DIR)
  message(WARNING "
      Vulkan Memory Allocator not found. 
      The VMA folder you would specify with VMA_LOCATION should contain:
      src folder: containing the include file vk_mem_alloc.h"
  )
endif(VMA_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(VMA DEFAULT_MSG
    VMA_INCLUDE_DIR
)
mark_as_advanced( VMA_FOUND )
