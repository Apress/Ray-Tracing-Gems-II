/* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//////////////////////////////////////////////////////////////////////////
/**

If desired some samples may want to easily switch between different
allocator classes.

This file will include the appropriate nvvk::Allocator? class depending 
on one of three possible defines that must be set prior to including:

- NVVK_ALLOC_DEDICATED : nvvk::AllocatorDedicated is a naive implementation that allocates one VkDeviceMemory per resource (VkBuffer/VkImage).
  This is not a recommended practice, but useful for basic testing and low complexity in the samples.
- NVVK_ALLOC_DMA : nvvk::AllocatorDma uses the nvvk::DeviceMemoryAllocator class to allocate VkDeviceMemory in chunks, which
  lowers the amount of overall allocations. This practice is recommended for Vulkan resource management
  in general. There are limits on the amount of allocations that can be fairly low, and there is also a performance
  penalty going through the OS to make such allocations.  
  Furthermore this allocator class uses nvvk::StagingMemoryManagerDma
  which uses multiple chunks of VkBuffers for staging memory. The motivation is the same as for memory, we reduce
  the amount of Vulkan object creations, by sub-allocating temporary space from a VkBuffer that is mapped to
  host to aid the upload process.
- NVVK_ALLOC_VMA : nvvk::AllocatorVma makes use of the [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
  and works similar to the `DMA` variant above, it also makes use of nvvk::StagingMemoryManagerVma.


It also provides structs such as nvvk::Image nvvk::Buffer etc. that map to the appropriate
structs, e.g. nvvk::ImageDma.

See more details in description of [nvvk::AllocatorDma](#class-nvvkallocatordma).
*/

#if defined(NVVK_ALLOC_DEDICATED) && !defined(NVVK_ALLOC_DMA) && !defined(NVVK_ALLOC_VMA)
#include "nvvk/allocator_dedicated_vk.hpp"
namespace nvvk {
using Buffer  = nvvk::BufferDedicated;
using Image   = nvvk::ImageDedicated;
using Texture = nvvk::TextureDedicated;
#if VK_NV_ray_tracing
using AccelNV = nvvk::AccelerationDedicatedNV;
#endif
#if VK_KHR_acceleration_structure
using AccelKHR = nvvk::AccelerationDedicatedKHR;
#endif
using Allocator    = nvvk::AllocatorDedicated;
using MemAllocator = VkPhysicalDevice;
}  // namespace nvvk
#elif defined(NVVK_ALLOC_DMA) && !defined(NVVK_ALLOC_DEDICATED) && !defined(NVVK_ALLOC_VMA)
#include "nvvk/allocator_dma_vk.hpp"
namespace nvvk {
using Buffer       = nvvk::BufferDma;
using Image        = nvvk::ImageDma;
using Texture      = nvvk::TextureDma;
#if VK_NV_ray_tracing
using AccelNV      = nvvk::AccelerationDmaNV;
#endif
#if VK_KHR_acceleration_structure
using AccelKHR     = nvvk::AccelerationDmaKHR;
#endif
using Allocator    = nvvk::AllocatorDma;
using MemAllocator = nvvk::DeviceMemoryAllocator;
}  // namespace nvvk
#elif defined(NVVK_ALLOC_VMA) && !defined(NVVK_ALLOC_DEDICATED) && !defined(NVVK_ALLOC_DMA)
#include "nvvk/allocator_vma_vk.hpp"
namespace nvvk {
using Buffer       = nvvk::BufferVma;
using Image        = nvvk::ImageVma;
using Texture      = nvvk::TextureVma;
#if VK_NV_ray_tracing
using AccelNV      = nvvk::AccelerationVmaNV;
#endif
#if VK_KHR_acceleration_structure
using AccelKHR     = nvvk::AccelerationVmaKHR;
#endif
using Allocator    = nvvk::AllocatorVma;
using MemAllocator = VmaAllocator;
}  // namespace nvvk
#else
#error "no, or multiple NVVK_ALLOC set"
#endif
