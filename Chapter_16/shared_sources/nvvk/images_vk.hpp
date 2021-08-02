/* Copyright (c) 2014-2019, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cmath>

namespace nvvk {
//////////////////////////////////////////////////////////////////////////
/**
    # functions in nvvk

    - makeImageMemoryBarrier : returns VkImageMemoryBarrier for an image based on provided layouts and access flags.
    - mipLevels : return number of mips for 2d/3d extent

    - accessFlagsForImageLayout : helps resource transtions
    - pipelineStageForLayout : helps resource transitions
    - cmdBarrierImageLayout : inserts barrier for image transition

    - cmdGenerateMipmaps : basic mipmap creation for images (meant for one-shot operations)

    - makeImage2DCreateInfo : aids 2d image creation
    - makeImage3DCreateInfo : aids 3d descriptor set updating
    - makeImageCubeCreateInfo : aids cube descriptor set updating
    - makeImageViewCreateInfo : aids common image view creation, derives info from VkImageCreateInfo
    - makeImage2DViewCreateInfo : aids 2d image view creation
  */

VkImageMemoryBarrier makeImageMemoryBarrier(VkImage            image,
                                            VkAccessFlags      srcAccess,
                                            VkAccessFlags      dstAccess,
                                            VkImageLayout      oldLayout,
                                            VkImageLayout      newLayout,
                                            VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT);


//--------------------------------------------------------------------------------------------------
inline uint32_t mipLevels(VkExtent2D extent)
{
  return static_cast<uint32_t>(std::floor(std::log2(std::max(extent.width, extent.height)))) + 1;
}

inline uint32_t mipLevels(VkExtent3D extent)
{
  return static_cast<uint32_t>(std::floor(std::log2(std::max(extent.width, extent.height)))) + 1;
}

//--------------------------------------------------------------------------------------------------
// Transition Pipeline Layout tools

VkAccessFlags        accessFlagsForImageLayout(VkImageLayout layout);
VkPipelineStageFlags pipelineStageForLayout(VkImageLayout layout);

void cmdBarrierImageLayout(VkCommandBuffer                cmdbuffer,
                           VkImage                        image,
                           VkImageLayout                  oldImageLayout,
                           VkImageLayout                  newImageLayout,
                           const VkImageSubresourceRange& subresourceRange);

void cmdBarrierImageLayout(VkCommandBuffer cmdbuffer, VkImage image, VkImageLayout oldImageLayout, VkImageLayout newImageLayout, VkImageAspectFlags aspectMask);

inline void cmdBarrierImageLayout(VkCommandBuffer cmdbuffer, VkImage image, VkImageLayout oldImageLayout, VkImageLayout newImageLayout)
{
  cmdBarrierImageLayout(cmdbuffer, image, oldImageLayout, newImageLayout, VK_IMAGE_ASPECT_COLOR_BIT);
}


VkImageCreateInfo makeImage3DCreateInfo(const VkExtent3D& size,
                                        VkFormat          format  = VK_FORMAT_R8G8B8A8_UNORM,
                                        VkImageUsageFlags usage   = VK_IMAGE_USAGE_SAMPLED_BIT,
                                        bool              mipmaps = false);


VkImageCreateInfo makeImage2DCreateInfo(const VkExtent2D& size,
                                        VkFormat          format  = VK_FORMAT_R8G8B8A8_UNORM,
                                        VkImageUsageFlags usage   = VK_IMAGE_USAGE_SAMPLED_BIT,
                                        bool              mipmaps = false);

VkImageCreateInfo makeImageCubeCreateInfo(const VkExtent2D& size,
                                          VkFormat          format  = VK_FORMAT_R8G8B8A8_UNORM,
                                          VkImageUsageFlags usage   = VK_IMAGE_USAGE_SAMPLED_BIT,
                                          bool              mipmaps = false);

// derives format and view type from imageInfo, special case for IMAGE_2D to treat as cube
// view enables all mips and layers
VkImageViewCreateInfo makeImageViewCreateInfo(VkImage image, const VkImageCreateInfo& imageInfo, bool isCube = false);


VkImageViewCreateInfo makeImage2DViewCreateInfo(VkImage            image,
                                                VkFormat           format         = VK_FORMAT_R8G8B8A8_UNORM,
                                                VkImageAspectFlags aspectFlags    = VK_IMAGE_ASPECT_COLOR_BIT,
                                                uint32_t           levels         = VK_REMAINING_MIP_LEVELS,
                                                const void*        pNextImageView = nullptr);

void cmdGenerateMipmaps(VkCommandBuffer cmdBuf, VkImage image, VkFormat imageFormat, const VkExtent2D& size, uint32_t levelCount, uint32_t layerCount = 1, VkImageLayout currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
#ifdef VULKAN_HPP

inline void cmdBarrierImageLayout(vk::CommandBuffer                cmdbuffer,
                                  vk::Image                        image,
                                  vk::ImageLayout                  oldImageLayout,
                                  vk::ImageLayout                  newImageLayout,
                                  const vk::ImageSubresourceRange& subresourceRange)
{
  cmdBarrierImageLayout(static_cast<VkCommandBuffer>(cmdbuffer), static_cast<VkImage>(image),
                        static_cast<VkImageLayout>(oldImageLayout), static_cast<VkImageLayout>(newImageLayout),
                        static_cast<const VkImageSubresourceRange&>(subresourceRange));
}

inline void cmdBarrierImageLayout(vk::CommandBuffer    cmdbuffer,
                                  vk::Image            image,
                                  vk::ImageLayout      oldImageLayout,
                                  vk::ImageLayout      newImageLayout,
                                  vk::ImageAspectFlags aspectMask)
{
  cmdBarrierImageLayout(static_cast<VkCommandBuffer>(cmdbuffer), static_cast<VkImage>(image),
                        static_cast<VkImageLayout>(oldImageLayout), static_cast<VkImageLayout>(newImageLayout),
                        static_cast<VkImageAspectFlags>(aspectMask));
}

inline void cmdBarrierImageLayout(vk::CommandBuffer cmdbuffer, vk::Image image, vk::ImageLayout oldImageLayout, vk::ImageLayout newImageLayout)
{
  cmdBarrierImageLayout(cmdbuffer, image, oldImageLayout, newImageLayout, vk::ImageAspectFlagBits::eColor);
}


inline vk::ImageCreateInfo makeImage2DCreateInfo(vk::Extent2D        size,
                                                 vk::Format          format  = vk::Format::eR8G8B8A8Unorm,
                                                 vk::ImageUsageFlags usage   = vk::ImageUsageFlagBits::eSampled,
                                                 bool                mipmaps = false)
{
  return makeImage2DCreateInfo(static_cast<VkExtent2D>(size), static_cast<VkFormat>(format),
                               static_cast<VkImageUsageFlags>(usage), mipmaps);
}

inline vk::ImageViewCreateInfo makeImage2DViewCreateInfo(vk::Image            image,
                                                         vk::Format           format,
                                                         vk::ImageAspectFlags aspectFlags = vk::ImageAspectFlagBits::eColor,
                                                         uint32_t             levels      = VK_REMAINING_MIP_LEVELS,
                                                         const void*          pNextImageView = nullptr)
{
  return makeImage2DViewCreateInfo(static_cast<VkImage>(image), static_cast<VkFormat>(format),
                                   static_cast<VkImageAspectFlags>(aspectFlags), levels, pNextImageView);
}

inline vk::ImageCreateInfo makeImageCubeCreateInfo(const vk::Extent2D& size,
                                                   vk::Format          format,
                                                   vk::ImageUsageFlags usage   = vk::ImageUsageFlagBits::eSampled,
                                                   bool                mipmaps = false)
{
  return makeImageCubeCreateInfo(static_cast<VkExtent2D>(size), static_cast<VkFormat>(format),
                                 static_cast<VkImageUsageFlags>(usage), mipmaps);
}

inline void cmdGenerateMipmaps(vk::CommandBuffer cmdBuf, vk::Image image, vk::Format imageFormat, const vk::Extent2D& size, uint32_t levelCount, uint32_t layerCount = 1, vk::ImageLayout currentLayout = vk::ImageLayout::eShaderReadOnlyOptimal)
{
  cmdGenerateMipmaps(static_cast<VkCommandBuffer>(cmdBuf), static_cast<VkImage>(image),
                     static_cast<VkFormat>(imageFormat), static_cast<VkExtent2D>(size), levelCount, layerCount, static_cast<VkImageLayout>(currentLayout));
}

inline vk::ImageCreateInfo makeImage3DCreateInfo(const vk::Extent3D& size,
                                                 vk::Format          format,
                                                 vk::ImageUsageFlags usage   = vk::ImageUsageFlagBits::eSampled,
                                                 bool                mipmaps = false)
{
  return makeImage3DCreateInfo(static_cast<VkExtent3D>(size), static_cast<VkFormat>(format),
                               static_cast<VkImageUsageFlags>(usage), mipmaps);
}

#endif


}  // namespace nvvk
