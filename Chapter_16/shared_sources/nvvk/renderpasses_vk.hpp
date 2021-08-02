/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include <vector>
#include <vulkan/vulkan_core.h>

namespace nvvk {
/**
  # functions in nvvk

  - findSupportedFormat : returns supported VkFormat from a list of candidates (returns first match)
  - findDepthFormat : returns supported depth format (24, 32, 16-bit)
  - findDepthStencilFormat : returns supported depth-stencil format (24/8, 32/8, 16/8-bit)
  - createRenderPass : wrapper for vkCreateRenderPass

*/
VkFormat findSupportedFormat(VkPhysicalDevice physicalDevice, const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
VkFormat findDepthFormat(VkPhysicalDevice physicalDevice);
VkFormat findDepthStencilFormat(VkPhysicalDevice physicalDevice);

//////////////////////////////////////////////////////////////////////////

VkRenderPass createRenderPass(VkDevice                     device,
                              const std::vector<VkFormat>& colorAttachmentFormats,
                              VkFormat                     depthAttachmentFormat,
                              uint32_t                     subpassCount  = 1,
                              bool                         clearColor    = true,
                              bool                         clearDepth    = true,
                              VkImageLayout                initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                              VkImageLayout                finalLayout   = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);


#ifdef VULKAN_HPP
inline vk::Format findDepthFormat(vk::PhysicalDevice physicalDevice)
{
  return (vk::Format)findDepthFormat((VkPhysicalDevice)physicalDevice);
}

inline vk::Format findDepthStencilFormat(vk::PhysicalDevice physicalDevice)
{
  return (vk::Format)findDepthStencilFormat((VkPhysicalDevice)physicalDevice);
}

inline vk::Format findSupportedFormat(vk::PhysicalDevice             physicalDevice,
                                      const std::vector<vk::Format>& candidates,
                                      vk::ImageTiling                tiling,
                                      vk::FormatFeatureFlags         features)
{
  return (vk::Format)findSupportedFormat((VkPhysicalDevice)physicalDevice, (const std::vector<VkFormat>&)candidates,
                                         (VkImageTiling)tiling, (VkFormatFeatureFlags)features);
}
inline vk::RenderPass createRenderPass(vk::Device                     device,
                                       const std::vector<vk::Format>& colorAttachmentFormats,
                                       vk::Format                     depthAttachmentFormat,
                                       uint32_t                       subpassCount  = 1,
                                       bool                           clearColor    = true,
                                       bool                           clearDepth    = true,
                                       vk::ImageLayout                initialLayout = vk::ImageLayout::eUndefined,
                                       vk::ImageLayout                finalLayout   = vk::ImageLayout::ePresentSrcKHR)
{
  return (vk::RenderPass)createRenderPass((VkDevice)device, (const std::vector<VkFormat>&)colorAttachmentFormats,
                                          (VkFormat)depthAttachmentFormat, subpassCount, clearColor, clearDepth,
                                          (VkImageLayout)initialLayout, (VkImageLayout)finalLayout);
}
#endif

}  // namespace nvvk
