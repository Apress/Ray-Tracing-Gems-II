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

#include "images_vk.hpp"
#include <assert.h>

namespace nvvk {

VkImageMemoryBarrier makeImageMemoryBarrier(VkImage            img,
                                            VkAccessFlags      srcAccess,
                                            VkAccessFlags      dstAccess,
                                            VkImageLayout      oldLayout,
                                            VkImageLayout      newLayout,
                                            VkImageAspectFlags aspectMask)
{
  VkImageMemoryBarrier barrier        = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.srcAccessMask               = srcAccess;
  barrier.dstAccessMask               = dstAccess;
  barrier.oldLayout                   = oldLayout;
  barrier.newLayout                   = newLayout;
  barrier.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
  barrier.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
  barrier.image                       = img;
  barrier.subresourceRange            = {0};
  barrier.subresourceRange.aspectMask = aspectMask;
  barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
  barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

  return barrier;
}

///////////////////////////////////////////////////////////////////////////////
// Return the access flag for an image layout
VkAccessFlags accessFlagsForImageLayout(VkImageLayout layout)
{
  switch(layout)
  {
    case VK_IMAGE_LAYOUT_PREINITIALIZED:
      return VK_ACCESS_HOST_WRITE_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
      return VK_ACCESS_TRANSFER_WRITE_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
      return VK_ACCESS_TRANSFER_READ_BIT;
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
      return VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
      return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
      return VK_ACCESS_SHADER_READ_BIT;
    default:
      return VkAccessFlags();
  }
}

VkPipelineStageFlags pipelineStageForLayout(VkImageLayout layout)
{
  switch(layout)
  {
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
      return VK_PIPELINE_STAGE_TRANSFER_BIT;
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
      return VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
      return VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
      return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    case VK_IMAGE_LAYOUT_PREINITIALIZED:
      return VK_PIPELINE_STAGE_HOST_BIT;
    case VK_IMAGE_LAYOUT_UNDEFINED:
      return VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    default:
      return VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  }
}

void cmdBarrierImageLayout(VkCommandBuffer                cmdbuffer,
                           VkImage                        image,
                           VkImageLayout                  oldImageLayout,
                           VkImageLayout                  newImageLayout,
                           const VkImageSubresourceRange& subresourceRange)
{
  // Create an image barrier to change the layout
  VkImageMemoryBarrier imageMemoryBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  imageMemoryBarrier.oldLayout        = oldImageLayout;
  imageMemoryBarrier.newLayout        = newImageLayout;
  imageMemoryBarrier.image            = image;
  imageMemoryBarrier.subresourceRange = subresourceRange;
  imageMemoryBarrier.srcAccessMask    = accessFlagsForImageLayout(oldImageLayout);
  imageMemoryBarrier.dstAccessMask    = accessFlagsForImageLayout(newImageLayout);
  VkPipelineStageFlags srcStageMask   = pipelineStageForLayout(oldImageLayout);
  VkPipelineStageFlags destStageMask  = pipelineStageForLayout(newImageLayout);
  vkCmdPipelineBarrier(cmdbuffer, srcStageMask, destStageMask, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
}

void cmdBarrierImageLayout(VkCommandBuffer cmdbuffer, VkImage image, VkImageLayout oldImageLayout, VkImageLayout newImageLayout, VkImageAspectFlags aspectMask)
{
  VkImageSubresourceRange subresourceRange;
  subresourceRange.aspectMask     = aspectMask;
  subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
  subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;
  subresourceRange.baseMipLevel   = 0;
  subresourceRange.baseArrayLayer = 0;
  cmdBarrierImageLayout(cmdbuffer, image, oldImageLayout, newImageLayout, subresourceRange);
}

VkImageCreateInfo makeImage2DCreateInfo(const VkExtent2D& size, VkFormat format, VkImageUsageFlags usage, bool mipmaps)
{
  VkImageCreateInfo icInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  icInfo.imageType         = VK_IMAGE_TYPE_2D;
  icInfo.format            = format;
  icInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
  icInfo.mipLevels         = mipmaps ? mipLevels(size) : 1;
  icInfo.arrayLayers       = 1;
  icInfo.extent.width      = size.width;
  icInfo.extent.height     = size.height;
  icInfo.extent.depth      = 1;
  icInfo.usage             = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  return icInfo;
}


VkImageViewCreateInfo makeImage2DViewCreateInfo(VkImage            image,
                                                VkFormat           format /*= VK_FORMAT_R8G8B8A8_UNORM*/,
                                                VkImageAspectFlags aspectFlags /*= VK_IMAGE_ASPECT_COLOR_BIT*/,
                                                uint32_t           levels /*= 1*/,
                                                const void*        pNextImageView /*= nullptr*/)
{
  VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  viewInfo.pNext                           = pNextImageView;
  viewInfo.image                           = image;
  viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format                          = format;
  viewInfo.subresourceRange.aspectMask     = aspectFlags;
  viewInfo.subresourceRange.baseMipLevel   = 0;
  viewInfo.subresourceRange.levelCount     = levels;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount     = 1;

  return viewInfo;
}

VkImageViewCreateInfo makeImageViewCreateInfo(VkImage image, const VkImageCreateInfo& imageInfo, bool isCube)
{
  VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  viewInfo.pNext = nullptr;
  viewInfo.image = image;

  switch(imageInfo.imageType)
  {
    case VK_IMAGE_TYPE_1D:
      viewInfo.viewType = VK_IMAGE_VIEW_TYPE_1D;
      break;
    case VK_IMAGE_TYPE_2D:
      viewInfo.viewType = isCube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
      break;
    case VK_IMAGE_TYPE_3D:
      viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
      break;
    default:
      assert(0);
  }

  viewInfo.format                          = imageInfo.format;
  viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseMipLevel   = 0;
  viewInfo.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;

  return viewInfo;
}

VkImageCreateInfo makeImage3DCreateInfo(const VkExtent3D& size, VkFormat format, VkImageUsageFlags usage, bool mipmaps)
{
  VkImageCreateInfo icInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  icInfo.imageType         = VK_IMAGE_TYPE_3D;
  icInfo.format            = format;
  icInfo.mipLevels         = mipmaps ? mipLevels(size) : 1;
  icInfo.arrayLayers       = 1;
  icInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
  icInfo.extent.width      = size.width;
  icInfo.extent.height     = size.height;
  icInfo.extent.depth      = size.depth;
  icInfo.usage             = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  return icInfo;
}

VkImageCreateInfo makeImageCubeCreateInfo(const VkExtent2D& size, VkFormat format, VkImageUsageFlags usage, bool mipmaps)
{
  VkImageCreateInfo icInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  icInfo.imageType     = VK_IMAGE_TYPE_2D;
  icInfo.format        = format;
  icInfo.mipLevels     = mipmaps ? mipLevels(size) : 1;
  icInfo.arrayLayers   = 6;
  icInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
  icInfo.extent.width  = size.width;
  icInfo.extent.height = size.height;
  icInfo.extent.depth  = 1;
  icInfo.usage         = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  icInfo.flags         = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  return icInfo;
}

// This mipmap generation relies on blitting
// A more sophisticated version could be done with computer shader
// We will publish how to in the future

void cmdGenerateMipmaps(VkCommandBuffer cmdBuf, VkImage image, VkFormat imageFormat, const VkExtent2D& size, uint32_t levelCount, uint32_t layerCount, VkImageLayout currentLayout)
{
  // Transfer the top level image to a layout 'eTransferSrcOptimal` and its access to 'eTransferRead'
  VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel   = 0;
  barrier.subresourceRange.layerCount     = layerCount;
  barrier.subresourceRange.levelCount     = 1;
  barrier.image                           = image;
  barrier.oldLayout                       = currentLayout;
  barrier.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barrier.srcAccessMask                   = accessFlagsForImageLayout(currentLayout);
  barrier.dstAccessMask                   = VK_ACCESS_TRANSFER_READ_BIT;
  barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
  vkCmdPipelineBarrier(cmdBuf, pipelineStageForLayout(currentLayout), VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);

  if(levelCount > 1)
  {
    // transfer remaining mips to DST optimal
    barrier.newLayout                     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.dstAccessMask                 = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.subresourceRange.baseMipLevel = 1;
    barrier.subresourceRange.levelCount   = VK_REMAINING_MIP_LEVELS;
    vkCmdPipelineBarrier(cmdBuf, pipelineStageForLayout(currentLayout), VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &barrier);
  };

  int32_t mipWidth  = size.width;
  int32_t mipHeight = size.height;

  for(uint32_t i = 1; i < levelCount; i++)
  {

    VkImageBlit blit;
    blit.srcOffsets[0]                 = {0, 0, 0};
    blit.srcOffsets[1]                 = {mipWidth, mipHeight, 1};
    blit.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.mipLevel       = i - 1;
    blit.srcSubresource.baseArrayLayer = 0;
    blit.srcSubresource.layerCount     = layerCount;
    blit.dstOffsets[0]                 = {0, 0, 0};
    blit.dstOffsets[1]                 = {mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1};
    blit.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.dstSubresource.mipLevel       = i;
    blit.dstSubresource.baseArrayLayer = 0;
    blit.dstSubresource.layerCount     = layerCount;

    vkCmdBlitImage(cmdBuf, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                   &blit, VK_FILTER_LINEAR);


    // Next
    {
      // Transition the current miplevel into a eTransferSrcOptimal layout, to be used as the source for the next one.
      barrier.subresourceRange.baseMipLevel = i;
      barrier.subresourceRange.levelCount   = 1;
      barrier.oldLayout                     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      barrier.newLayout                     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      barrier.srcAccessMask                 = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask                 = VK_ACCESS_TRANSFER_READ_BIT;
      vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                           nullptr, 1, &barrier);
    }


    if(mipWidth > 1)
      mipWidth /= 2;
    if(mipHeight > 1)
      mipHeight /= 2;
  }

  // Transition all miplevels (now in VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) back to currentLayout
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount   = VK_REMAINING_MIP_LEVELS;
  barrier.oldLayout                     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barrier.newLayout                     = currentLayout;
  barrier.srcAccessMask                 = VK_ACCESS_TRANSFER_READ_BIT;
  barrier.dstAccessMask                 = accessFlagsForImageLayout(currentLayout);
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, pipelineStageForLayout(currentLayout), 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);
}

}  // namespace nvvk
