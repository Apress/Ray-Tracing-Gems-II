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

#if USEOPENGL

#include <vulkan/vulkan.h>

#include "allocator_dma_vkgl.hpp"

namespace nvvk {

//////////////////////////////////////////////////////////////////////////

nvvk::BufferDmaGL AllocatorDmaGL::createBuffer(VkCommandBuffer       cmd,
                                               VkDeviceSize          size,
                                               VkBufferUsageFlags    usage,
                                               const void*           data /*= nullptr*/,
                                               VkMemoryPropertyFlags memProps /*= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT*/)
{
  BufferDmaGL resultBuffer = createBuffer(size, usage, memProps);
  if(data)
  {
    m_staging.cmdToBuffer(cmd, resultBuffer.buffer, 0, size, data);
  }

  return resultBuffer;
}

nvvk::BufferDmaGL AllocatorDmaGL::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps /*= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT*/)
{
  BufferDmaGL resultBuffer;
  resultBuffer.buffer = m_allocator->createBuffer(size, usage, resultBuffer.allocation, memProps);

  AllocationGL alloc = m_allocator->getAllocationGL(resultBuffer.allocation);
  glCreateBuffers(1, &resultBuffer.bufferGL);
  glNamedBufferStorageMemEXT(resultBuffer.bufferGL, alloc.size, alloc.memoryObject, alloc.offset);

  return resultBuffer;
}

nvvk::BufferDmaGL AllocatorDmaGL::createBuffer(const VkBufferCreateInfo& info,
                                               VkMemoryPropertyFlags memProps /*= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT*/)
{
  BufferDmaGL resultBuffer;
  resultBuffer.buffer = m_allocator->createBuffer(info, resultBuffer.allocation, memProps);

  AllocationGL alloc = m_allocator->getAllocationGL(resultBuffer.allocation);
  glCreateBuffers(1, &resultBuffer.bufferGL);
  glNamedBufferStorageMemEXT(resultBuffer.bufferGL, info.size, alloc.memoryObject, alloc.offset);

  return resultBuffer;
}

nvvk::ImageDmaGL AllocatorDmaGL::createImage(VkCommandBuffer          cmd,
                                             const VkImageCreateInfo& info,
                                             GLenum                   formatGL,
                                             VkImageLayout            layout,
                                             VkDeviceSize             size,
                                             const void*              data,
                                             VkMemoryPropertyFlags memProps /*= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT*/)
{
  ImageDmaGL resultImage = createImage(info, formatGL, memProps);

  // Copy the data to staging buffer than to image
  if(data != nullptr)
  {
    nvvk::cmdBarrierImageLayout(cmd, resultImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkOffset3D               offset      = {0};
    VkImageSubresourceLayers subresource = {0};
    subresource.aspectMask               = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource.layerCount               = 1;

    m_staging.cmdToImage(cmd, resultImage.image, offset, info.extent, subresource, size, data);

    // Setting final image layout
    nvvk::cmdBarrierImageLayout(cmd, resultImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, layout);
  }
  else
  {
    // Setting final image layout
    nvvk::cmdBarrierImageLayout(cmd, resultImage.image, VK_IMAGE_LAYOUT_UNDEFINED, layout);
  }

  return resultImage;
}


nvvk::ImageDmaGL AllocatorDmaGL::createImage(const VkImageCreateInfo& info, GLenum formatGL, VkMemoryPropertyFlags memProps /*= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT*/)
{
  ImageDmaGL resultImage;
  resultImage.image = m_allocator->createImage(info, resultImage.allocation, memProps);

  AllocationGL alloc = m_allocator->getAllocationGL(resultImage.allocation);

  GLenum tilingGL;
  switch(info.tiling)
  {
    case VK_IMAGE_TILING_LINEAR:
      tilingGL = GL_LINEAR_TILING_EXT;
      break;
    case VK_IMAGE_TILING_OPTIMAL:
      tilingGL = GL_OPTIMAL_TILING_EXT;
      break;
    default:
      assert(0 && "illegal tiling type");
  }

  GLuint sampleCount = (GLuint)info.samples;

  switch(info.imageType)
  {
    case VK_IMAGE_TYPE_1D:
      glCreateTextures(info.arrayLayers > 1 ? GL_TEXTURE_1D_ARRAY : GL_TEXTURE_1D, 1, &resultImage.texGL);
      glTextureParameteri(resultImage.texGL, GL_TEXTURE_TILING_EXT, tilingGL);

      if(info.arrayLayers > 1)
      {
        glTextureStorageMem2DEXT(resultImage.texGL, info.mipLevels, formatGL, info.extent.width, info.arrayLayers,
                                 alloc.memoryObject, alloc.offset);
      }
      else
      {
        glTextureStorageMem1DEXT(resultImage.texGL, info.mipLevels, formatGL, info.extent.width, alloc.memoryObject,
                                 alloc.offset);
      }
      break;
    case VK_IMAGE_TYPE_2D:
      if(sampleCount > 1)
      {
        // FIXME check if this is true
        GLenum fixedLocations = info.flags & VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT ? GL_FALSE : GL_TRUE;

        glCreateTextures(info.arrayLayers > 1 ? GL_TEXTURE_2D_MULTISAMPLE_ARRAY : GL_TEXTURE_2D_MULTISAMPLE, 1,
                         &resultImage.texGL);
        glTextureParameteri(resultImage.texGL, GL_TEXTURE_TILING_EXT, tilingGL);
        if(info.arrayLayers > 1)
        {
          glTextureStorageMem3DMultisampleEXT(resultImage.texGL, sampleCount, formatGL, info.extent.width, info.extent.height,
                                              info.arrayLayers, fixedLocations, alloc.memoryObject, alloc.offset);
        }
        else
        {
          glTextureStorageMem2DMultisampleEXT(resultImage.texGL, sampleCount, formatGL, info.extent.width,
                                              info.extent.height, fixedLocations, alloc.memoryObject, alloc.offset);
        }
      }
      else
      {
        glCreateTextures(info.arrayLayers > 1 ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D, 1, &resultImage.texGL);
        glTextureParameteri(resultImage.texGL, GL_TEXTURE_TILING_EXT, tilingGL);
        if(info.arrayLayers > 1)
        {
          glTextureStorageMem3DEXT(resultImage.texGL, info.mipLevels, formatGL, info.extent.width, info.extent.height,
                                   info.arrayLayers, alloc.memoryObject, alloc.offset);
        }
        else
        {
          glTextureStorageMem2DEXT(resultImage.texGL, info.mipLevels, formatGL, info.extent.width, info.extent.height,
                                   alloc.memoryObject, alloc.offset);
        }
      }
      break;
    case VK_IMAGE_TYPE_3D:
      glCreateTextures(GL_TEXTURE_3D, 1, &resultImage.texGL);
      glTextureParameteri(resultImage.texGL, GL_TEXTURE_TILING_EXT, tilingGL);
      glTextureStorageMem3DEXT(resultImage.texGL, info.mipLevels, formatGL, info.extent.width, info.extent.height,
                               info.extent.depth, alloc.memoryObject, alloc.offset);
      break;
  }

  return resultImage;
}

void AllocatorDmaGL::destroy(BufferDmaGL& buffer)
{
  if(buffer.bufferGL)
  {
    glDeleteBuffers(1, &buffer.bufferGL);
  }
  if(buffer.buffer)
  {
    vkDestroyBuffer(m_device, buffer.buffer, nullptr);
  }
  if(buffer.allocation)
  {
    m_allocator->free(buffer.allocation);
  }

  buffer = BufferDmaGL();
}

void AllocatorDmaGL::destroy(ImageDmaGL& image)
{
  if(image.texGL)
  {
    glDeleteTextures(1, &image.texGL);
  }
  if(image.image)
  {
    vkDestroyImage(m_device, image.image, nullptr);
  }
  if(image.allocation)
  {
    m_allocator->free(image.allocation);
  }

  image = ImageDmaGL();
}

}  // namespace nvvk


#endif
