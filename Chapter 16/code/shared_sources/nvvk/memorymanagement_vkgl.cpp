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

#include "memorymanagement_vkgl.hpp"

#ifdef LINUX
#include <unistd.h>
#endif

namespace nvvk {


//////////////////////////////////////////////////////////////////////////

VkExternalMemoryHandleTypeFlags DeviceMemoryAllocatorGL::getExternalMemoryHandleTypeFlags()
{
#ifdef VK_USE_PLATFORM_WIN32_KHR
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
}

VkResult DeviceMemoryAllocatorGL::allocBlockMemory(BlockID id, VkMemoryAllocateInfo& memInfo, VkDeviceMemory& deviceMemory)
{
  BlockGL& blockGL = m_blockGLs[id.index];

  bool               isDedicated = false;
  const StructChain* extChain    = (const StructChain*)memInfo.pNext;
  while(extChain)
  {
    if(extChain->sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO)
    {
      isDedicated = true;
      break;
    }
    extChain = extChain->pNext;
  }

  // prepare memory allocation for export
  VkExportMemoryAllocateInfo exportInfo = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
  exportInfo.handleTypes = getExternalMemoryHandleTypeFlags();

  exportInfo.pNext = memInfo.pNext;
  memInfo.pNext    = &exportInfo;


  VkResult result = vkAllocateMemory(m_device, &memInfo, nullptr, &deviceMemory);
  if(result != VK_SUCCESS)
  {
    return result;
  }
  // get OS-handle (warning must not forget close)
#ifdef VK_USE_PLATFORM_WIN32_KHR
  VkMemoryGetWin32HandleInfoKHR memGetHandle = {VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
  memGetHandle.memory                        = deviceMemory;
  memGetHandle.handleType                    = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
  result                                     = vkGetMemoryWin32HandleKHR(m_device, &memGetHandle, &blockGL.handle);
#else
  VkMemoryGetFdInfoKHR memGetHandle = {VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
  memGetHandle.memory               = deviceMemory;
  memGetHandle.handleType           = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
  result                            = vkGetMemoryFdKHR(m_device, &memGetHandle, &blockGL.handle);
#endif
  if(result != VK_SUCCESS)
  {
    return result;
  }
  // import into GL
  GLint param = isDedicated ? GL_TRUE : GL_FALSE;
  glCreateMemoryObjectsEXT(1, &blockGL.memoryObject);
  glMemoryObjectParameterivEXT(blockGL.memoryObject, GL_DEDICATED_MEMORY_OBJECT_EXT, &param);
#ifdef VK_USE_PLATFORM_WIN32_KHR
  glImportMemoryWin32HandleEXT(blockGL.memoryObject, memInfo.allocationSize, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, blockGL.handle);
#else
  glImportMemoryFdEXT(blockGL.memoryObject, memInfo.allocationSize, GL_HANDLE_TYPE_OPAQUE_FD_EXT, blockGL.handle);
  // the Fd got consumed
  blockGL.handle = -1;
#endif

  return result;
}

void DeviceMemoryAllocatorGL::freeBlockMemory(BlockID id, VkDeviceMemory deviceMemory)
{
  BlockGL& blockGL = m_blockGLs[id.index];
  // free vulkan memory
  vkFreeMemory(m_device, deviceMemory, nullptr);

  glDeleteMemoryObjectsEXT(1, &blockGL.memoryObject);
  blockGL.memoryObject = 0;

  // don't forget the OS-handle it is ref-counted and can leak memory!
#ifdef VK_USE_PLATFORM_WIN32_KHR
  CloseHandle(blockGL.handle);
  blockGL.handle = NULL;
#else
  if(blockGL.handle != -1)
  {
    close(blockGL.handle);
    blockGL.handle = -1;
  }
#endif
}

void DeviceMemoryAllocatorGL::resizeBlocks(uint32_t count)
{
  if(count == 0)
  {
    m_blockGLs.clear();
  }
  else
  {
    m_blockGLs.resize(count);
  }
}

VkResult DeviceMemoryAllocatorGL::createBufferInternal(VkDevice device, const VkBufferCreateInfo* info, VkBuffer* buffer)
{
  VkBufferCreateInfo               infoNew  = *info;
  VkExternalMemoryBufferCreateInfo external = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
  external.handleTypes = getExternalMemoryHandleTypeFlags();
  external.pNext = infoNew.pNext;
  infoNew.pNext  = &external;
  return vkCreateBuffer(device, &infoNew, nullptr, buffer);
}

VkResult DeviceMemoryAllocatorGL::createImageInternal(VkDevice device, const VkImageCreateInfo* info, VkImage* image)
{
  VkImageCreateInfo               infoNew  = *info;
  VkExternalMemoryImageCreateInfo external = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
  external.handleTypes = getExternalMemoryHandleTypeFlags();
  external.pNext = infoNew.pNext;
  infoNew.pNext  = &external;
  return vkCreateImage(device, &infoNew, nullptr, image);
}

}  // namespace nvvk


#endif
