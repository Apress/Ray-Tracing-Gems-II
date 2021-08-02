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

#ifndef NV_VK_SWAPCHAIN_INCLUDED
#define NV_VK_SWAPCHAIN_INCLUDED


#include <stdio.h>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace nvvk {

/**
# class nvvk::SwapChain

Its role is to help using VkSwapchainKHR. In Vulkan we have 
to synchronize the backbuffer access ourselves, meaning we
must not write into images that the operating system uses for
presenting the image on the desktop or monitor.

For each swapchain image there is an imageView,
and one read and write semaphore. Furthermore there
is a utility function to setup the image transitions from 
VK_IMAGE_LAYOUT_UNDEFINED to VK_IMAGE_LAYOUT_PRESENT_SRC_KHR.

Example in combination with nvvk::Context :

* get the window handle
* create its related surface
* make sure the Queue is the one we need to render in this surface

~~~ C++
// could be arguments of a function/method :
nvvk::Context ctx;
NVPWindow     win;
...

// get the surface of the window in which to render
VkWin32SurfaceCreateInfoKHR createInfo = {};
... populate the fields of createInfo ...
createInfo.hwnd = glfwGetWin32Window(win.m_internal);
result = vkCreateWin32SurfaceKHR(ctx.m_instance, &createInfo, nullptr, &m_surface);

...
// make sure we assign the proper Queue to m_queueGCT, from what the surface tells us
ctx.setGCTQueueWithPresent(m_surface);
~~~

The initialization can happen now :

~~~ C+
m_swapChain.init(ctx.m_device, ctx.m_physicalDevice, ctx.m_queueGCT, ctx.m_queueGCT.familyIndex, 
                 m_surface, VK_FORMAT_B8G8R8A8_UNORM);
...
// after init or update you also have to setup the image layouts at some point
VkCommandBuffer cmd = ...
m_swapChain.cmdUpdateBarriers(cmd);
~~~

During a resizing of a window, you must update the swapchain as well :

~~~ C++
bool WindowSurface::resize(int w, int h)
{
...
  m_swapChain.update(w, h);
  // be cautious to also transition the image layouts
...
}
~~~


A typical renderloop would look as follows:

~~~ C++
  // handles vkAcquireNextImageKHR and setting the active image
  if(!m_swapChain.acquire())
  {
    ... handle acquire error
  }

  VkCommandBuffer cmd = ...

  if (m_swapChain.getChangeID() != lastChangeID){
    // after init or resize you have to setup the image layouts
    m_swapChain.cmdUpdateBarriers(cmd);

    lastChangeID = m_swapChain.getChangeID();
  }

  // do render operations either directly using the imageview
  VkImageView swapImageView = m_swapChain.getActiveImageView();

  // or you may always render offline int your own framebuffer
  // and then simply blit into the backbuffer
  VkImage swapImage = m_swapChain.getActiveImage();
  vkCmdBlitImage(cmd, ... swapImage ...);

  // setup submit
  VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers    = &cmd;

  // we need to ensure to wait for the swapchain image to have been read already
  // so we can safely blit into it

  VkSemaphore swapchainReadSemaphore      = m_swapChain->getActiveReadSemaphore();
  VkPipelineStageFlags swapchainReadFlags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores    = &swapchainReadSemaphore;
  submitInfo.pWaitDstStageMask  = &swapchainReadFlags);

  // once this submit completed, it means we have written the swapchain image
  VkSemaphore swapchainWrittenSemaphore = m_swapChain->getActiveWrittenSemaphore();
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores    = &swapchainWrittenSemaphore;

  // submit it
  vkQueueSubmit(m_queue, 1, &submitInfo, fence);

  // present via a queue that supports it
  // this will also setup the dependency for the appropriate written semaphore
  // and bump the semaphore cycle
  m_swapChain.present(m_queue);
~~~

*/
class SwapChain
{
private:
  struct Entry
  {
    VkImage     image{};
    VkImageView imageView{};
    // be aware semaphore index may not match active image index
    VkSemaphore readSemaphore{};
    VkSemaphore writtenSemaphore{};
  };

  VkDevice         m_device         = VK_NULL_HANDLE;
  VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;

  VkQueue  m_queue{};
  uint32_t m_queueFamilyIndex{0};

  VkSurfaceKHR    m_surface{};
  VkFormat        m_surfaceFormat{};
  VkColorSpaceKHR m_surfaceColor{};

  uint32_t       m_imageCount{0};
  VkSwapchainKHR m_swapchain{};

  std::vector<Entry>                m_entries;
  std::vector<VkImageMemoryBarrier> m_barriers;

  // index for current image, returned by vkAcquireNextImageKHR
  // vk spec: The order in which images are acquired is implementation-dependent,
  // and may be different than the order the images were presented
  uint32_t m_currentImage{0};
  // index for current semaphore, incremented by `SwapChain::present`
  uint32_t m_currentSemaphore{0};
  // incremented by `SwapChain::update`, use to update other resources or track changes
  uint32_t m_changeID{0};
  // surface
  VkExtent2D m_extent{0,0};
  // requested on update
  uint32_t m_updateWidth{0};
  uint32_t m_updateHeight{0};
  // if the swap operation is sync'ed with monitor
  bool m_vsync = false;

  // triggers device wait idle
  void deinitResources();

public:
  SwapChain(SwapChain const&) = delete;
  SwapChain& operator=(SwapChain const&) = delete;

  SwapChain() {}
  SwapChain(VkDevice device, VkPhysicalDevice physicalDevice, VkQueue queue, uint32_t queueFamilyIndex, VkSurfaceKHR surface, VkFormat format = VK_FORMAT_B8G8R8A8_UNORM)
  {
    init(device, physicalDevice, queue, queueFamilyIndex, surface, format);
  }
  ~SwapChain() { deinit(); }

  bool init(VkDevice device, VkPhysicalDevice physicalDevice, VkQueue queue, uint32_t queueFamilyIndex, VkSurfaceKHR surface, VkFormat format = VK_FORMAT_B8G8R8A8_UNORM);

  // triggers device wait idle
  void deinit();

  // update the swapchain configuration
  // (must be called at least once after init)
  // triggers device wait idle
  // returns actual swapchain dimensions, which may differ from requested
  VkExtent2D update(int width, int height, bool vsync);
  VkExtent2D update(int width, int height) { return update(width, height, m_vsync); }

  // returns true on success
  // sets active index
  // uses getActiveReadSemaphore()
  bool acquire();

  // returns true on success
  // sets active index
  // allows to provide your own semaphore
  bool acquireCustom(VkSemaphore semaphore);

  // all present functions bump semaphore cycle

  // present on provided queue
  void present(VkQueue queue);
  // present using a default queue from init time
  void present() { present(m_queue); }
  // present via a custom function
  // (e.g. when extending via VkDeviceGroupPresentInfoKHR)
  // fills in defaults for provided presentInfo
  // with getActiveImageIndex()
  // and getActiveWrittenSemaphore()
  void presentCustom(VkPresentInfoKHR& outPresentInfo);

  VkSemaphore getActiveReadSemaphore() const;
  VkSemaphore getActiveWrittenSemaphore() const;
  VkImage     getActiveImage() const;
  VkImageView getActiveImageView() const;
  uint32_t    getActiveImageIndex() const { return m_currentImage; }

  uint32_t       getImageCount() const { return m_imageCount; }
  VkImage        getImage(uint32_t i) const;
  VkImageView    getImageView(uint32_t i) const;
  VkFormat       getFormat() const { return m_surfaceFormat; }
  uint32_t       getWidth() const { return m_extent.width; }
  uint32_t       getHeight() const { return m_extent.height; }
  VkExtent2D     getExtent() const { return m_extent; }
  uint32_t       getUpdateWidth() const { return m_updateWidth; }
  uint32_t       getUpdateHeight() const { return m_updateHeight; }
  bool           getVsync() const { return m_vsync; }
  VkSwapchainKHR getSwapchain() const { return m_swapchain; }

  // does a vkCmdPipelineBarrier for VK_IMAGE_LAYOUT_UNDEFINED to VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
  // must apply resource transitions after update calls
  void cmdUpdateBarriers(VkCommandBuffer cmd) const;

  uint32_t getChangeID() const;
};
}  // namespace nvvk
#endif
