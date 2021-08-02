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

#pragma once

#include <platform.h>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace nvvk {

//--------------------------------------------------------------------------------------------------
/**
# functions in nvvk

- makeAccessMaskPipelineStageFlags : depending on accessMask returns appropriate VkPipelineStageFlagBits
- cmdBegin : wraps vkBeginCommandBuffer with VkCommandBufferUsageFlags and implicitly handles VkCommandBufferBeginInfo setup
- makeSubmitInfo : VkSubmitInfo struct setup using provided arrays of signals and commandbuffers, leaving rest zeroed
*/

// useful for barriers, derive all compatible stage flags from an access mask


uint32_t makeAccessMaskPipelineStageFlags(uint32_t accessMask,
                                          VkPipelineStageFlags supportedShaderBits = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT
                                                                                     | VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT
                                                                                     | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                                                                                     | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

void cmdBegin(VkCommandBuffer cmd, VkCommandBufferUsageFlags flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

inline VkSubmitInfo makeSubmitInfo(uint32_t numCmds, VkCommandBuffer* cmds, uint32_t numSignals, VkSemaphore* signals)
{
  VkSubmitInfo submitInfo         = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.pCommandBuffers      = cmds;
  submitInfo.commandBufferCount   = numCmds;
  submitInfo.pSignalSemaphores    = signals;
  submitInfo.signalSemaphoreCount = numSignals;

  return submitInfo;
}

//--------------------------------------------------------------------------------------------------
/**
  # class nvvk::CommandPool

  CommandPool stores a single VkCommandPool and provides utility functions
  to create VkCommandBuffers from it.

  Example:
  ``` C++
  {
    nvvk::CommandPool cmdPool;
    cmdPool.init(...);

    // some setup/one shot work
    {
      vkCommandBuffer cmd = scopePool.createAndBegin();
      ... record commands ...
      // trigger execution with a blocking operation
      // not recommended for performance
      // but useful for sample setup
      scopePool.submitAndWait(cmd, queue);
    }

    // other cmds you may batch, or recycle
    std::vector<VkCommandBuffer> cmds;
    {
      vkCommandBuffer cmd = scopePool.createAndBegin();
      ... record commands ...
      cmds.push_back(cmd);
    }
    {
      vkCommandBuffer cmd = scopePool.createAndBegin();
      ... record commands ...
      cmds.push_back(cmd);
    }

    // do some form of batched submission of cmds

    // after completion destroy cmd
    cmdPool.destroy(cmds.size(), cmds.data());
    cmdPool.deinit();
  }
  ```
*/

class CommandPool
{
public:
  CommandPool(CommandPool const&) = delete;
  CommandPool& operator=(CommandPool const&) = delete;

  CommandPool() {}
  ~CommandPool() { deinit(); }

  // if defaultQueue is null, uses first queue from familyIndex as default
  CommandPool(VkDevice                 device,
              uint32_t                 familyIndex,
              VkCommandPoolCreateFlags flags        = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
              VkQueue                  defaultQueue = VK_NULL_HANDLE)
  {
    init(device, familyIndex, flags, defaultQueue);
  }

  // if defaultQueue is null, uses first queue from familyIndex as default
  void init(VkDevice                 device,
            uint32_t                 familyIndex,
            VkCommandPoolCreateFlags flags        = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            VkQueue                  defaultQueue = VK_NULL_HANDLE);
  void deinit();

#ifdef VULKAN_HPP
  void init(vk::Device device, uint32_t familyIndex, vk::CommandPoolCreateFlags flags, vk::Queue defaultQueue = nullptr)
  {
    init(device, familyIndex, (VkCommandPoolCreateFlags)flags, defaultQueue);
  }
  CommandPool(VkDevice device, uint32_t familyIndex, vk::CommandPoolCreateFlags flags, vk::Queue defaultQueue = nullptr)
  {
    init(device, familyIndex, (VkCommandPoolCreateFlags)flags, defaultQueue);
  }
#endif


  VkCommandBuffer createCommandBuffer(VkCommandBufferLevel      level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                      bool                      begin = true,
                                      VkCommandBufferUsageFlags flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                                      const VkCommandBufferInheritanceInfo* pInheritanceInfo = nullptr);
#ifdef VULKAN_HPP
  // ensure proper cycle is set prior this
  VkCommandBuffer createCommandBuffer(vk::CommandBufferLevel      level,
                                      bool                        begin = true,
                                      vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
                                      const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr)
  {
    return createCommandBuffer((VkCommandBufferLevel)level, begin, (VkCommandBufferUsageFlags)flags,
                               (const VkCommandBufferInheritanceInfo*)pInheritanceInfo);
  }
#endif

  // free cmdbuffers from this pool
  void destroy(size_t count, const VkCommandBuffer* cmds);
  void destroy(const std::vector<VkCommandBuffer>& cmds) { destroy(cmds.size(), cmds.data()); }
  void destroy(VkCommandBuffer cmd) { destroy(1, &cmd); }

  // ends and submits to queue, waits for queue idle and destroys cmds
  void submitAndWait(size_t count, const VkCommandBuffer* cmds, VkQueue queue);
  void submitAndWait(const std::vector<VkCommandBuffer>& cmds, VkQueue queue)
  {
    submitAndWait(cmds.size(), cmds.data(), queue);
  }
  void submitAndWait(VkCommandBuffer cmd, VkQueue queue) { submitAndWait(1, &cmd, queue); }

  // ends and submits to default queue, waits for queue idle and destroys cmds
  void submitAndWait(size_t count, const VkCommandBuffer* cmds) { submitAndWait(count, cmds, m_queue); }
  void submitAndWait(const std::vector<VkCommandBuffer>& cmds) { submitAndWait(cmds.size(), cmds.data(), m_queue); }
  void submitAndWait(VkCommandBuffer cmd) { submitAndWait(1, &cmd, m_queue); }
  VkCommandPool getCommandPool() const { return m_commandPool; }

#ifdef VULKAN_HPP
  void destroy(size_t count, const vk::CommandBuffer* cmds) { destroy(count, (const VkCommandBuffer*)cmds); }
  void destroy(const std::vector<vk::CommandBuffer>& cmds)
  {
    destroy(cmds.size(), (const VkCommandBuffer*)cmds.data());
  }
  void submitAndWait(size_t count, const vk::CommandBuffer* cmds, VkQueue queue)
  {
    submitAndWait(count, (const VkCommandBuffer*)cmds, queue);
  }
  void submitAndWait(const std::vector<vk::CommandBuffer>& cmds, VkQueue queue)
  {
    submitAndWait(cmds.size(), (const VkCommandBuffer*)cmds.data(), queue);
  }
  void submitAndWait(size_t count, const vk::CommandBuffer* cmds)
  {
    submitAndWait(count, (const VkCommandBuffer*)cmds, m_queue);
  }
  void submitAndWait(const std::vector<vk::CommandBuffer>& cmds)
  {
    submitAndWait(cmds.size(), (const VkCommandBuffer*)cmds.data(), m_queue);
  }
#endif

private:
  VkDevice      m_device      = VK_NULL_HANDLE;
  VkQueue       m_queue       = VK_NULL_HANDLE;
  VkCommandPool m_commandPool = VK_NULL_HANDLE;
};


//--------------------------------------------------------------------------------------------------
/**
  # class nvvk::ScopeCommandBuffer

  Provides a single VkCommandBuffer that lives within the scope
  and is directly submitted and deleted when the scope is left.
  Not recommended for efficiency, since it results in a blocking
  operation, but aids sample writing.

  Example:
  ``` C++
  {
    ScopeCommandBuffer cmd(device, queueFamilyIndex, queue);
    ... do stuff
    vkCmdCopyBuffer(cmd, ...);
  }
  ```
*/

class ScopeCommandBuffer : public CommandPool
{
public:
  // if queue is null, uses first queue from familyIndex
  ScopeCommandBuffer(VkDevice device, uint32_t familyIndex, VkQueue queue = VK_NULL_HANDLE)
  {
    CommandPool::init(device, familyIndex, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, queue);
    m_cmd = createCommandBuffer();
  }

  ~ScopeCommandBuffer()
  {
    submitAndWait(m_cmd);
    CommandPool::deinit();
  }

  operator VkCommandBuffer() const { return m_cmd; };
#ifdef VULKAN_HPP
  operator vk::CommandBuffer() const { return (vk::CommandBuffer)m_cmd; };
#endif

private:
  VkCommandBuffer m_cmd;
};

//--------------------------------------------------------------------------------------------------
/**
  # classes **nvvk::Ring...**

  In real-time processing, the CPU typically generates commands 
  in advance to the GPU and send them in batches for execution.

  To avoid having the CPU to wait for the GPU'S completion and let it "race ahead"
  we make use of double, or tripple-buffering techniques, where we cycle through
  a pool of resources every frame. We know that those resources are currently 
  not in use by the GPU and can therefore manipulate them directly.
  
  Especially in Vulkan it is the developer's responsibility to avoid such
  access of resources that are in-flight.

  The "Ring" classes cycle through a pool of resources. The default value
  is set to allow two frames in-flight, assuming one fence is used per-frame.
*/

// typically the driver will not let the CPU race ahead more than two frames of GPU
// during swapchain operations.
static const uint32_t DEFAULT_RING_SIZE = 3;
//--------------------------------------------------------------------------------------------------
/**
  ## class nvvk::RingFences

  Recycles a fixed number of fences, provides information in which cycle
  we are currently at, and prevents accidental access to a cycle in-flight.

  A typical frame would start by "setCycleAndWait", which waits for the
  requested cycle to be available.
*/

class RingFences
{
public:
  RingFences(RingFences const&) = delete;
  RingFences& operator=(RingFences const&) = delete;

  RingFences() {}
  RingFences(VkDevice device, uint32_t ringSize = DEFAULT_RING_SIZE) { init(device, ringSize); }
  ~RingFences() { deinit(); }

  void init(VkDevice device, uint32_t ringSize = DEFAULT_RING_SIZE);
  void deinit();
  void reset()
  {
    VkDevice device   = m_device;
    uint32_t ringSize = m_cycleSize;
    deinit();
    init(device, ringSize);
  }

  // ensures the availability of the passed cycle
  void setCycleAndWait(uint32_t cycle);
  // get current cycle fence
  VkFence getFence();

  // query current cycle index
  uint32_t getCycleIndex() const { return m_cycleIndex; }
  uint32_t getCycleSize() const { return m_cycleSize; }

private:
  struct Entry
  {
    VkFence fence;
    bool    active;
  };

  uint32_t           m_cycleIndex{0};
  uint32_t           m_cycleSize{0};
  std::vector<Entry> m_fences;
  VkDevice           m_device = VK_NULL_HANDLE;
};
//--------------------------------------------------------------------------------------------------
/**
  ## class nvvk::RingCommandPool

  Manages a fixed cycle set of VkCommandBufferPools and
  one-shot command buffers allocated from them.

  The usage of multiple command buffer pools also means we get nice allocation
  behavior (linear allocation from frame start to frame end) without fragmentation.
  If we were using a single command pool over multiple frames, it could fragment easily.

  You must ensure cycle is available manually, typically by keeping in sync
  with ring fences.

  Example:

  ~~~ C++
  {
    frame++;

    // wait until we can use the new cycle 
    // (very rare if we use the fence at then end once per-frame)
    ringFences.setCycleAndWait( frame );

    // update cycle state, allows recycling of old resources
    ringPool.setCycle( frame );

    VkCommandBuffer cmd = ringPool.createCommandBuffer(...);
    ... do stuff / submit etc...

    VkFence fence = ringFences.getFence();
    // use this fence in the submit
    vkQueueSubmit(...fence..);
  }
  ~~~
*/

class RingCommandPool
{
public:
  RingCommandPool(RingCommandPool const&) = delete;
  RingCommandPool& operator=(RingCommandPool const&) = delete;

  RingCommandPool(VkDevice                 device,
                  uint32_t                 queueFamilyIndex,
                  VkCommandPoolCreateFlags flags    = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                  uint32_t                 ringSize = DEFAULT_RING_SIZE)
  {
    init(device, queueFamilyIndex, flags, ringSize);
  }
  RingCommandPool() {}
  ~RingCommandPool() { deinit(); }

  void init(VkDevice                 device,
            uint32_t                 queueFamilyIndex,
            VkCommandPoolCreateFlags flags    = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            uint32_t                 ringSize = DEFAULT_RING_SIZE);
  void deinit();

#ifdef VULKAN_HPP
  void init(vk::Device device, uint32_t queueFamilyIndex, vk::CommandPoolCreateFlags flags, uint32_t ringSize = DEFAULT_RING_SIZE)
  {
    init(device, queueFamilyIndex, (VkCommandPoolCreateFlags)flags, ringSize);
  }
  RingCommandPool(vk::Device device, uint32_t queueFamilyIndex, vk::CommandPoolCreateFlags flags, uint32_t ringSize = DEFAULT_RING_SIZE)
  {
    init(device, queueFamilyIndex, (VkCommandPoolCreateFlags)flags, ringSize);
  }
#endif

  void reset()
  {
    VkDevice                 device           = m_device;
    VkCommandPoolCreateFlags flags            = m_flags;
    uint32_t                 queueFamilyIndex = m_familyIndex;
    uint32_t                 ringSize         = m_cycleSize;
    deinit();
    init(device, queueFamilyIndex, flags, ringSize);
  }

  // call when cycle has changed, prior creating command buffers
  // resets old pools etc.
  void setCycle(uint32_t cycle);

  // ensure proper cycle or frame is set prior these
  VkCommandBuffer createCommandBuffer(VkCommandBufferLevel      level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                      bool                      begin = true,
                                      VkCommandBufferUsageFlags flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                                      const VkCommandBufferInheritanceInfo* pInheritanceInfo = nullptr);

  // pointer is only valid until next create
  const VkCommandBuffer* createCommandBuffers(VkCommandBufferLevel level, uint32_t count);

#ifdef VULKAN_HPP
  // ensure proper cycle is set prior this
  VkCommandBuffer createCommandBuffer(vk::CommandBufferLevel      level,
                                      bool                        begin = true,
                                      vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
                                      const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr)
  {
    return createCommandBuffer((VkCommandBufferLevel)level, begin, (VkCommandBufferUsageFlags)flags,
                               (const VkCommandBufferInheritanceInfo*)pInheritanceInfo);
  }

  // pointer is only valid until next create
  const vk::CommandBuffer* createCommandBuffers(vk::CommandBufferLevel level, uint32_t count)
  {
    return (const vk::CommandBuffer*)createCommandBuffers((VkCommandBufferLevel)level, count);
  }
#endif


protected:
  struct Entry
  {
    VkCommandPool                pool;
    std::vector<VkCommandBuffer> cmds;
  };

  uint32_t                 m_cycleIndex{0};
  uint32_t                 m_cycleSize{0};
  std::vector<Entry>       m_pools;
  VkDevice                 m_device = VK_NULL_HANDLE;
  VkCommandPoolCreateFlags m_flags{0};
  uint32_t                 m_familyIndex{0};
};

//--------------------------------------------------------------------------------------------------
/**
  # class nvvk::BatchSubmission

  Batches the submission arguments of VkSubmitInfo for VkQueueSubmit.

  vkQueueSubmit is a rather costly operation (depending on OS)
  and should be avoided to be done too often (e.g. < 10 per frame). Therefore 
  this utility class allows adding commandbuffers, semaphores etc. and
  submit them later in a batch.

  When using manual locks, it can also be useful to feed commandbuffers
  from different threads and then later kick it off.

  Example

  ~~~ C++
    // within upload logic
    {
      semTransfer = handleUpload(...);
      // for example trigger async upload on transfer queue here
      vkQueueSubmit(... semTransfer ...);

      // tell next frame's batch submission 
      // that its commandbuffers should wait for transfer
      // to be completed
      graphicsSubmission.enqueWait(semTransfer)
    }

    // within present logic
    {
      // for example ensure the next frame waits until proper present semaphore was triggered
      graphicsSubmission.enqueueWait(presentSemaphore);
    }

    // within drawing logic
    {
      // enqueue some graphics work for submission
      graphicsSubmission.enqueue(getSceneCmdBuffer());
      graphicsSubmission.enqueue(getUiCmdBuffer());

      graphicsSubmission.execute(frameFence);
    }
  ~~~
*/

class BatchSubmission
{
private:
  VkQueue                           m_queue = nullptr;
  std::vector<VkSemaphore>          m_waits;
  std::vector<VkPipelineStageFlags> m_waitFlags;
  std::vector<VkSemaphore>          m_signals;
  std::vector<VkCommandBuffer>      m_commands;

public:
  BatchSubmission(BatchSubmission const&) = delete;
  BatchSubmission& operator=(BatchSubmission const&) = delete;

  BatchSubmission() {}
  BatchSubmission(VkQueue queue) { init(queue); }

  uint32_t getCommandBufferCount() const { return uint32_t(m_commands.size()); }
  VkQueue  getQueue() const { return m_queue; }

  // can change queue if nothing is pending
  void init(VkQueue queue);

  void enqueue(uint32_t num, const VkCommandBuffer* cmdbuffers);
  void enqueue(VkCommandBuffer cmdbuffer);
  void enqueueSignal(VkSemaphore sem);
  void enqueueWait(VkSemaphore sem, VkPipelineStageFlags flag);
#ifdef VULKAN_HPP
  void enqueue(uint32_t num, const vk::CommandBuffer* cmdbuffers) { enqueue(num, (const VkCommandBuffer*)cmdbuffers); }
  void enqueueWait(vk::Semaphore sem, vk::PipelineStageFlags flag) { enqueueWait(sem, (VkPipelineStageFlags)flag); }
#endif

  // submits the work and resets internal state
  VkResult execute(VkFence fence = nullptr, uint32_t deviceMask = 0);

  void waitIdle() const;
};

//////////////////////////////////////////////////////////////////////////
/**
  # class nvvk::FencedCommandPools

  This container class contains the typical utilities to handle
  command submission. It contains RingFences, RingCommandPool and BatchSubmission
  with a convenient interface.

*/
class FencedCommandPools : private RingFences, private RingCommandPool, private BatchSubmission
{
public:
  FencedCommandPools(FencedCommandPools const&) = delete;
  FencedCommandPools& operator=(FencedCommandPools const&) = delete;

  FencedCommandPools() {}
  ~FencedCommandPools() { deinit(); }

  FencedCommandPools(VkDevice                 device,
                     VkQueue                  queue,
                     uint32_t                 queueFamilyIndex,
                     VkCommandPoolCreateFlags flags    = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                     uint32_t                 ringSize = DEFAULT_RING_SIZE)
  {
    init(device, queue, queueFamilyIndex, flags, ringSize);
  }

  void init(VkDevice                 device,
            VkQueue                  queue,
            uint32_t                 queueFamilyIndex,
            VkCommandPoolCreateFlags flags    = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            uint32_t                 ringSize = DEFAULT_RING_SIZE)
  {
    RingFences::init(device, ringSize);
    RingCommandPool::init(device, queueFamilyIndex, flags, ringSize);
    BatchSubmission::init(queue);
  }
#ifdef VULKAN_HPP
  FencedCommandPools(vk::Device device, vk::Queue queue, uint32_t queueFamilyIndex, vk::CommandPoolCreateFlags flags, uint32_t ringSize = DEFAULT_RING_SIZE)
  {
    init(device, queue, queueFamilyIndex, (VkCommandPoolCreateFlags)flags, ringSize);
  }
  void init(vk::Device device, vk::Queue queue, uint32_t queueFamilyIndex, vk::CommandPoolCreateFlags flags, uint32_t ringSize = DEFAULT_RING_SIZE)
  {
    init(device, queue, queueFamilyIndex, (VkCommandPoolCreateFlags)flags, ringSize);
  }
#endif

  void deinit()
  {
    RingFences::deinit();
    RingCommandPool::deinit();
    //BatchSubmission::deinit();
  }

  void reset()
  {
    waitIdle();
    RingFences::reset();
    RingCommandPool::reset();
  }

  void enqueue(uint32_t num, const VkCommandBuffer* cmdbuffers) { BatchSubmission::enqueue(num, cmdbuffers); }
  void enqueue(VkCommandBuffer cmdbuffer) { BatchSubmission::enqueue(cmdbuffer); }
  void enqueueSignal(VkSemaphore sem) { BatchSubmission::enqueueSignal(sem); }
  void enqueueWait(VkSemaphore sem, VkPipelineStageFlags flag) { BatchSubmission::enqueueWait(sem, flag); }
#ifdef VULKAN_HPP
  void enqueue(uint32_t num, const vk::CommandBuffer* cmdbuffers)
  {
    BatchSubmission::enqueue(num, (const VkCommandBuffer*)cmdbuffers);
  }
  void enqueueWait(vk::Semaphore sem, vk::PipelineStageFlags flag)
  {
    BatchSubmission::enqueueWait(sem, (VkPipelineStageFlags)flag);
  }
#endif
  VkResult execute(uint32_t deviceMask = 0) { return BatchSubmission::execute(getFence(), deviceMask); }

  void waitIdle() const { BatchSubmission::waitIdle(); }

  void setCycleAndWait(uint32_t cycle)
  {
    RingFences::setCycleAndWait(cycle);
    RingCommandPool::setCycle(cycle);
  }

  // ensure proper cycle is set prior this
  VkCommandBuffer createCommandBuffer(VkCommandBufferLevel      level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                      bool                      begin = true,
                                      VkCommandBufferUsageFlags flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                                      const VkCommandBufferInheritanceInfo* pInheritanceInfo = nullptr)
  {
    return RingCommandPool::createCommandBuffer(level, begin, flags, pInheritanceInfo);
  }

  // pointer is only valid until next create
  const VkCommandBuffer* createCommandBuffers(VkCommandBufferLevel level, uint32_t count)
  {
    return RingCommandPool::createCommandBuffers(level, count);
  }

#ifdef VULKAN_HPP
  // ensure proper cycle is set prior this
  VkCommandBuffer createCommandBuffer(vk::CommandBufferLevel      level,
                                      bool                        begin = true,
                                      vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
                                      const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr)
  {
    return RingCommandPool::createCommandBuffer((VkCommandBufferLevel)level, begin, (VkCommandBufferUsageFlags)flags,
                                                (const VkCommandBufferInheritanceInfo*)pInheritanceInfo);
  }

  // pointer is only valid until next create
  const vk::CommandBuffer* createCommandBuffers(vk::CommandBufferLevel level, uint32_t count)
  {
    return (const vk::CommandBuffer*)createCommandBuffers((VkCommandBufferLevel)level, count);
  }
#endif


  struct ScopedCmd
  {
    FencedCommandPools* pCmdPools;
    VkCommandBuffer     cmd;

    ScopedCmd(FencedCommandPools& cp)
    {
      pCmdPools = &cp;
      cmd       = cp.createCommandBuffer();
    }
    ~ScopedCmd()
    {
      vkEndCommandBuffer(cmd);
      pCmdPools->enqueue(cmd);
      pCmdPools->execute();
      pCmdPools->waitIdle();
    }

    operator VkCommandBuffer() { return cmd; }
#ifdef VULKAN_HPP
    operator vk::CommandBuffer() const { return (vk::CommandBuffer)cmd; };
#endif
  };
};


}  // namespace nvvk
