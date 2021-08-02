# Vulkan Api Helpers

Non-exhaustive list of utilities provided in the `nvvk` directory

If you intend to use the Vulkan C++ api, include <vulkan/vulkan.hpp> before including the helper files.

Table of Contents:
- [allocator_dedicated_vk.hpp:](#allocator_dedicated_vkhpp)
  - class [nvvk::AllocatorDedicated](#class-nvvkallocatordedicated)
  - class [nvvk::AllocatorVkExport](#class-nvvkallocatorvkexport)
- [allocator_dma_vk.hpp:](#allocator_dma_vkhpp)
  - class [nvvk::AllocatorDma](#class-nvvkallocatordma)
- [allocator_dma_vkgl.hpp:](#allocator_dma_vkglhpp)
  - class [nvkk::AllocatorDmaGL](#class-nvkkallocatordmagl)
- [allocator_vk.hpp:](#allocator_vkhpp)
- [allocator_vma_vk.hpp:](#allocator_vma_vkhpp)
  - class [nvvk::StagingMemoryManagerVma](#class-nvvkstagingmemorymanagervma)
  - class [nvvk::AllocatorVma](#class-nvvkallocatorvma)
- [appbase_vkpp.hpp:](#appbase_vkpphpp)
  - class [nvvk::AppBase](#class-nvvkappbase)
- [appwindowprofiler_vk.hpp:](#appwindowprofiler_vkhpp)
  - class [nvvk::AppWindowProfilerVK](#class-nvvkappwindowprofilervk)
- [buffers_vk.hpp:](#buffers_vkhpp)
- [commands_vk.hpp:](#commands_vkhpp)
  - class [nvvk::CommandPool](#class-nvvkcommandpool)
  - class [nvvk::ScopeCommandBuffer](#class-nvvkscopecommandbuffer)
  - class [nvvk::RingFences](#class-nvvkringfences)
  - class [nvvk::RingCommandPool](#class-nvvkringcommandpool)
  - class [nvvk::BatchSubmission](#class-nvvkbatchsubmission)
  - class [nvvk::FencedCommandPools](#class-nvvkfencedcommandpools)
- [context_vk.hpp:](#context_vkhpp)
  - class [nvvk::Context](#class-nvvkcontext)
- [descriptorsets_vk.hpp:](#descriptorsets_vkhpp)
  - class [nvvk::DescriptorSetBindings](#class-nvvkdescriptorsetbindings)
  - class [nvvk::DescriptorSetContainer](#class-nvvkdescriptorsetcontainer)
  - class [nvvk::TDescriptorSetContainer<SETS,PIPES=1>](#class-nvvktdescriptorsetcontainersetspipes1)
- [error_vk.hpp:](#error_vkhpp)
- [extensions_vk.hpp:](#extensions_vkhpp)
- [gizmos_vk.hpp:](#gizmos_vkhpp)
  - class [nvvk::Axis](#class-nvvkaxis)
- [images_vk.hpp:](#images_vkhpp)
- [memorymanagement_vk.hpp:](#memorymanagement_vkhpp)
  - class [nvvk::DeviceMemoryAllocator](#class-nvvkdevicememoryallocator)
  - class [nvvk::StagingMemoryManager](#class-nvvkstagingmemorymanager)
  - class [nvvk::StagingMemoryManagerDma](#class-nvvkstagingmemorymanagerdma)
- [memorymanagement_vkgl.hpp:](#memorymanagement_vkglhpp)
  - class [nvvk::DeviceMemoryAllocatorGL](#class-nvvkdevicememoryallocatorgl)
- [pipeline_vk.hpp:](#pipeline_vkhpp)
  - class [nvvk::GraphicsPipelineState](#class-nvvkgraphicspipelinestate)
  - class [nvvk::GraphicsPipelineGenerator](#class-nvvkgraphicspipelinegenerator)
  - class [nvvk::GraphicsPipelineGeneratorCombined](#class-nvvkgraphicspipelinegeneratorcombined)
- [profiler_vk.hpp:](#profiler_vkhpp)
  - class [nvvk::ProfilerVK](#class-nvvkprofilervk)
- [raytraceKHR_vk.hpp:](#raytracekhr_vkhpp)
  - class [nvvk::RaytracingBuilderKHR](#class-nvvkraytracingbuilderkhr)
- [raytraceNV_vk.hpp:](#raytracenv_vkhpp)
  - class [nvvk::RaytracingBuilderNV](#class-nvvkraytracingbuildernv)
- [renderpasses_vk.hpp:](#renderpasses_vkhpp)
- [samplers_vk.hpp:](#samplers_vkhpp)
- [shadermodulemanager_vk.hpp:](#shadermodulemanager_vkhpp)
  - class [nvvk::ShaderModuleManager](#class-nvvkshadermodulemanager)
- [shaders_vk.hpp:](#shaders_vkhpp)
- [structs_vk.hpp:](#structs_vkhpp)
- [swapchain_vk.hpp:](#swapchain_vkhpp)
  - class [nvvk::SwapChain](#class-nvvkswapchain)

_____

## allocator_dedicated_vk.hpp

### class **nvvk::AllocatorDedicated**

This is the allocator specialization using only Vulkan where there will be one memory
allocation for each buffer or image.
See more details in description of [nvvk::AllocatorDma](#class-nvvkallocatordma) for the
general use of allocator classes.

> Note: this should be used only when really needed, as it is making one allocation per buffer,
>       which is not efficient. 

### Initialization

~~~~~~ C++
nvvk::AllocatorVk m_alloc;
m_alloc.init(device, physicalDevice);
~~~~~~


### class **nvvk::AllocatorVkExport**

This version of the **AllocatorDedicated** will export all memory allocations, which can then be used by CUDA or OpenGL.

## allocator_dma_vk.hpp

### class **nvvk::AllocatorDma**

The goal of the `AllocatorABC` classes is to have common work-flow
even if the underlying allocator classes are different.
This should make it relatively easy to switch between different
allocator implementations (more or less only changing typedefs).

The `BufferABC`, `ImageABC` etc. structs always contain the native
resource handle as well as the allocator system's handle.

This utility class wraps the usage of **nvvk::DeviceMemoryAllocator**
as well as **nvvk::StagingMemoryManagerDma** to have a simpler interface
for handling resources with content uploads.

> Note: These classes are foremost to showcase principle components that
> a Vulkan engine would most likely have.
> They are geared towards ease of use in this sample framework, and 
> not optimized nor meant for production code.

~~~ C++
DeviceMemoryAllocator memAllocator;
AllocatorDma          allocator;

memAllocator.init(device, physicalDevice);
allocator.init(device, physicalDevice, &memAllocator);

...

VkCommandBuffer cmd = ... transfer queue command buffer

// creates new resources and 
// implicitly triggers staging transfer copy operations into cmd
BufferDma vbo = allocator.createBuffer(cmd, vboSize, vboData, vboUsage);
BufferDma ibo = allocator.createBuffer(cmd, iboSize, iboData, iboUsage);

// use functions from staging memory manager
// here we associate the temporary staging resources with a fence
allocator.finalizeStaging( fence );

// submit cmd buffer with staging copy operations
vkQueueSubmit(... cmd ... fence ...)

...

// if you do async uploads you would
// trigger garbage collection somewhere per frame
allocator.releaseStaging();

~~~

## allocator_dma_vkgl.hpp

This file contains helpers for resource interoperability between OpenGL and Vulkan.
they only exist if the shared_sources project is compiled with Vulkan AND OpenGL support.

> WARNING: untested code


### class **nvkk::AllocatorDmaGL**

This utility has the same operations like nvvk::AllocatorDMA (see for more help), but
targets interop between OpenGL and Vulkan.
It uses **nvkk::DeviceMemoryAllocatorGL** to provide **BufferDmaGL** and **ImageDmaGL** utility classes that wrap an **nvvk::AllocationID**
as well as the native Vulkan and OpenGL resource objects.

## allocator_vk.hpp

If desired some samples may want to easily switch between different
allocator classes.

This file will include the appropriate nvvk::Allocator? class depending 
on one of three possible defines that must be set prior to including:

- **NVVK_ALLOC_DEDICATED** : **nvvk::AllocatorDedicated** is a naive implementation that allocates one `VkDeviceMemory` per resource (VkBuffer/VkImage).
  This is not a recommended practice, but useful for basic testing and low complexity in the samples.
- **NVVK_ALLOC_DMA** : **nvvk::AllocatorDma** uses the **nvvk::DeviceMemoryAllocator** class to allocate `VkDeviceMemory` in chunks, which
  lowers the amount of overall allocations. This practice is recommended for Vulkan resource management
  in general. There are limits on the amount of allocations that can be fairly low, and there is also a performance
  penalty going through the OS to make such allocations.  
  Furthermore this allocator class uses **nvvk::StagingMemoryManagerDma**
  which uses multiple chunks of `VkBuffers` for staging memory. The motivation is the same as for memory, we reduce
  the amount of Vulkan object creations, by sub-allocating temporary space from a `VkBuffer` that is mapped to
  host to aid the upload process.
- **NVVK_ALLOC_VMA** : **nvvk::AllocatorVma** makes use of the [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
  and works similar to the `DMA` variant above, it also makes use of **nvvk::StagingMemoryManagerVma**.


It also provides structs such as nvvk::Image **nvvk::Buffer** etc. that map to the appropriate
structs, e.g. **nvvk::ImageDma**.

See more details in description of [nvvk::AllocatorDma](#class-nvvkallocatordma).

## allocator_vma_vk.hpp

### class **nvvk::StagingMemoryManagerVma**

This utility class wraps the usage of [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
to allocate the memory for **nvvk::StagingMemoryManager**


### class **nvvk::AllocatorVma**

This utility class wraps the usage of [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
as well as **nvvk::StagingMemoryManager** to have a simpler interface
for handling resources with content uploads.

See more details in description of [nvvk::AllocatorDma](#class-nvvkallocatordma).

## appbase_vkpp.hpp

### class **nvvk::AppBase**

The framework comes with a few `App???` classes, these can serve as base classes for various samples.
They might differ a bit in setup and functionality, but in principle aid the setup of context and window,
as well as some common event processing.

The **nvvk::AppBase** serves as the base class for many ray tracing examples and makes use of the Vulkan C++ API (`vulkan.hpp`).
It does the basics for Vulkan, by holding a reference to the instance and device, but also comes with optional default setups 
for the render passes and the swapchain.

#### Usage

An example will derive from this class:

~~~~ C++
class MyExample : public AppBase 
{
};
~~~~

#### Setup

In the `main()` of the example, after creating the Vulkan instance and device, call `setup()`.
This will hold the `VkInstance`, `VkDevice`, `VkPhysicalDevice` and create the `VkQueue`, `VkPool`, plus it
will initialize all Vulkan extensions for the C++ API (vulkan.hpp).

Prior to calling setup, if you are using the `nvvk::Context` class to create and initalize Vulkan instances,
you may want to create a `VkSurfaceKHR` from the window (glfw for example) and call `setGCTQueueWithPresent()`.
This will make sure the **Queue** indices are adapted.

Creating the surface, will actually create the swapchain for displaying. Arguments are
width and height, color and depth format, and vsync on/off. Defaults will create the best format.

Before creating the framebuffers to display the results, a depth buffer has to be create which will
be used by all framebuffers. Then we can create the framebuffer.

**Note**: the imageView(s) are part of the swapchain. 

There is also a 'default renderpass', which is a color/depth, clear both buffers.

If the application is using Dear ImGui, there are convenient functions for initializing it and
setting the callbacks (glfw). The first one to call is `initGUI(0)`, where the argument is the subpass
where it will be use. Default is 0, but if the application creates a renderpass with multi-sampling and
resolves in the second subpass, this makes it possible.

Then call `setupGlfwCallbacks(window)` to have all the window callback: key, mouse, window resizing.
By default **AppBase** will handle resizing of the window. It will recreate the images and framebuffers,
but a sample may need to overload that function, or to be aware of that change, therefore can overload
`onResize(width, height)`.

Last setup for ImGui is to call `ImGui_ImplGlfw_InitForVulkan(window, true)`, where true is for the 
callbacks for Imgui.

**Note**: All the methods are virtual and can be overloaded if they are not doing the typical setup. 

~~~~ C++
MyExample example;

const vk::SurfaceKHR surface = example.getVkSurface(vkctx.m_instance, window);
vkctx.setGCTQueueWithPresent(surface);

example.setup(vkctx.m_instance, vkctx.m_device, vkctx.m_physicalDevice, vkctx.m_queueGCT.familyIndex);
example.createSurface(surface, SAMPLE_SIZE_WIDTH, SAMPLE_SIZE_HEIGHT);
example.createDepthBuffer();
example.createFrameBuffers();
example.createRenderPass();
example.initGUI(0);
example.setupGlfwCallbacks(window);

ImGui_ImplGlfw_InitForVulkan(window, true);
~~~~

#### Drawing loop

The drawing loop in the main() is the typicall loop you will find in glfw examples. Note that
**AppBase** has a convenient function to tell if the window is minimize, therefore not doing any 
work and contain a sleep(), so the CPU is not going crazy. 


~~~~ C++
// Window system loop
while(!glfwWindowShouldClose(window))
{
  glfwPollEvents();
  if(example.isMinimized())
    continue;

  example.display();  // infinitely drawing
}
~~~~

#### Display

A typical display() function will need the following: 

* Acquiring the next image: `prepareFrame()`
* Get the command buffer for the frame. There are n command buffers equal to the number of in-flight frames.
* Clearing values
* Start rendering pass
* Drawing
* End rendering
* Submitting frame to display

~~~~ C++
void MyExample::display()
{
  // Acquire 
  prepareFrame();

  // Command buffer for current frame
  const vk::CommandBuffer& cmdBuff = m_commandBuffers[getCurFrame()];
  cmdBuff.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  // Clearing values
  vk::ClearValue clearValues[2];
  clearValues[0].setColor(std::array<float, 4>({0.1f, 0.1f, 0.4f, 0.f}));
  clearValues[1].setDepthStencil({1.0f, 0});

  // Begin rendering
  vk::RenderPassBeginInfo renderPassBeginInfo{m_renderPass, m_framebuffers[getCurFrame()], {{}, m_size}, 2, clearValues};
  cmdBuff.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
  
  // .. draw scene ...

  // Draw UI
  ImGui::RenderDrawDataVK(cmdBuff, ImGui::GetDrawData());

  // End rendering
  cmdBuff.endRenderPass();

  // End of the frame and present the one which is ready
  cmdBuff.end();
  submitFrame();
}
~~~~~

#### Closing

Finally, all resources can be destroyed by calling `destroy()` at the end of main().

~~~~ C++
example.destroy();
~~~~

## appwindowprofiler_vk.hpp

### class **nvvk::AppWindowProfilerVK**

**AppWindowProfilerVK** derives from **nvh::AppWindowProfiler**
and overrides the context and swapbuffer functions.
The nvh class itself provides several utilities and 
command line options to run automated benchmarks etc.

To influence the vulkan instance/device creation modify 
`m_contextInfo` prior running AppWindowProfiler::run,
which triggers instance, device, window, swapchain creation etc.

The class comes with a **nvvk::ProfilerVK** instance that references the 
AppWindowProfiler::m_profiler's data.

## buffers_vk.hpp

The utilities in this file provide a more direct approach, we encourage to use
higher-level mechanisms also provided in the allocator / memorymanagement classes.

### functions in nvvk

- **makeBufferCreateInfo** : wraps setup of `VkBufferCreateInfo` (implicitly sets VK_BUFFER_USAGE_TRANSFER_DST_BIT)
- **makeBufferViewCreateInfo** : wraps setup of `VkBufferViewCreateInfo`
- **createBuffer** : wraps `vkCreateBuffer`
- **createBufferView** : wraps `vkCreateBufferView`

~~~ C++
VkBufferCreateInfo bufferCreate = makeBufferCreateInfo (size, VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT);
VkBuffer buffer                 = createBuffer(device, bufferCreate);
VkBufferView bufferView         = createBufferView(device, makeBufferViewCreateInfo(buffer, VK_FORMAT_R8G8B8A8_UNORM, size));
~~~

## commands_vk.hpp

### functions in nvvk

- **makeAccessMaskPipelineStageFlags** : depending on accessMask returns appropriate `VkPipelineStageFlagBits`
- **cmdBegin** : wraps `vkBeginCommandBuffer` with `VkCommandBufferUsageFlags` and implicitly handles `VkCommandBufferBeginInfo` setup
- **makeSubmitInfo** : `VkSubmitInfo` struct setup using provided arrays of signals and commandbuffers, leaving rest zeroed


### class **nvvk::CommandPool**

**CommandPool** stores a single `VkCommandPool` and provides utility functions
to create `VkCommandBuffers` from it.

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


### class **nvvk::ScopeCommandBuffer**

Provides a single `VkCommandBuffer` that lives within the scope
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


### classes **nvvk::Ring...**

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


#### class **nvvk::RingFences**

Recycles a fixed number of fences, provides information in which cycle
we are currently at, and prevents accidental access to a cycle in-flight.

A typical frame would start by "setCycleAndWait", which waits for the
requested cycle to be available.


#### class **nvvk::RingCommandPool**

Manages a fixed cycle set of `VkCommandBufferPools` and
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


### class **nvvk::BatchSubmission**

Batches the submission arguments of `VkSubmitInfo` for `VkQueueSubmit`.

`vkQueueSubmit` is a rather costly operation (depending on OS)
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


### class **nvvk::FencedCommandPools**

This container class contains the typical utilities to handle
command submission. It contains **RingFences**, **RingCommandPool** and **BatchSubmission**
with a convenient interface.

## context_vk.hpp

To run a Vulkan application, you need to create the Vulkan instance and device.
This is done using the `nvvk::Context`, which wraps the creation of `VkInstance`
and `VkDevice`.

First, any application needs to specify how instance and device should be created:
Version, layers, instance and device extensions influence the features available.
This is done through a temporary and intermediate class that will allow you to gather
all the required conditions for the device creation.


### struct **ContextCreateInfo**

This structure allows the application to specify a set of features
that are expected for the creation of
- `VkInstance`
- `VkDevice`

It is consumed by the `nvvk::Context::init` function.

Example on how to populate information in it : 

~~~~ C++
nvvk::ContextCreateInfo ctxInfo;
ctxInfo.setVersion(1, 1);
ctxInfo.addInstanceExtension(VK_KHR_SURFACE_EXTENSION_NAME, false);
ctxInfo.addInstanceExtension(VK_KHR_WIN32_SURFACE_EXTENSION_NAME, false);
ctxInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME, false);
~~~~

then you are ready to create initialize `nvvk::Context`

> Note: In debug builds, the extension `VK_EXT_DEBUG_UTILS_EXTENSION_NAME` and the layer `VK_LAYER_KHRONOS_validation` are added to help finding issues early.


### class **nvvk::Context**

**Context** class helps creating the Vulkan instance and to choose the logical device for the mandatory extensions. First is to fill the `ContextCreateInfo` structure, then call:

~~~ C++
// Creating the Vulkan instance and device
nvvk::ContextCreateInfo ctxInfo;
... see above ...

nvvk::Context vkctx;
vkctx.init(ctxInfo);

// after init the ctxInfo is no longer needed
~~~ 

At this point, the class will have created the `VkInstance` and `VkDevice` according to the information passed. It will also keeps track or have query the information of:
 
* Physical Device information that you can later query : `PhysicalDeviceInfo` in which lots of `VkPhysicalDevice...` are stored
* `VkInstance` : the one instance being used for the programm
* `VkPhysicalDevice` : physical device(s) used for the logical device creation. In case of more than one physical device, we have a std::vector for this purpose...
* `VkDevice` : the logical device instanciated
* `VkQueue` : we will enumerate all the available queues and make them available in `nvvk::Context`. Some queues are specialized, while other are for general purpose (most of the time, only one can handle everything, while other queues are more specialized). We decided to make them all available in some explicit way :
 * `Queue m_queueGCT` : Graphics/Compute/Transfer **Queue** + family index
 * `Queue m_queueT` : async Transfer **Queue** + family index
 * `Queue m_queueC` : Compute **Queue** + family index
* maintains what extensions are finally available
* implicitly hooks up the debug callback

#### Choosing the device
When there are multiple devices, the `init` method is choosing the first compatible device available, but it is also possible the choose another one.
~~~ C++
vkctx.initInstance(deviceInfo); 
// Find all compatible devices
auto compatibleDevices = vkctx.getCompatibleDevices(deviceInfo);
assert(!compatibleDevices.empty());

// Use first compatible device
vkctx.initDevice(compatibleDevices[0], deviceInfo);
~~~

#### Multi-GPU

When multiple graphic cards should be used as a single device, the `ContextCreateInfo::useDeviceGroups` need to be set to `true`.
The above methods will transparently create the `VkDevice` using `VkDeviceGroupDeviceCreateInfo`.
Especially in the context of NVLink connected cards this is useful.

## descriptorsets_vk.hpp

### functions in nvvk

- **createDescriptorPool** : wrappers for `vkCreateDescriptorPool`
- **allocateDescriptorSet** : allocates a single `VkDescriptorSet`
- **allocateDescriptorSets** : allocates multiple VkDescriptorSets


### class **nvvk::DescriptorSetBindings**

Helper class that keeps a vector of `VkDescriptorSetLayoutBinding` for a single
`VkDescriptorSetLayout`. Provides helper functions to create `VkDescriptorSetLayout`
as well as `VkDescriptorPool` based on this information, as well as utilities
to fill the `VkWriteDescriptorSet` structure with binding information stored
within the class.

The class comes with the convenience functionality that when you make a
`VkWriteDescriptorSet` you provide the binding slot, rather than the
index of the binding's storage within this class. This results in a small
linear search, but makes it easy to change the content/order of bindings
at creation time.

Example :
~~~C++
DescriptorSetBindings binds;

binds.addBinding( VIEW_BINDING, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT);
binds.addBinding(XFORM_BINDING, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT);

VkDescriptorSetLayout layout = binds.createLayout(device);

#if SINGLE_LAYOUT_POOL
  // let's create a pool with 2 sets
  VkDescriptorPool      pool   = binds.createPool(device, 2);
#else
  // if you want to combine multiple layouts into a common pool
  std::vector<VkDescriptorPoolSize> poolSizes;
  bindsA.addRequiredPoolSizes(poolSizes, numSetsA);
  bindsB.addRequiredPoolSizes(poolSizes, numSetsB);
  VkDescriptorPool      pool   = nvvk::createDescriptorPool(device, poolSizes,
                                                            numSetsA + numSetsB);
#endif

// fill them
std::vector<VkWriteDescriptorSet> updates;

updates.push_back(binds.makeWrite(0, VIEW_BINDING, &view0BufferInfo));
updates.push_back(binds.makeWrite(1, VIEW_BINDING, &view1BufferInfo));
updates.push_back(binds.makeWrite(0, XFORM_BINDING, &xform0BufferInfo));
updates.push_back(binds.makeWrite(1, XFORM_BINDING, &xform1BufferInfo));

vkUpdateDescriptorSets(device, updates.size(), updates.data(), 0, nullptr);
~~~


### class **nvvk::DescriptorSetContainer**

Container class that stores allocated DescriptorSets
as well as reflection, layout and pool for a single
`VkDescripterSetLayout`.

Example:
~~~ C++
container.init(device, allocator);

// setup dset layouts
container.addBinding(0, UBO...)
container.addBinding(1, SSBO...)
container.initLayout();

// allocate descriptorsets
container.initPool(17);

// update descriptorsets
writeUpdates.push_back( container.makeWrite(0, 0, &..) );
writeUpdates.push_back( container.makeWrite(0, 1, &..) );
writeUpdates.push_back( container.makeWrite(1, 0, &..) );
writeUpdates.push_back( container.makeWrite(1, 1, &..) );
writeUpdates.push_back( container.makeWrite(2, 0, &..) );
writeUpdates.push_back( container.makeWrite(2, 1, &..) );
...

// at render time

vkCmdBindDescriptorSets(cmd, GRAPHICS, pipeLayout, 1, 1, container.at(7).getSets());
~~~


### class **nvvk::TDescriptorSetContainer**<SETS,PIPES=1>

Templated version of **DescriptorSetContainer** :

- SETS  - many **DescriptorSetContainers**
- PIPES - many `VkPipelineLayouts`

The pipeline layouts are stored separately, the class does
not use the pipeline layouts of the embedded **DescriptorSetContainers**.

Example :

~~~ C++
Usage, e.g.SETS = 2, PIPES = 2

container.init(device, allocator);

// setup dset layouts
container.at(0).addBinding(0, UBO...)
container.at(0).addBinding(1, SSBO...)
container.at(0).initLayout();
container.at(1).addBinding(0, COMBINED_SAMPLER...)
container.at(1).initLayout();

// pipe 0 uses set 0 alone
container.initPipeLayout(0, 1);
// pipe 1 uses sets 0, 1
container.initPipeLayout(1, 2);

// allocate descriptorsets
container.at(0).initPool(1);
container.at(1).initPool(16);

// update descriptorsets

writeUpdates.push_back(container.at(0).makeWrite(0, 0, &..));
writeUpdates.push_back(container.at(0).makeWrite(0, 1, &..));
writeUpdates.push_back(container.at(1).makeWrite(0, 0, &..));
writeUpdates.push_back(container.at(1).makeWrite(1, 0, &..));
writeUpdates.push_back(container.at(1).makeWrite(2, 0, &..));
...

// at render time

vkCmdBindDescriptorSets(cmd, GRAPHICS, container.getPipeLayout(0), 0, 1, container.at(0).getSets());
..
vkCmdBindDescriptorSets(cmd, GRAPHICS, container.getPipeLayout(1), 1, 1, container.at(1).getSets(7));
~~~

## error_vk.hpp

### function nvvk::checkResult
Returns true on critical error result, logs errors.
Use `NVVK_CHECK(result)` to automatically log filename/linenumber.

## extensions_vk.hpp

### Vulkan Extension Loader

The extensions_vk files takes care of loading and providing the symbols of
Vulkan C Api extensions.
It is generated by `extensions_vk.lua` which contains a whitelist of
extensions to be made available.

The framework triggers this implicitly in the `nvvk::Context` class.

If you want to use it in your own code, see the instructions in the 
lua file how to generate it.

~~~ c++
// loads all known extensions
load_VK_EXTENSION_SUBSET(instance, vkGetInstanceProcAddr, device, vkGetDeviceProcAddr);

// load individual extension
load_VK_KHR_push_descriptor(instance, vkGetInstanceProcAddr, device, vkGetDeviceProcAddr);
~~~

## gizmos_vk.hpp

### class nvvk::Axis

Display an Axis representing the orientation of the camera in the bottom left corner of the window.
- Initialize the Axis using `init()`
- Add `display()` in a inline rendering pass, one of the lass command

Example:  
~~~~~~ C++
m_axis.display(cmdBuf, CameraManip.getMatrix(), windowSize);
~~~~~~

## images_vk.hpp

### functions in nvvk

- **makeImageMemoryBarrier** : returns `VkImageMemoryBarrier` for an image based on provided layouts and access flags.
- **mipLevels** : return number of mips for 2d/3d extent

- **accessFlagsForImageLayout** : helps resource transtions
- **pipelineStageForLayout** : helps resource transitions
- **cmdBarrierImageLayout** : inserts barrier for image transition

- **cmdGenerateMipmaps** : basic mipmap creation for images (meant for one-shot operations)

- **makeImage2DCreateInfo** : aids 2d image creation
- **makeImage3DCreateInfo** : aids 3d descriptor set updating
- **makeImageCubeCreateInfo** : aids cube descriptor set updating
- **makeImageViewCreateInfo** : aids common image view creation, derives info from `VkImageCreateInfo`
- **makeImage2DViewCreateInfo** : aids 2d image view creation

## memorymanagement_vk.hpp

This framework assumes that memory heaps exists that support:

- VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
  for uploading data to the device
- VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & VK_MEMORY_PROPERTY_HOST_CACHED_BIT
  for downloading data from the device

This is typical on all major desktop platforms and vendors.
See http://vulkan.gpuinfo.org for information of various devices and platforms.

### functions in nvvk

* getMemoryInfo : fills the `VkMemoryAllocateInfo` based on device's memory properties and memory requirements and property flags. Returns `true` on success.


### class **nvvk::DeviceMemoryAllocator**

**DeviceMemoryAllocator** allocates and manages device memory in fixed-size memory blocks.

It sub-allocates from the blocks, and can re-use memory if it finds empty
regions. Because of the fixed-block usage, you can directly create resources
and don't need a phase to compute the allocation sizes first.

It will create compatible chunks according to the memory requirements and
usage flags. Therefore you can easily create mappable host allocations
and delete them after usage, without inferring device-side allocations.

We return `AllocationID` rather than the allocation details directly, which
you can query separately.

Several utility functions are provided to handle the binding of memory
directly with the resource creation of buffers, images and acceleration
structures. This utilities also make implicit use of Vulkan's dedicated
allocation mechanism.

> **WARNING** : The memory manager serves as proof of concept for some key concepts
> however it is not meant for production use and it currently lacks de-fragmentation logic
> as well. You may want to look at [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
> for a more production-focused solution.

You can derive from this calls and overload the 

Example :
~~~ C++
nvvk::DeviceMemoryAllocator memAllocator;

memAllocator.init(device, physicalDevice);

// low-level
aid = memAllocator.alloc(memRequirements,...);
...
memAllocator.free(aid);

// utility wrapper
buffer = memAllocator.createBuffer(bufferSize, bufferUsage, bufferAid);
...
memAllocator.free(bufferAid);


// It is also possible to not track individual resources
// and free everything in one go. However, this is
// not recommended for general purpose use.

bufferA = memAllocator.createBuffer(sizeA, usageA);
bufferB = memAllocator.createBuffer(sizeB, usageB);
...
memAllocator.freeAll();

~~~


### class **nvvk::StagingMemoryManager**

**StagingMemoryManager** class is a utility that manages host visible
buffers and their allocations in an opaque fashion to assist
asynchronous transfers between device and host.

The collection of the transfer resources is represented by nvvk::StagingID.

The necessary buffer space is sub-allocated and recycled in blocks internally.
This way we avoid creating lots of small `VkBuffers` and avoid calling the Vulkan
API at all. While Vulkan is more efficient than previous APIs, creating lots
of objects for it, is still not good for overall performance. It will result 
into more cache misses and use more system memory over all.

The default implementation will create one dedicated memory allocation per block.
You can derive from this class and overload the virtual functions,
if you want to use a different allocation system.

- **allocBlockMemory**
- **freeBlockMemory**

> **WARNING:**
> - cannot manage a copy > 4 GB

Usage:
- Enqueue transfers into your `VkCommandBuffer` and then finalize the copy operations.
- Associate the copy operations with a `VkFence`
- The release of the resources allows to safely recycle the buffer space for future transfers.

> We use fences as a way to garbage collect here, however a more robust solution
> may be implementing some sort of ticketing/timeline system.
> If a fence is recycled, then this class may not be aware that the fence represents a different
> submission, likewise if the fence is deleted elsewhere problems can occur.

Example :

~~~ C++
StagingMemoryManager  staging;
staging.init(device, physicalDevice);


// Enqueue copy operations of data to target buffer.
// This internally manages the required staging resources
staging.cmdToBuffer(cmd, targetBufer, 0, targetSize, targetData);

// you can also get access to a temporary mapped pointer and fill
// the staging buffer directly
vertices = staging.cmdToBufferT<Vertex>(cmd, targetBufer, 0, targetSize);

// OPTION A:
// associate all previous copy operations with a fence
staging.finalizeResources( fence );
..
// every once in a while call
staging.releaseResources();

// OPTION B
// alternatively manage the resource release yourself
sid = staging.finalizeResources();

... you need to ensure these uploads completed

staging.releaseResources();

~~~


### class **nvvk::StagingMemoryManagerDma**

Derives from **nvvk::StagingMemoryManager** and uses the referenced **nvvk::DeviceMemoryAllocator**
for allocations.

~~~ C++
DeviceMemoryAllocator    memAllocator;
memAllocator.init(device, physicalDevice);

StagingMemoryManagerDma  staging;
staging.init(memAllocator);

// rest as usual
staging.cmdToBuffer(cmd, targetBufer, 0, targetSize, targetData);
~~~

## memorymanagement_vkgl.hpp

This file contains helpers for resource interoperability between OpenGL and Vulkan.
they only exist if the shared_sources project is compiled with Vulkan AND OpenGL support.


### class **nvvk::DeviceMemoryAllocatorGL**

Derived from **nvvk::DeviceMemoryAllocator** it uses vulkan memory that is exported
and directly imported into OpenGL. Requires GL_EXT_memory_object.

Used just like the original class however a new function to get the 
GL memory object exists: `getAllocationGL`.

Look at source of **nvvk::AllocatorDmaGL** for usage.

## pipeline_vk.hpp

### functions in nvvk

- **nvprintPipelineStats** : prints stats of the pipeline using VK_KHR_pipeline_executable_properties (don't forget to enable extension and set VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR)
- dumpPipelineStats    : dumps stats of the pipeline using VK_KHR_pipeline_executable_properties to a text file (don't forget to enable extension and set VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR)
- **dumpPipelineBinCodes** : dumps shader binaries using VK_KHR_pipeline_executable_properties to multiple binary files (don't forget to enable extension and set VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR)


### class **nvvk::GraphicsPipelineState**

Most graphic pipelines have similar states, therefore the helper `GraphicsPipelineStage` holds all the elements and 
initialize the structures with the proper default values, such as the primitive type, `PipelineColorBlendAttachmentState` 
with their mask, `DynamicState` for viewport and scissor, adjust depth test if enabled, line width to 1 pixel, for 
example. 

This structure is instantiated using C++ Vulkan objects if VULKAN_HPP is defined, and C otherwise.

Example of usage :
~~~~ c++
nvvk::GraphicsPipelineState pipelineState();
pipelineState.depthStencilState.setDepthTestEnable(true);
pipelineState.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
pipelineState.addBindingDescription({0, sizeof(Vertex)});
pipelineState.addAttributeDescriptions ({
    {0, 0, vk::Format::eR32G32B32Sfloat, static_cast<uint32_t>(offsetof(Vertex, pos))},
    {1, 0, vk::Format::eR32G32B32Sfloat, static_cast<uint32_t>(offsetof(Vertex, nrm))},
    {2, 0, vk::Format::eR32G32B32Sfloat, static_cast<uint32_t>(offsetof(Vertex, col))}});
~~~~


### class **nvvk::GraphicsPipelineGenerator**

The graphics pipeline generator takes a **GraphicsPipelineState** object and pipeline-specific information such as 
the render pass and pipeline layout to generate the final pipeline. 

This structure is instantiated using C++ Vulkan objects if VULKAN_HPP is defined, and C otherwise.

Example of usage :
~~~~ c++
nvvk::GraphicsPipelineState pipelineState();
...
nvvk::GraphicsPipelineGenerator pipelineGenerator(m_device, m_pipelineLayout, m_renderPass, pipelineState);
pipelineGenerator.loadShader(readFile("shaders/vert_shader.vert.spv"), VkShaderStageFlagBits::eVertex);
pipelineGenerator.loadShader(readFile("shaders/frag_shader.frag.spv"), VkShaderStageFlagBits::eFragment);

m_pipeline = pipelineGenerator.createPipeline();
~~~~


### class **nvvk::GraphicsPipelineGeneratorCombined**

In some cases the application may have each state associated to a single pipeline. For convenience, 
**GraphicsPipelineGeneratorCombined** combines both the state and generator into a single object.

Example of usage :
~~~~ c++
nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_pipelineLayout, m_renderPass);
pipelineGenerator.depthStencilState.setDepthTestEnable(true);
pipelineGenerator.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
pipelineGenerator.addBindingDescription({0, sizeof(Vertex)});
pipelineGenerator.addAttributeDescriptions ({
    {0, 0, vk::Format::eR32G32B32Sfloat, static_cast<uint32_t>(offsetof(Vertex, pos))},
    {1, 0, vk::Format::eR32G32B32Sfloat, static_cast<uint32_t>(offsetof(Vertex, nrm))},
    {2, 0, vk::Format::eR32G32B32Sfloat, static_cast<uint32_t>(offsetof(Vertex, col))}});

pipelineGenerator.loadShader(readFile("shaders/vert_shader.vert.spv"), VkShaderStageFlagBits::eVertex);
pipelineGenerator.loadShader(readFile("shaders/frag_shader.frag.spv"), VkShaderStageFlagBits::eFragment);

m_pipeline = pipelineGenerator.createPipeline();
~~~~

## profiler_vk.hpp

### class **nvvk::ProfilerVK**

**ProfilerVK** derives from **nvh::Profiler** and uses `vkCmdWriteTimestamp`
to measure the gpu time within a section.

If profiler.setMarkerUsage(true) was used then it will make use
of `vkCmdDebugMarkerBeginEXT` and `vkCmdDebugMarkerEndEXT` for each
section so that it shows up in tools like NsightGraphics and renderdoc.

Currently the commandbuffers must support `vkCmdResetQueryPool` as well.

When multiple queues are used there could be problems with the "nesting"
of sections. In that case multiple profilers, one per queue, are most
likely better.


Example:

``` c++
nvvk::ProfilerVK profiler;
std::string     profilerStats;

profiler.init(device, physicalDevice);
profiler.setMarkerUsage(true); // depends on VK_EXT_debug_utils

while(true)
{
  profiler.beginFrame();

  ... setup frame ...


  {
    // use the Section class to time the scope
    auto sec = profiler.timeRecurring("draw", cmd);

    vkCmdDraw(cmd, ...);
  }

  ... submit cmd buffer ...

  profiler.endFrame();

  // generic print to string
  profiler.print(profilerStats);

  // or access data directly
  nvh::Profiler::TimerInfo info;
  if( profiler.getTimerInfo("draw", info)) {
    // do some updates
    updateProfilerUi("draw", info.gpu.average);
  }
}

```

## raytraceKHR_vk.hpp

### class **nvvk::RaytracingBuilderKHR**

Base functionality of raytracing

This class does not implement all what you need to do raytracing, but
helps creating the BLAS and TLAS, which then can be used by different
raytracing usage.

### Setup and Usage
~~~~ C++
m_rtBuilder.setup(device, memoryAllocator, queueIndex);
// Create array of VkGeometryNV
m_rtBuilder.buildBlas(allBlas);
// Create array of RaytracingBuilder::instance
m_rtBuilder.buildTlas(instances);
// Retrieve the acceleration structure
const VkAccelerationStructureNV& tlas = m.rtBuilder.getAccelerationStructure()
~~~~

## raytraceNV_vk.hpp

### class **nvvk::RaytracingBuilderNV**

Base functionality of raytracing

This class does not implement all what you need to do raytracing, but
helps creating the BLAS and TLAS, which then can be used by different
raytracing usage.

### Setup and Usage
~~~~ C++
m_rtBuilder.setup(device, memoryAllocator, queueIndex);
// Create array of VkGeometryNV
m_rtBuilder.buildBlas(allBlas);
// Create array of RaytracingBuilder::instance
m_rtBuilder.buildTlas(instances);
// Retrieve the acceleration structure
const VkAccelerationStructureNV& tlas = m.rtBuilder.getAccelerationStructure()
~~~~

## renderpasses_vk.hpp

### functions in nvvk

- **findSupportedFormat** : returns supported `VkFormat` from a list of candidates (returns first match)
- **findDepthFormat** : returns supported depth format (24, 32, 16-bit)
- **findDepthStencilFormat** : returns supported depth-stencil format (24/8, 32/8, 16/8-bit)
- **createRenderPass** : wrapper for vkCreateRenderPass

## samplers_vk.hpp

### **nvvk::SamplerPool**

This class manages unique `VkSampler` objects. To minimize the total
number of sampler objects, this class ensures that identical configurations
return the same sampler

Example :
~~~C++
nvvk::SamplerPool pool(device);

for (auto it : textures) {
  VkSamplerCreateInfo info = {...};

  // acquire ensures we create the minimal subset of samplers
  it.sampler = pool.acquireSampler(info);
}

// you can manage releases individually, or just use deinit/destructor of pool
for (auto it : textures) {
  pool.releaseSampler(it.sampler);
}
~~~

- **makeSamplerCreateInfo** : aids for sampler creation

## shadermodulemanager_vk.hpp

### class **nvvk::ShaderModuleManager**

The **ShaderModuleManager** manages `VkShaderModules` stored in files (SPIR-V or GLSL)

Using **ShaderFileManager** it will find the files and resolve #include for GLSL.
You must add include directories to the base-class for this.

It also comes with some convenience functions to reload shaders etc.
That is why we pass out the **ShaderModuleID** rather than a `VkShaderModule` directly.

To change the compilation behavior manipulate the public member variables
prior createShaderModule.

m_filetype is crucial for this. You can pass raw spir-v files or GLSL.
If GLSL is used, shaderc must be used as well (which must be added via 
_add_package_ShaderC() in CMake of the project)

Example:

``` c++
ShaderModuleManager mgr(myDevice);

// derived from ShaderFileManager
mgr.addDirectory("shaders/");

// all shaders get this injected after #version statement
mgr.m_prepend = "#define USE_NOISE 1\n";

vid = mgr.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT,   "object.vert.glsl");
fid = mgr.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "object.frag.glsl");

// ... later use module
info.module = mgr.get(vid);
```

## shaders_vk.hpp

### function in nvvk

- **createShaderModule** : create the shader module from various binary code inputs
- createShaderStageInfo: create the shader module and setup the stage from the incoming binary code

## structs_vk.hpp

### function nvvk::make, nvvk::clear
Contains templated `nvvk::make<T>` and `nvvk::clear<T>` functions that are 
auto-generated by `structs.lua`. The functions provide default 
structs for the Vulkan C api by initializing the `VkStructureType sType`
field (also for nested structs) and clearing the rest to zero.

``` c++
auto compCreateInfo = nvvk::make<VkComputePipelineCreateInfo>;
```

## swapchain_vk.hpp

### class **nvvk::SwapChain**

Its role is to help using `VkSwapchainKHR`. In Vulkan we have 
to synchronize the backbuffer access ourselves, meaning we
must not write into images that the operating system uses for
presenting the image on the desktop or monitor.

For each swapchain image there is an imageView,
and one read and write semaphore. Furthermore there
is a utility function to setup the image transitions from 
VK_IMAGE_LAYOUT_UNDEFINED to VK_IMAGE_LAYOUT_PRESENT_SRC_KHR.

Example in combination with **nvvk::Context** :

* get the window handle
* create its related surface
* make sure the **Queue** is the one we need to render in this surface

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




_____
auto-generated by `docgen.lua`
