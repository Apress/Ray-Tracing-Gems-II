// Copyright 2021 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
#include <array>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <fileformats/stb_image_write.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <fileformats/tiny_obj_loader.h>
#include <nvh/fileoperations.hpp>  // For nvh::loadFile
#include <nvpsystem.hpp>           // For exePath() and logging
#define NVVK_ALLOC_DMA
#include <nvvk/allocator_vk.hpp>   // For NVVK memory allocators
#include <nvvk/debug_util_vk.hpp>  // For nvvk::DebugUtil
#include <nvvk/extensions_vk.hpp>

#include "common.h"


static const uint64_t render_width  = 3200;
static const uint64_t render_height = 2400;

// Define a callback to capture debug messages.
// (This is here so that we can show adding extensions with raw Vulkan, since
// a point of this sample is to show device creation.)
VKAPI_ATTR VkBool32 VKAPI_CALL debugMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
                                                      VkDebugUtilsMessageTypeFlagsEXT             messageType,
                                                      const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
                                                      void*                                       userData)
{
  int level = LOGLEVEL_INFO;
  // repeating nvprintfLevel to help with breakpoints : so we can selectively break right after the print
  if(messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT)
  {
    nvprintfLevel(level, "VERBOSE: %s \n --> %s\n", callbackData->pMessageIdName, callbackData->pMessage);
  }
  else if(messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
  {
    nvprintfLevel(level, "INFO: %s \n --> %s\n", callbackData->pMessageIdName, callbackData->pMessage);
  }
  else if(messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
  {
    level = LOGLEVEL_WARNING;
    nvprintfLevel(level, "WARNING: %s \n --> %s\n", callbackData->pMessageIdName, callbackData->pMessage);
  }
  else if(messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
  {
    level = LOGLEVEL_ERROR;
    nvprintfLevel(level, "ERROR: %s \n --> %s\n", callbackData->pMessageIdName, callbackData->pMessage);
  }
  else if(messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT)
  {
    nvprintfLevel(level, "GENERAL: %s \n --> %s\n", callbackData->pMessageIdName, callbackData->pMessage);
  }
  else
  {
    nvprintfLevel(level, "%s \n --> %s\n", callbackData->pMessageIdName, callbackData->pMessage);
  }

  // Don't bail out, but keep going.
  return VK_FALSE;
}


// Allocates and begins a one-time command buffer from the command pool.
VkCommandBuffer AllocateAndBeginOneTimeCommandBuffer(VkDevice device, VkCommandPool cmdPool)
{
  VkCommandBufferAllocateInfo cmdAllocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cmdAllocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdAllocInfo.commandPool        = cmdPool;
  cmdAllocInfo.commandBufferCount = 1;
  VkCommandBuffer cmdBuffer;
  NVVK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmdBuffer));

  VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  NVVK_CHECK(vkBeginCommandBuffer(cmdBuffer, &beginInfo));
  return cmdBuffer;
}


// Ends recording, submits, waits, and then frees the command buffer.
void EndSubmitWaitAndFreeCommandBuffer(VkDevice device, VkQueue queue, VkCommandPool cmdPool, VkCommandBuffer& cmdBuffer)
{
  NVVK_CHECK(vkEndCommandBuffer(cmdBuffer));

  VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers    = &cmdBuffer;
  NVVK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

  NVVK_CHECK(vkQueueWaitIdle(queue));

  vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
}


// Gets the device address of a buffer.
VkDeviceAddress GetBufferDeviceAddress(VkDevice device, VkBuffer buffer)
{
  VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  addressInfo.buffer = buffer;
  return vkGetBufferDeviceAddress(device, &addressInfo);
}


// Creates a shader module from an std::string containing SPIR-V bytecode.
VkShaderModule CreateShaderModule(VkDevice device, const std::string& binaryCode)
{
  const size_t    spirvSize = binaryCode.size();
  const uint32_t* spirvData = reinterpret_cast<const uint32_t*>(binaryCode.data());

  VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  createInfo.codeSize = spirvSize;
  createInfo.pCode    = spirvData;

  VkShaderModule shaderModule = VK_NULL_HANDLE;
  NVVK_CHECK(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule));

  return shaderModule;
}


bool AreExtensionsIncluded(const std::vector<const char*> requestedExtensions, const std::vector<VkExtensionProperties> deviceExtensionProperties)
{
  for(const char* extensionName : requestedExtensions)
  {
    bool extensionFound = false;
    for(const VkExtensionProperties& property : deviceExtensionProperties)
    {
      if(strcmp(extensionName, property.extensionName) == 0)
      {
        extensionFound = true;
        break;
      }
    }

    if(!extensionFound)
    {
      return false;
    }
  }

  return true;
}


int main(int argc, const char** argv)
{
  NVPSystem sys(argv[0], PROJECT_NAME);  // Sets up exePath() and logging

  // Initialize the Vulkan instance, physical device, and logical device.
  // For this sample, we show how to do this manually, instead of using
  // nvvk::Context. This is so that the sample code shows adding extensions
  // without references to NVVK.
  VkInstance                          instance;
  VkDebugUtilsMessengerEXT            debugMessenger;
  PFN_vkDestroyDebugUtilsMessengerEXT destroyDebugUtilsMessengerEXT;
  VkPhysicalDevice                    physicalDevice = VK_NULL_HANDLE;
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtPipelineProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkDevice                                        device;
  VkQueue                                         queueGCT;
  uint32_t                                        queueGCTFamilyIndex;
  {
    // Instance creation, requesting Vulkan version 1.2.0
    {
      VkApplicationInfo applicationInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
      applicationInfo.pApplicationName = "vk_ray_tracing_gems_2_ao";
      applicationInfo.pEngineName      = "vk_ray_tracing_gems_2_ao";
      applicationInfo.apiVersion       = VK_MAKE_VERSION(1, 2, 0);

      // Enable the debug utils extension. Fail if not available.
      const char* debugUtilsExtensionName = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
      // Enable the validation layer. Fail if not available.
      const char* validationLayerName = "VK_LAYER_KHRONOS_validation";

      VkInstanceCreateInfo instanceCreateInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
      instanceCreateInfo.pApplicationInfo        = &applicationInfo;
      instanceCreateInfo.enabledExtensionCount   = 1;
      instanceCreateInfo.ppEnabledExtensionNames = &debugUtilsExtensionName;
      instanceCreateInfo.enabledLayerCount       = 1;
      instanceCreateInfo.ppEnabledLayerNames     = &validationLayerName;

      NVVK_CHECK(vkCreateInstance(&instanceCreateInfo, nullptr, &instance));
    }

    // Initialize the debug utils messenger.
    {
      // Since VK_EXT_debug_utils isn't in core, we need to manually load it:
      auto createDebugUtilsMessengerEXT =
          (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
      destroyDebugUtilsMessengerEXT =
          (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

      // Create a Debug Utils messenger that will trigger the callback for any
      // info message, warning,  or error.
      VkDebugUtilsMessengerCreateInfoEXT debugMessengerInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
      debugMessengerInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                                           | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      debugMessengerInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                                       | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
      debugMessengerInfo.pfnUserCallback = debugMessengerCallback;
      NVVK_CHECK(createDebugUtilsMessengerEXT(instance, &debugMessengerInfo, nullptr, &debugMessenger));
    }

    // Get a list of the physical devices.
    std::vector<VkPhysicalDevice> physicalDevices;
    {
      uint32_t numPhysicalDevices;
      NVVK_CHECK(vkEnumeratePhysicalDevices(instance, &numPhysicalDevices, nullptr));
      physicalDevices.resize(numPhysicalDevices);
      NVVK_CHECK(vkEnumeratePhysicalDevices(instance, &numPhysicalDevices, physicalDevices.data()));
    }

    // A table of the extensions we'll request.
    std::vector<const char*> deviceExtensions = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                                 VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
                                                 VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME};

    // Find a physical device that supports all above extensions.
    for(VkPhysicalDevice consideredDevice : physicalDevices)
    {
      // Get the list of extensions the device supports
      uint32_t numExtensions;
      NVVK_CHECK(vkEnumerateDeviceExtensionProperties(consideredDevice, nullptr, &numExtensions, nullptr));
      std::vector<VkExtensionProperties> extensionProperties(numExtensions);
      NVVK_CHECK(vkEnumerateDeviceExtensionProperties(consideredDevice, nullptr, &numExtensions, extensionProperties.data()));

      if(AreExtensionsIncluded(deviceExtensions, extensionProperties))
      {
        physicalDevice = consideredDevice;
        break;
      }
    }

    assert(physicalDevice != VK_NULL_HANDLE);  // At least one device must be found


    // Get the properties of ray tracing pipelines on this physical device,
    // using the following structure chain:
    // VkPhysicalDeviceProperties2 -> VkPhysicalDeviceRayTracingPipelinePropertiesKHR
    {
      VkPhysicalDeviceProperties2 physicalDeviceProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
      physicalDeviceProperties.pNext = &rtPipelineProperties;
      vkGetPhysicalDeviceProperties2(physicalDevice, &physicalDeviceProperties);
    }


    // Find a general-purpose (graphics, compute, and transfer) queue family
    float                   queue0Priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    {
      // Get the properties of each queue family of the physical device.
      uint32_t numQueueFamilies;
      vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueueFamilies, nullptr);
      std::vector<VkQueueFamilyProperties> queueProperties(numQueueFamilies);
      vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueueFamilies, queueProperties.data());

      // Select a queue family that is general-purpose.
      bool queueFamilyFound = false;
      for(uint32_t i = 0; i < numQueueFamilies; i++)
      {
        VkQueueFamilyProperties& properties = queueProperties[i];
        const VkQueueFlags       gctFlags   = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
        if((properties.queueFlags & gctFlags) == gctFlags)
        {
          // Choose a queue from the queue family
          queueInfo.queueFamilyIndex = i;
          queueInfo.queueCount       = 1;
          queueInfo.pQueuePriorities = &queue0Priority;
          queueFamilyFound           = true;
          break;
        }
      }

      assert(queueFamilyFound);
    }

    // Create the following structure chain:
    // VkPhysicalDeviceFeatures2
    // -> VkPhysicalDeviceVulkan12Features
    // -> VkPhysicalDeviceVulkan11Features
    // -> VkPhysicalDeviceAccelerationStructureFeaturesKHR
    // -> VkPhysicalDeviceRayTracingPipelineFeaturesKHR
    // We'll query which features are supported. Then we'll use the populated
    // feature structure chain to create the device.
    VkPhysicalDeviceFeatures2                        features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan12Features                 features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features                 features11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    features2.pNext  = &features12;
    features12.pNext = &features11;
    features11.pNext = &asFeatures;
    asFeatures.pNext = &rtPipelineFeatures;

    // Query supported features
    vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

    // Create the Vulkan device
    {
      VkDeviceCreateInfo deviceCreateInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
      deviceCreateInfo.enabledExtensionCount   = uint32_t(deviceExtensions.size());
      deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
      deviceCreateInfo.pEnabledFeatures        = nullptr;
      deviceCreateInfo.pNext                   = &features2;
      deviceCreateInfo.queueCreateInfoCount    = 1;
      deviceCreateInfo.pQueueCreateInfos       = &queueInfo;
      NVVK_CHECK(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));
    }

    // Load a subset of instance and device extension functions
    load_VK_EXTENSION_SUBSET(instance, vkGetInstanceProcAddr, device, vkGetDeviceProcAddr);

    // Get the general (GCT) queue.
    vkGetDeviceQueue(device, queueInfo.queueFamilyIndex, 0, &queueGCT);
    queueGCTFamilyIndex = queueInfo.queueFamilyIndex;
  }

  // Initialize the debug utilities:
  nvvk::DebugUtil debugUtil(device);
  debugUtil.setEnabled(true);


  // Get the properties of ray tracing pipelines on this device.
  const VkDeviceSize sbtHeaderSize      = rtPipelineProperties.shaderGroupHandleSize;
  const VkDeviceSize sbtBaseAlignment   = rtPipelineProperties.shaderGroupBaseAlignment;
  const VkDeviceSize sbtHandleAlignment = rtPipelineProperties.shaderGroupHandleAlignment;

  // Compute the stride between shader binding table (SBT) records.
  // This must be:
  // - Greater than rtPipelineProperties.shaderGroupHandleSize (since a record
  //     contains a shader group handle)
  // - A multiple of rtPipelineProperties.shaderGroupHandleAlignment
  // - Less than or equal to rtPipelineProperties.maxShaderGroupStride
  // In addition, each SBT must start at a multiple of
  // rtPipelineProperties.shaderGroupBaseAlignment.
  // Since we store all records contiguously in a single SBT, we assert that
  // sbtBaseAlignment is a multiple of sbtHandleAlignment, round sbtHeaderSize
  // up to a multiple of sbtBaseAlignment, and then assert that the result is
  // less than or equal to maxShaderGroupStride.
  assert(sbtBaseAlignment % sbtHandleAlignment == 0);
  const VkDeviceSize sbtStride = sbtBaseAlignment *  //
                                 ((sbtHeaderSize + sbtBaseAlignment - 1) / sbtBaseAlignment);
  assert(sbtStride <= rtPipelineProperties.maxShaderGroupStride);


  nvvk::DeviceMemoryAllocator allocatorDma;
  allocatorDma.setAllocateFlags(VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR, true);
  allocatorDma.init(device, physicalDevice);
  nvvk::Allocator allocator;
  allocator.init(device, physicalDevice, &allocatorDma);


  // Create the ambient occlusion image, which is a VK_FORMAT_R8_UNORM image,
  // and its view.
  nvvk::Image imageAO;
  VkImageView imageAOView;
  {
    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageInfo.extent        = {render_width, render_height, 1};
    imageInfo.mipLevels     = 1;
    imageInfo.arrayLayers   = 1;
    imageInfo.format        = VK_FORMAT_R8_UNORM;
    imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage         = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imageAO                 = allocator.createImage(imageInfo);
    debugUtil.setObjectName(imageAO.image, "imageAO");

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image                       = imageAO.image;
    viewInfo.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format                      = imageInfo.format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    NVVK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &imageAOView));
    debugUtil.setObjectName(imageAOView, "imageAOView");
  }

  // Staging buffer for copy back to the CPU
  nvvk::Buffer bufferStaging = allocator.createBuffer(
      sizeof(uint8_t) * render_width * render_height, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
  debugUtil.setObjectName(bufferStaging.buffer, "bufferStaging");

  // Set up search paths
  std::vector<std::string> searchPaths = {NVPSystem::exePath() + PROJECT_NAME, NVPSystem::exePath() + "media",
                                          NVPSystem::exePath() + PROJECT_RELDIRECTORY,
                                          NVPSystem::exePath() + PROJECT_RELDIRECTORY "media",
                                          NVPSystem::exePath() + PROJECT_DOWNLOAD_RELDIRECTORY};

  // Load the mesh of the first shape from an OBJ file
  std::vector<tinyobj::real_t> objVertices;
  std::vector<uint32_t>        objIndices;
  {
    tinyobj::ObjReader reader;  // Used to read an OBJ file
    reader.ParseFromFile(nvh::findFile("Medieval_building_flat_faces.obj", searchPaths));
    assert(reader.Valid());  // Make sure tinyobj was able to parse this file
    objVertices                                    = reader.GetAttrib().GetVertices();
    const std::vector<tinyobj::shape_t>& objShapes = reader.GetShapes();  // All shapes in the file
    assert(objShapes.size() == 1);                                        // Check that this file has only one shape
    const tinyobj::shape_t& objShape = objShapes[0];                      // Get the first shape
    // Get the indices of the vertices of the first mesh of `objShape` in `attrib.vertices`:
    objIndices.reserve(objShape.mesh.indices.size());
    for(const tinyobj::index_t& index : objShape.mesh.indices)
    {
      objIndices.push_back(index.vertex_index);
    }
  }


  // Create the command pool
  VkCommandPool cmdPool;
  {
    VkCommandPoolCreateInfo cmdPoolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cmdPoolInfo.queueFamilyIndex = queueGCTFamilyIndex;
    NVVK_CHECK(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool));
  }


  // Upload the vertex and index buffers to the GPU.
  nvvk::Buffer bufferVertex, bufferIndex;
  {
    // Start a command buffer for uploading the buffers
    VkCommandBuffer cmdBuffer = AllocateAndBeginOneTimeCommandBuffer(device, cmdPool);
    // We get these buffers' device addresses, and use them as storage buffers and build inputs.
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                     | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    bufferVertex = allocator.createBuffer(cmdBuffer, objVertices, usage);
    bufferIndex  = allocator.createBuffer(cmdBuffer, objIndices, usage);

    // Also transfer the image layout to its future layout:
    VkImageMemoryBarrier imageBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    imageBarrier.srcAccessMask               = 0;
    imageBarrier.dstAccessMask               = VK_ACCESS_SHADER_WRITE_BIT;
    imageBarrier.oldLayout                   = VK_IMAGE_LAYOUT_UNDEFINED;
    imageBarrier.newLayout                   = VK_IMAGE_LAYOUT_GENERAL;
    imageBarrier.image                       = imageAO.image;
    imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBarrier.subresourceRange.levelCount = 1;
    imageBarrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(cmdBuffer,                                     //
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,             // From no stages
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,  // To ray tracing shader stages
                         0,                                             //
                         0, nullptr,                                    //
                         0, nullptr,                                    //
                         1, &imageBarrier);

    EndSubmitWaitAndFreeCommandBuffer(device, queueGCT, cmdPool, cmdBuffer);
    allocator.finalizeAndReleaseStaging();
  }


  // Descriptor set variables
  VkDescriptorSetLayout descriptorSetLayout;
  VkDescriptorPool      descriptorPool;
  VkPipelineLayout      pipelineLayout;
  VkDescriptorSet       descriptorSet;  // This app uses only 1 descriptor set

  // Ray tracing setup
  // Based on the code in nvvk/raytraceKHR_vk.hpp.
  nvvk::Buffer               bufferBLAS, bufferTLAS;
  VkAccelerationStructureKHR blas, tlas;
  VkShaderModule             aoRgen, aoRMiss, aoCHitPrimary;
  VkPipeline                 pipelineRayTrace;
  nvvk::Buffer               bufferSBT;
  {
    // Build a bottom-level acceleration structure (BLAS).
    {
      // Start a command buffer:
      VkCommandBuffer cmdBuffer = AllocateAndBeginOneTimeCommandBuffer(device, cmdPool);

      // Get the device addresses of the vertex and index buffers:
      VkDeviceAddress vertexBufferAddress = GetBufferDeviceAddress(device, bufferVertex.buffer);
      VkDeviceAddress indexBufferAddress  = GetBufferDeviceAddress(device, bufferIndex.buffer);

      // Specify where the builder can find the vertices and indices for
      // triangles, and their formats:
      VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
      triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;
      triangles.vertexData.deviceAddress = vertexBufferAddress;
      triangles.vertexStride             = 3 * sizeof(float);
      triangles.indexType                = VK_INDEX_TYPE_UINT32;
      triangles.indexData.deviceAddress  = indexBufferAddress;
      triangles.maxVertex                = uint32_t(objVertices.size() - 1);
      triangles.transformData            = {0};  // No transform

      // Encapsulate this in a polymorphic acceleration structure geometry object:
      VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
      geometry.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
      geometry.geometry.triangles = triangles;
      // Disable anyhit for faster performance by setting geometry flags:
      geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

      // Create offset info that says how many triangles (and thus vertices)
      // to read:
      VkAccelerationStructureBuildRangeInfoKHR rangeInfo;
      rangeInfo.firstVertex     = 0;
      rangeInfo.primitiveCount  = uint32_t(objIndices.size() / 3);  // Number of triangles
      rangeInfo.primitiveOffset = 0;
      rangeInfo.transformOffset = 0;

      // This struct points to an array of geometries to be built into the
      // BLAS, as well as build settings.
      VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
      buildInfo.flags                    = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
      buildInfo.geometryCount            = 1;
      buildInfo.pGeometries              = &geometry;
      buildInfo.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      buildInfo.type                     = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
      // We'll set dstAccelerationStructure and scratchData once we've created
      // those objects.

      // We need to allocate memory to build and to store the BLAS. Query the
      // worst-case size of the acceleration structure and the amount of
      // scratch memory needed, based on the number of primitives.
      VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
      vkGetAccelerationStructureBuildSizesKHR(              //
          device,                                           // The device
          VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,  // Built on device instead of host
          &buildInfo,                                       // Pointer to build info
          &rangeInfo.primitiveCount,                        // Array of numbers of primitives
          &sizeInfo);                                       // Pointer to store sizes

      // Allocate a buffer for the acceleration structure.
      bufferBLAS = allocator.createBuffer(sizeInfo.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                                                                  | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                                                  | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      // Create the acceleration structure object. (Data has not yet been set.)
      VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
      createInfo.type   = buildInfo.type;
      createInfo.size   = sizeInfo.accelerationStructureSize;
      createInfo.buffer = bufferBLAS.buffer;
      createInfo.offset = 0;
      NVVK_CHECK(vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, &blas));
      buildInfo.dstAccelerationStructure = blas;

      // Allocate the scratch buffer holding temporary build data.
      nvvk::Buffer scratchBuffer = allocator.createBuffer(sizeInfo.buildScratchSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                                                         | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      buildInfo.scratchData.deviceAddress = GetBufferDeviceAddress(device, scratchBuffer.buffer);

      // Since Vulkan requries an array of pointers to
      // VkAccelerationStructureBuildRangeInfoKHR objects, get a pointer to
      // rangeInfo:
      VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

      // Build one acceleration structure:
      vkCmdBuildAccelerationStructuresKHR(cmdBuffer, 1, &buildInfo, &pRangeInfo);

      // End, submit, and wait for the command buffer (synchronously):
      EndSubmitWaitAndFreeCommandBuffer(device, queueGCT, cmdPool, cmdBuffer);

      // Free the scratch buffer:
      allocator.destroy(scratchBuffer);
    }

    // Build a top-level acceleration structure (TLAS).
    {
      // Start a command buffer:
      VkCommandBuffer cmdBuffer = AllocateAndBeginOneTimeCommandBuffer(device, cmdPool);

      // Get the device address of the BLAS:
      VkAccelerationStructureDeviceAddressInfoKHR addressInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
      addressInfo.accelerationStructure = blas;
      VkDeviceAddress blasAddress       = vkGetAccelerationStructureDeviceAddressKHR(device, &addressInfo);

      // This TLAS will have one instance of the BLAS.
      VkAccelerationStructureInstanceKHR instance{};  // Zero-initialize
      // Set the instance transform to a 135-degree rotation around the y axis:
      const float rcpSqrt2                            = sqrtf(0.5f);
      instance.transform.matrix[0][0]                 = -rcpSqrt2;
      instance.transform.matrix[0][2]                 = rcpSqrt2;
      instance.transform.matrix[1][1]                 = 1.0f;
      instance.transform.matrix[2][0]                 = -rcpSqrt2;
      instance.transform.matrix[2][2]                 = -rcpSqrt2;
      instance.instanceCustomIndex                    = 0;                         // Custom 24-bit int per instance
      instance.mask                                   = 0xFF;                      // Ray mask
      instance.instanceShaderBindingTableRecordOffset = 0;                         // SBT offset
      instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;  // Disable backface culling
      instance.accelerationStructureReference = blasAddress;                       // Pointer to BLAS

      VkAccelerationStructureBuildRangeInfoKHR rangeInfo;
      rangeInfo.primitiveOffset = 0;
      rangeInfo.primitiveCount  = 1;
      rangeInfo.firstVertex     = 0;
      rangeInfo.transformOffset = 0;

      // Upload the instance struct to the device.
      nvvk::Buffer bufferInstances =
          allocator.createBuffer(cmdBuffer, sizeof(instance), &instance, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

      // Add a memory barrier to ensure that createBuffer's upload command
      // finishes before starting the TLAS build.
      VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
      vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);

      VkAccelerationStructureGeometryInstancesDataKHR instancesVk{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
      instancesVk.arrayOfPointers    = VK_FALSE;
      instancesVk.data.deviceAddress = GetBufferDeviceAddress(device, bufferInstances.buffer);

      // Like creating the BLAS, point to the geometry (in this case the
      // instances) in a polymorphic object:
      VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
      geometry.geometryType       = VK_GEOMETRY_TYPE_INSTANCES_KHR;
      geometry.geometry.instances = instancesVk;

      // Create the build info - in this case, pointing to only one
      // geometry object.
      VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
      buildInfo.flags                    = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
      buildInfo.geometryCount            = 1;
      buildInfo.pGeometries              = &geometry;
      buildInfo.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      buildInfo.type                     = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
      buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;

      // Query the worst-case AS size and scratch space size based on the
      // number of instances (in this case, 1)
      VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
      vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo,
                                              &rangeInfo.primitiveCount, &sizeInfo);

      // Allocate a buffer for the acceleration structure.
      bufferTLAS = allocator.createBuffer(sizeInfo.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                                                                  | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                                                  | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

      // Create the acceleration structure object. (Data has not yet been set.)
      VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
      createInfo.type   = buildInfo.type;
      createInfo.size   = sizeInfo.accelerationStructureSize;
      createInfo.buffer = bufferTLAS.buffer;
      createInfo.offset = 0;
      NVVK_CHECK(vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, &tlas));
      buildInfo.dstAccelerationStructure = tlas;

      // Allocate the scratch buffer holding temporary build data.
      nvvk::Buffer bufferScratch = allocator.createBuffer(sizeInfo.buildScratchSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                                                         | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      buildInfo.scratchData.deviceAddress = GetBufferDeviceAddress(device, bufferScratch.buffer);

      // Create a 1-element array of pointers to range info objects:
      VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

      // Build the TLAS
      vkCmdBuildAccelerationStructuresKHR(cmdBuffer, 1, &buildInfo, &pRangeInfo);

      // End, submit, and wait for the command buffer (synchronously):
      EndSubmitWaitAndFreeCommandBuffer(device, queueGCT, cmdPool, cmdBuffer);

      // Clean up scratch memory:
      allocator.destroy(bufferScratch);
      allocator.destroy(bufferInstances);
    }


    // Create the descriptor set structure
    // Descriptor set layout
    {
      // List all bindings:
      // AO: 1 storage image, accessible from the raygen stage
      // TLAS: 1 acceleration structure, accessible from the raygen stage
      // Vertices: 1 buffer, accessible from the closest hit stage
      // Indices: 1 buffer, accessible from the closest hit stage
      std::array<VkDescriptorSetLayoutBinding, 4> bindings;

      bindings[0].binding         = BINDING_IMAGE_AO;
      bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      bindings[0].descriptorCount = 1;
      bindings[0].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

      bindings[1].binding         = BINDING_TLAS;
      bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
      bindings[1].descriptorCount = 1;
      bindings[1].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

      bindings[2].binding         = BINDING_VERTICES;
      bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[2].descriptorCount = 1;
      bindings[2].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

      bindings[3]         = bindings[2];
      bindings[3].binding = BINDING_INDICES;

      VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
      layoutInfo.bindingCount = uint32_t(bindings.size());
      layoutInfo.pBindings    = bindings.data();
      NVVK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout));
    }

    // Descriptor pool with enough space for 1 set
    {
      // 1 set needs space for 1 storage image descriptors, 1 TLAS descriptor,
      // and 2 storage buffers.
      std::array<VkDescriptorPoolSize, 3> poolSizes;

      poolSizes[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      poolSizes[0].descriptorCount = 1;

      poolSizes[1].type            = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
      poolSizes[1].descriptorCount = 1;

      poolSizes[2].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      poolSizes[2].descriptorCount = 2;

      VkDescriptorPoolCreateInfo descriptorPoolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      descriptorPoolInfo.maxSets       = 1;
      descriptorPoolInfo.poolSizeCount = uint32_t(poolSizes.size());
      descriptorPoolInfo.pPoolSizes    = poolSizes.data();
      NVVK_CHECK(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
    }

    // Pipeline layout from the descriptor set layout, with no additional
    // push constants
    {
      VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      pipelineLayoutInfo.setLayoutCount = 1;
      pipelineLayoutInfo.pSetLayouts    = &descriptorSetLayout;
      NVVK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout));
    }

    // Allocate 1 descriptor set of the given layout from the descriptor pool
    {
      VkDescriptorSetAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      allocInfo.descriptorPool              = descriptorPool;
      allocInfo.descriptorSetCount          = 1;
      allocInfo.pSetLayouts                 = &descriptorSetLayout;
      NVVK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
    }


    // Write values into the descriptor set
    // When used, the images must be in GENERAL layout.
    {
      std::array<VkWriteDescriptorSet, 4> writeDescriptorSets{};

      // Ambient occlusion image
      VkDescriptorImageInfo descriptorImageAOInfo{};
      descriptorImageAOInfo.imageLayout      = VK_IMAGE_LAYOUT_GENERAL;
      descriptorImageAOInfo.imageView        = imageAOView;
      writeDescriptorSets[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writeDescriptorSets[0].descriptorCount = 1;
      writeDescriptorSets[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      writeDescriptorSets[0].dstBinding      = BINDING_IMAGE_AO;
      writeDescriptorSets[0].dstSet          = descriptorSet;
      writeDescriptorSets[0].pImageInfo      = &descriptorImageAOInfo;

      // TLAS
      VkWriteDescriptorSetAccelerationStructureKHR descriptorAS{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
      descriptorAS.accelerationStructureCount = 1;
      descriptorAS.pAccelerationStructures    = &tlas;
      writeDescriptorSets[1].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writeDescriptorSets[1].descriptorCount  = 1;
      writeDescriptorSets[1].descriptorType   = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
      writeDescriptorSets[1].dstBinding       = BINDING_TLAS;
      writeDescriptorSets[1].dstSet           = descriptorSet;
      writeDescriptorSets[1].pNext            = &descriptorAS;

      // Vertex buffer
      VkDescriptorBufferInfo descriptorVtxInfo{};
      descriptorVtxInfo.buffer               = bufferVertex.buffer;
      descriptorVtxInfo.offset               = 0;
      descriptorVtxInfo.range                = VK_WHOLE_SIZE;
      writeDescriptorSets[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writeDescriptorSets[2].descriptorCount = 1;
      writeDescriptorSets[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writeDescriptorSets[2].dstBinding      = BINDING_VERTICES;
      writeDescriptorSets[2].dstSet          = descriptorSet;
      writeDescriptorSets[2].pBufferInfo     = &descriptorVtxInfo;

      // Index buffer
      VkDescriptorBufferInfo descriptorIdxInfo = descriptorVtxInfo;
      descriptorIdxInfo.buffer                 = bufferIndex.buffer;
      writeDescriptorSets[3]                   = writeDescriptorSets[2];
      writeDescriptorSets[3].dstBinding        = BINDING_INDICES;
      writeDescriptorSets[3].pBufferInfo       = &descriptorIdxInfo;

      vkUpdateDescriptorSets(device,                                // The device
                             uint32_t(writeDescriptorSets.size()),  // Number of VkWriteDescriptorSet objects
                             writeDescriptorSets.data(),            // Pointer to VkWriteDescriptorSet objects
                             0, nullptr);                           // An array of VkCopyDescriptorSet objects (unused)
    }


    // Load shaders
    {
      aoRgen        = CreateShaderModule(device, nvh::loadFile("autogen/ao.rgen.spv", true, searchPaths));
      aoRMiss       = CreateShaderModule(device, nvh::loadFile("autogen/ao.rmiss.spv", true, searchPaths));
      aoCHitPrimary = CreateShaderModule(device, nvh::loadFile("autogen/ao_primary.rchit.spv", true, searchPaths));
      // Set their names so they show in validation layer messages:
      debugUtil.setObjectName(aoRgen, "aoRgen");
      debugUtil.setObjectName(aoRMiss, "aoRMiss");
      debugUtil.setObjectName(aoCHitPrimary, "aoCHitPrimary");
    }


    // Create the ray tracing pipeline
    {
      std::array<VkPipelineShaderStageCreateInfo, 3> stages;
      // Ray generation shader stage
      stages[0]        = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
      stages[0].stage  = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
      stages[0].module = aoRgen;
      stages[0].pName  = "main";

      // Miss shader
      stages[1]        = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
      stages[1].stage  = VK_SHADER_STAGE_MISS_BIT_KHR;
      stages[1].module = aoRMiss;
      stages[1].pName  = "main";

      // Closest hit shader (used for primary ray casts)
      stages[2]        = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
      stages[2].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
      stages[2].module = aoCHitPrimary;
      stages[2].pName  = "main";


      std::array<VkRayTracingShaderGroupCreateInfoKHR, 3> groups;
      // Ray generation shader group
      groups[0]                    = {VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
      groups[0].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
      groups[0].generalShader      = 0;
      groups[0].closestHitShader   = VK_SHADER_UNUSED_KHR;
      groups[0].anyHitShader       = VK_SHADER_UNUSED_KHR;
      groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

      // Miss shader group
      groups[1]               = groups[0];
      groups[1].generalShader = 1;

      // Closest hit shader group (used for primary ray casts)
      groups[2]                    = {VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
      groups[2].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
      groups[2].generalShader      = VK_SHADER_UNUSED_KHR;
      groups[2].closestHitShader   = 2;
      groups[2].anyHitShader       = VK_SHADER_UNUSED_KHR;
      groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

      // Now, describe the ray tracing pipeline, ike creating a compute pipeline:
      VkRayTracingPipelineCreateInfoKHR pipelineCreateInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
      pipelineCreateInfo.flags                        = 0;  // No flags to set
      pipelineCreateInfo.stageCount                   = uint32_t(stages.size());
      pipelineCreateInfo.pStages                      = stages.data();
      pipelineCreateInfo.groupCount                   = uint32_t(groups.size());
      pipelineCreateInfo.pGroups                      = groups.data();
      pipelineCreateInfo.maxPipelineRayRecursionDepth = 1;  // Depth of call tree
      pipelineCreateInfo.layout                       = pipelineLayout;
      NVVK_CHECK(vkCreateRayTracingPipelinesKHR(device,                  // Device
                                                VK_NULL_HANDLE,          // Deferred operation or VK_NULL_HANDLE
                                                VK_NULL_HANDLE,          // Pipeline cache or VK_NULL_HANDLE
                                                1, &pipelineCreateInfo,  // Array of create infos
                                                nullptr,                 // Allocator
                                                &pipelineRayTrace));
      debugUtil.setObjectName(pipelineRayTrace, "pipelineRayTrace");


      // Get the shader group handles:
      std::vector<uint8_t> cpuShaderHandleStorage(sbtHeaderSize * groups.size());
      NVVK_CHECK(vkGetRayTracingShaderGroupHandlesKHR(device,                           // Device
                                                      pipelineRayTrace,                 // Pipeline
                                                      0,                                // First group
                                                      uint32_t(groups.size()),          // Number of groups
                                                      cpuShaderHandleStorage.size(),    // Size of buffer
                                                      cpuShaderHandleStorage.data()));  // Data buffer


      // Allocate the shader binding table. We get its device address, and
      // use it as a shader binding table. As before, we set its memory property
      // flags so that it can be read and written from the CPU.
      const uint32_t sbtSize = uint32_t(sbtStride * groups.size());
      bufferSBT              = allocator.createBuffer(sbtSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                             | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      debugUtil.setObjectName(bufferSBT.buffer, "rtSBTBuffer");
      // Copy the shader group handles to the SBT:
      uint8_t* mappedSBT = reinterpret_cast<uint8_t*>(allocator.map(bufferSBT));
      for(size_t groupIndex = 0; groupIndex < groups.size(); groupIndex++)
      {
        memcpy(&mappedSBT[groupIndex * sbtStride], &cpuShaderHandleStorage[groupIndex * sbtHeaderSize], sbtHeaderSize);
      }
      allocator.unmap(bufferSBT);
      // Clean up:
      allocator.finalizeAndReleaseStaging();
    }
  }


  // Create and start recording a command buffer
  VkCommandBuffer cmdBuffer = AllocateAndBeginOneTimeCommandBuffer(device, cmdPool);
  {
    // Ray tracing: Compute ambient occlusion using the position and normal buffers.
    {
      // Bind the ray tracing pipeline
      vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineRayTrace);
      // Bind the descriptor set
      vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

      VkStridedDeviceAddressRegionKHR sbtRegionRGen, sbtRegionMiss, sbtRegionHit, sbtRegionCallable;
      {
        const VkDeviceAddress sbtStartAddress = GetBufferDeviceAddress(device, bufferSBT.buffer);

        // One ray gen shader
        sbtRegionRGen.deviceAddress = sbtStartAddress;
        sbtRegionRGen.stride        = sbtStride;
        sbtRegionRGen.size          = sbtStride;

        // One miss shader
        sbtRegionMiss.deviceAddress = sbtStartAddress + sbtStride;
        sbtRegionMiss.stride        = sbtStride;
        sbtRegionMiss.size          = sbtStride;

        // One closest hit shader
        sbtRegionHit.deviceAddress = sbtStartAddress + 2 * sbtStride;
        sbtRegionHit.stride        = sbtStride;
        sbtRegionHit.size          = sbtStride;

        // No callable shaders
        sbtRegionCallable.deviceAddress = 0;
        sbtRegionCallable.stride        = 0;
        sbtRegionCallable.size          = 0;
      }

      // Trace rays!
      vkCmdTraceRaysKHR(cmdBuffer,           // Command buffer
                        &sbtRegionRGen,      //
                        &sbtRegionMiss,      //
                        &sbtRegionHit,       //
                        &sbtRegionCallable,  //
                        render_width,        //
                        render_height,       //
                        1);
    }

    // Copy the ambient occlusion image to the staging buffer.
    {
      // Transition it to the TRANSFER_SRC_OPTIMAL layout with access
      // for transfer reads:
      VkPipelineStageFlags srcStages = 0, dstStages = 0;
      VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      barrier.srcAccessMask               = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask               = VK_ACCESS_TRANSFER_READ_BIT;
      barrier.oldLayout                   = VK_IMAGE_LAYOUT_GENERAL;
      barrier.newLayout                   = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      barrier.image                       = imageAO.image;
      barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
      barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
      vkCmdPipelineBarrier(cmdBuffer,                                     //
                           VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,  // Source stages
                           VK_PIPELINE_STAGE_TRANSFER_BIT,                // Destination stages
                           0,                                             //
                           0, nullptr,                                    //
                           0, nullptr,                                    //
                           1, &barrier);                                  // Image memory barriers

      // Now, call vkCmdCopyImageToBuffer:
      VkBufferImageCopy region{};
      region.bufferOffset      = 0;  // Write to the start of the buffer
      region.bufferRowLength   = 0;  // Assume image data is tightly packed
      region.bufferImageHeight = 0;  // Assume image data is tightly packed
      // We copy the image aspect, layer 0, mip 0:
      region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      region.imageSubresource.baseArrayLayer = 0;
      region.imageSubresource.layerCount     = 1;
      region.imageSubresource.mipLevel       = 0;
      region.imageOffset                     = {0, 0, 0};
      region.imageExtent                     = {render_width, render_height, 1};

      vkCmdCopyImageToBuffer(cmdBuffer,                             // Command buffer
                             imageAO.image,                         // Source image
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,  // Source image layout
                             bufferStaging.buffer,                  // Destination buffer
                             1, &region);                           // Regions
    }

    // At the end of the command buffer, add a pipeline barrier to ensure that
    // writes to memory are visible by the CPU:
    VkMemoryBarrier toCPUBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    toCPUBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;    // Make shader writes
    toCPUBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;       // Readable by the CPU
    vkCmdPipelineBarrier(cmdBuffer,                             // The command buffer
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // From the compute shader
                         VK_PIPELINE_STAGE_HOST_BIT,            // To the CPU
                         0,                                     // No special flags
                         1,
                         &toCPUBarrier,  // An array of memory barriers
                         0, nullptr, 0,
                         nullptr);  // No other barriers
  }

  // End and submit the command buffer, then wait for it to finish:
  EndSubmitWaitAndFreeCommandBuffer(device, queueGCT, cmdPool, cmdBuffer);


  // Get the image data back from the GPU
  void*             data    = allocatorDma.map(bufferStaging.allocation);
  const std::string outPath = NVPSystem::exePath() + "out.png";
  // Write the linear data to an 8-bit PNG, without converting linear to sRGB.
  // (So this will appear a bit darker than expected.)
  stbi_write_png(outPath.c_str(), render_width, render_height, 1, data, sizeof(uint8_t) * render_width);
  // Unmap the staging buffer
  allocatorDma.unmap(bufferStaging.allocation);


  // Don't forget to clean up at the end of the program!
  allocator.destroy(bufferSBT);
  vkDestroyPipeline(device, pipelineRayTrace, nullptr);
  vkDestroyShaderModule(device, aoCHitPrimary, nullptr);
  vkDestroyShaderModule(device, aoRMiss, nullptr);
  vkDestroyShaderModule(device, aoRgen, nullptr);
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
  vkDestroyAccelerationStructureKHR(device, tlas, nullptr);
  allocator.destroy(bufferTLAS);
  vkDestroyAccelerationStructureKHR(device, blas, nullptr);
  allocator.destroy(bufferBLAS);

  allocator.destroy(bufferVertex);
  allocator.destroy(bufferIndex);
  vkDestroyCommandPool(device, cmdPool, nullptr);
  allocator.destroy(bufferStaging);
  vkDestroyImageView(device, imageAOView, nullptr);
  allocator.destroy(imageAO);
  allocator.deinit();
  allocatorDma.deinit();

  vkDestroyDevice(device, nullptr);
  destroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
  vkDestroyInstance(instance, nullptr);
}