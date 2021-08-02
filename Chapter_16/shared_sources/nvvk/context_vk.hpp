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

#ifndef NV_VK_DEVICEINSTANCE_INCLUDED
#define NV_VK_DEVICEINSTANCE_INCLUDED

#include <vector>
#include <string.h>  // memcpy
#include <unordered_set>
#include <vulkan/vulkan_core.h>

static_assert(VK_HEADER_VERSION >= 131, "Vulkan SDK version needs to be 1.2.131.1 or greater");

namespace nvvk {
/**
To run a Vulkan application, you need to create the Vulkan instance and device.
This is done using the `nvvk::Context`, which wraps the creation of `VkInstance`
and `VkDevice`.

First, any application needs to specify how instance and device should be created:
Version, layers, instance and device extensions influence the features available.
This is done through a temporary and intermediate class that will allow you to gather
all the required conditions for the device creation.
*/

//////////////////////////////////////////////////////////////////////////
/**
# struct ContextCreateInfo

This structure allows the application to specify a set of features
that are expected for the creation of
- VkInstance
- VkDevice

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


*/
struct ContextCreateInfo
{
  ContextCreateInfo(bool bUseValidation = true);

  void setVersion(int major, int minor);

  void addInstanceExtension(const char* name, bool optional = false);
  void addInstanceLayer(const char* name, bool optional = false);
  // version = 0: don't care, otherwise check against equality (useful for provisional exts)
  void addDeviceExtension(const char* name, bool optional = false, void* pFeatureStruct = nullptr, uint32_t version = 0);

  void removeInstanceExtension(const char* name);
  void removeInstanceLayer(const char* name);
  void removeDeviceExtension(const char* name);


  // Configure additional device creation with these variables and functions

  // use device groups
  bool useDeviceGroups = false;

  // which compatible device or device group to pick
  // only used by All-in-one Context::init(...)
  uint32_t compatibleDeviceIndex = 0;

  // instance properties
  const char* appEngine = "nvpro-sample";
  const char* appTitle  = "nvpro-sample";

  // may impact performance hence disable by default
  bool disableRobustBufferAccess = true;

  // Information printed at Context::init time
  bool verboseCompatibleDevices = true;
  bool verboseUsed              = true;  // Print what is used
  bool verboseAvailable         =        // Print what is available
#ifdef _DEBUG
      true;
#else
      false;
#endif

  struct Entry
  {
    Entry(const char* entryName, bool isOptional = false, void* pointerFeatureStruct = nullptr, uint32_t checkVersion = 0)
        : name(entryName)
        , optional(isOptional)
        , pFeatureStruct(pointerFeatureStruct)
        , version(checkVersion)
    {
    }
    const char* name{nullptr};
    bool        optional{false};
    void*       pFeatureStruct{nullptr};
    uint32_t    version{0};
  };

  int apiMajor = 1;
  int apiMinor = 1;

  using EntryArray = std::vector<Entry>;
  EntryArray instanceLayers;
  EntryArray instanceExtensions;
  EntryArray deviceExtensions;
  void*      deviceCreateInfoExt = nullptr;
  void*      instanceCreateInfoExt = nullptr;
};

//////////////////////////////////////////////////////////////////////////
/**
# class nvvk::Context

Context class helps creating the Vulkan instance and to choose the logical device for the mandatory extensions. First is to fill the `ContextCreateInfo` structure, then call:

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
 * `Queue m_queueGCT` : Graphics/Compute/Transfer Queue + family index
 * `Queue m_queueT` : async Transfer Queue + family index
 * `Queue m_queueC` : Compute Queue + family index
* maintains what extensions are finally available
* implicitly hooks up the debug callback

## Choosing the device
When there are multiple devices, the `init` method is choosing the first compatible device available, but it is also possible the choose another one.
~~~ C++
  vkctx.initInstance(deviceInfo); 
  // Find all compatible devices
  auto compatibleDevices = vkctx.getCompatibleDevices(deviceInfo);
  assert(!compatibleDevices.empty());

  // Use first compatible device
  vkctx.initDevice(compatibleDevices[0], deviceInfo);
~~~

## Multi-GPU

When multiple graphic cards should be used as a single device, the `ContextCreateInfo::useDeviceGroups` need to be set to `true`.
The above methods will transparently create the `VkDevice` using `VkDeviceGroupDeviceCreateInfo`.
Especially in the context of NVLink connected cards this is useful.


*/
class Context
{
public:
  Context(Context const&) = delete;
  Context& operator=(Context const&) = delete;

  Context() = default;

  using NameArray = std::vector<const char*>;

  // Vulkan == 1.1 used individual structs
  // Vulkan >= 1.2  have per-version structs
  struct Features11Old
  {
    VkPhysicalDeviceMultiviewFeatures    multiview{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES};
    VkPhysicalDevice16BitStorageFeatures t16BitStorage{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    VkPhysicalDeviceSamplerYcbcrConversionFeatures samplerYcbcrConversion{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES};
    VkPhysicalDeviceProtectedMemoryFeatures protectedMemory{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES};
    VkPhysicalDeviceShaderDrawParameterFeatures drawParameters{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETER_FEATURES};
    VkPhysicalDeviceVariablePointerFeatures variablePointers{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTER_FEATURES};

    Features11Old()
    {
      multiview.pNext              = &t16BitStorage;
      t16BitStorage.pNext          = &samplerYcbcrConversion;
      samplerYcbcrConversion.pNext = &protectedMemory;
      protectedMemory.pNext        = &drawParameters;
      drawParameters.pNext         = &variablePointers;
      variablePointers.pNext       = nullptr;
    }

    void read(const VkPhysicalDeviceVulkan11Features& features11)
    {
      multiview.multiview                              = features11.multiview;
      multiview.multiviewGeometryShader                = features11.multiviewGeometryShader;
      multiview.multiviewTessellationShader            = features11.multiviewTessellationShader;
      t16BitStorage.storageBuffer16BitAccess           = features11.storageBuffer16BitAccess;
      t16BitStorage.storageInputOutput16               = features11.storageInputOutput16;
      t16BitStorage.storagePushConstant16              = features11.storagePushConstant16;
      t16BitStorage.uniformAndStorageBuffer16BitAccess = features11.uniformAndStorageBuffer16BitAccess;
      samplerYcbcrConversion.samplerYcbcrConversion    = features11.samplerYcbcrConversion;
      protectedMemory.protectedMemory                  = features11.protectedMemory;
      drawParameters.shaderDrawParameters              = features11.shaderDrawParameters;
      variablePointers.variablePointers                = features11.variablePointers;
      variablePointers.variablePointersStorageBuffer   = features11.variablePointersStorageBuffer;
    }

    void write(VkPhysicalDeviceVulkan11Features& features11)
    {
      features11.multiview                          = multiview.multiview;
      features11.multiviewGeometryShader            = multiview.multiviewGeometryShader;
      features11.multiviewTessellationShader        = multiview.multiviewTessellationShader;
      features11.storageBuffer16BitAccess           = t16BitStorage.storageBuffer16BitAccess;
      features11.storageInputOutput16               = t16BitStorage.storageInputOutput16;
      features11.storagePushConstant16              = t16BitStorage.storagePushConstant16;
      features11.uniformAndStorageBuffer16BitAccess = t16BitStorage.uniformAndStorageBuffer16BitAccess;
      features11.samplerYcbcrConversion             = samplerYcbcrConversion.samplerYcbcrConversion;
      features11.protectedMemory                    = protectedMemory.protectedMemory;
      features11.shaderDrawParameters               = drawParameters.shaderDrawParameters;
      features11.variablePointers                   = variablePointers.variablePointers;
      features11.variablePointersStorageBuffer      = variablePointers.variablePointersStorageBuffer;
    }
  };
  struct Properties11Old
  {
    VkPhysicalDeviceMaintenance3Properties maintenance3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
    VkPhysicalDeviceIDProperties           deviceID{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceMultiviewProperties    multiview{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES};
    VkPhysicalDeviceProtectedMemoryProperties protectedMemory{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_PROPERTIES};
    VkPhysicalDevicePointClippingProperties pointClipping{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_POINT_CLIPPING_PROPERTIES};
    VkPhysicalDeviceSubgroupProperties      subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};

    Properties11Old()
    {
      maintenance3.pNext    = &deviceID;
      deviceID.pNext        = &multiview;
      multiview.pNext       = &protectedMemory;
      protectedMemory.pNext = &pointClipping;
      pointClipping.pNext   = &subgroup;
      subgroup.pNext        = nullptr;
    }

    void write(VkPhysicalDeviceVulkan11Properties& properties11)
    {
      memcpy(properties11.deviceLUID, deviceID.deviceLUID, sizeof(properties11.deviceLUID));
      memcpy(properties11.deviceUUID, deviceID.deviceUUID, sizeof(properties11.deviceUUID));
      memcpy(properties11.driverUUID, deviceID.driverUUID, sizeof(properties11.driverUUID));
      properties11.deviceLUIDValid                   = deviceID.deviceLUIDValid;
      properties11.deviceNodeMask                    = deviceID.deviceNodeMask;
      properties11.subgroupSize                      = subgroup.subgroupSize;
      properties11.subgroupSupportedStages           = subgroup.supportedStages;
      properties11.subgroupSupportedOperations       = subgroup.supportedOperations;
      properties11.subgroupQuadOperationsInAllStages = subgroup.quadOperationsInAllStages;
      properties11.pointClippingBehavior             = pointClipping.pointClippingBehavior;
      properties11.maxMultiviewViewCount             = multiview.maxMultiviewViewCount;
      properties11.maxMultiviewInstanceIndex         = multiview.maxMultiviewInstanceIndex;
      properties11.protectedNoFault                  = protectedMemory.protectedNoFault;
      properties11.maxPerSetDescriptors              = maintenance3.maxPerSetDescriptors;
      properties11.maxMemoryAllocationSize           = maintenance3.maxMemoryAllocationSize;
    }
  };

  // This struct holds all core feature information for a physical device
  struct PhysicalDeviceInfo
  {
    VkPhysicalDeviceMemoryProperties     memoryProperties{};
    std::vector<VkQueueFamilyProperties> queueProperties;

    VkPhysicalDeviceFeatures         features10{};
    VkPhysicalDeviceVulkan11Features features11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};

    VkPhysicalDeviceProperties         properties10{};
    VkPhysicalDeviceVulkan11Properties properties11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    VkPhysicalDeviceVulkan12Properties properties12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
  };

  struct Queue
  {
    VkQueue  queue       = VK_NULL_HANDLE;
    uint32_t familyIndex = ~0;
    uint32_t queueIndex  = ~0;

    operator VkQueue() const { return queue; }
    operator uint32_t() const { return familyIndex; }
  };


  VkInstance         m_instance{VK_NULL_HANDLE};
  VkDevice           m_device{VK_NULL_HANDLE};
  VkPhysicalDevice   m_physicalDevice{VK_NULL_HANDLE};
  PhysicalDeviceInfo m_physicalInfo;

  // All the queues (if present) is distinct from each other
  Queue m_queueGCT;  // for Graphics/Compute/Transfer (must exist)
  Queue m_queueT;    // for pure async Transfer Queue (can exist, supports at least transfer)
  Queue m_queueC;    // for async Compute (can exist, supports at least compute)

  operator VkDevice() const { return m_device; }

  // All-in-one instance and device creation
  bool init(const ContextCreateInfo& info);
  void deinit();

  // Individual object creation
  bool initInstance(const ContextCreateInfo& info);
  // deviceIndex is an index either into getPhysicalDevices or getPhysicalDeviceGroups
  // depending on info.useDeviceGroups
  bool initDevice(uint32_t deviceIndex, const ContextCreateInfo& info);

  // Helpers
  std::vector<int>                             getCompatibleDevices(const ContextCreateInfo& info);
  std::vector<VkPhysicalDevice>                getPhysicalDevices();
  std::vector<VkPhysicalDeviceGroupProperties> getPhysicalDeviceGroups();
  std::vector<VkExtensionProperties>           getInstanceExtensions();
  std::vector<VkLayerProperties>               getInstanceLayers();
  std::vector<VkExtensionProperties>           getDeviceExtensions(VkPhysicalDevice physicalDevice);
  bool hasMandatoryExtensions(VkPhysicalDevice physicalDevice, const ContextCreateInfo& info, bool bVerbose);

  // Ensures the GCT queue can present to the provided surface (return false if fails to set)
  bool setGCTQueueWithPresent(VkSurfaceKHR surface);

  uint32_t getQueueFamily(VkQueueFlags flagsSupported, VkQueueFlags flagsDisabled = 0, VkSurfaceKHR surface = VK_NULL_HANDLE);

  // true if the context has the optional extension activated
  bool hasDeviceExtension(const char* name) const;
  bool hasInstanceExtension(const char* name) const;

  void ignoreDebugMessage(int32_t msgID)
  {
    m_dbgIgnoreMessages.insert(msgID);
  }

private:
  static VKAPI_ATTR VkBool32 VKAPI_CALL debugMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
                                                      VkDebugUtilsMessageTypeFlagsEXT             messageType,
                                                      const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
                                                      void*                                       userData);

  NameArray m_usedInstanceLayers;
  NameArray m_usedInstanceExtensions;
  NameArray m_usedDeviceExtensions;

  // New Debug system
  PFN_vkCreateDebugUtilsMessengerEXT  m_createDebugUtilsMessengerEXT  = nullptr;
  PFN_vkDestroyDebugUtilsMessengerEXT m_destroyDebugUtilsMessengerEXT = nullptr;
  VkDebugUtilsMessengerEXT            m_dbgMessenger                  = nullptr;

  std::unordered_set<int32_t>         m_dbgIgnoreMessages;

  void initDebugUtils();

  VkResult    fillFilteredNameArray(Context::NameArray&                   used,
                                    const std::vector<VkLayerProperties>& properties,
                                    const ContextCreateInfo::EntryArray&  requested);
  VkResult    fillFilteredNameArray(Context::NameArray&                       used,
                                    const std::vector<VkExtensionProperties>& properties,
                                    const ContextCreateInfo::EntryArray&      requested,
                                    std::vector<void*>&                       featureStructs);
  bool        checkEntryArray(const std::vector<VkExtensionProperties>& properties, const ContextCreateInfo::EntryArray& requested, bool bVerbose);
  static void initPhysicalInfo(PhysicalDeviceInfo& info, VkPhysicalDevice physicalDevice, uint32_t versionMajor, uint32_t versionMinor);
};


}  // namespace nvvk

#endif
