#pragma once

struct ImDrawData;

#include <vulkan/vulkan_core.h>

namespace ImGui
{
  void InitVK(VkDevice device, VkPhysicalDevice physicalDevice, VkQueue queue, uint32_t queueFamilyIndex, VkRenderPass pass, int subPassIndex = 0);
  void ReInitPipelinesVK(VkRenderPass pass, int subPassIndex = 0);

  void ShutdownVK();
  void RenderDrawDataVK(VkCommandBuffer cmd, const ImDrawData* drawData);
}
