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

#include <cassert>
#include <iterator>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace nvvk {
//--------------------------------------------------------------------------------------------------
/** 
  # functions in nvvk

  - nvprintPipelineStats : prints stats of the pipeline using VK_KHR_pipeline_executable_properties (don't forget to enable extension and set VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR)
  - dumpPipelineStats    : dumps stats of the pipeline using VK_KHR_pipeline_executable_properties to a text file (don't forget to enable extension and set VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR)
  - dumpPipelineBinCodes : dumps shader binaries using VK_KHR_pipeline_executable_properties to multiple binary files (don't forget to enable extension and set VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR)
*/
// nvprints stats to LOGLEVEL_STATS stream
void nvprintPipelineStats(VkDevice device, VkPipeline pipeline, const char* name, bool verbose = false);
// writes stats into single file
void dumpPipelineStats(VkDevice device, VkPipeline pipeline, const char* fileName);
// creates multiple files, one for each pipe executable and representation.
// The baseFilename will get appended along the lines of ".some details.bin"
void dumpPipelineInternals(VkDevice device, VkPipeline pipeline, const char* baseFileName);

//--------------------------------------------------------------------------------------------------
/** 
# class nvvk::GraphicsPipelineState

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
*/


struct GraphicsPipelineState
{
  // Initialize the state to common values: triangle list topology, depth test enabled,
  // dynamic viewport and scissor, one render target, blending disabled
  GraphicsPipelineState()
  {
    rasterizationState.flags                   = {};
    rasterizationState.depthClampEnable        = {};
    rasterizationState.rasterizerDiscardEnable = {};
    setValue(rasterizationState.polygonMode, VK_POLYGON_MODE_FILL);
    setValue(rasterizationState.cullMode, VK_CULL_MODE_BACK_BIT);
    setValue(rasterizationState.frontFace, VK_FRONT_FACE_COUNTER_CLOCKWISE);

    rasterizationState.depthBiasEnable         = {};
    rasterizationState.depthBiasConstantFactor = {};
    rasterizationState.depthBiasClamp          = {};
    rasterizationState.depthBiasSlopeFactor    = {};
    rasterizationState.lineWidth               = 1.f;

    inputAssemblyState.flags = {};
    setValue(inputAssemblyState.topology, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    inputAssemblyState.primitiveRestartEnable = {};


    colorBlendState.flags         = {};
    colorBlendState.logicOpEnable = {};
    setValue(colorBlendState.logicOp, VK_LOGIC_OP_CLEAR);
    colorBlendState.attachmentCount = {};
    colorBlendState.pAttachments    = {};
    for(int i = 0; i < 4; i++)
    {
      colorBlendState.blendConstants[i] = 0.f;
    }


    dynamicState.flags             = {};
    dynamicState.dynamicStateCount = {};
    dynamicState.pDynamicStates    = {};


    vertexInputState.flags                           = {};
    vertexInputState.vertexBindingDescriptionCount   = {};
    vertexInputState.pVertexBindingDescriptions      = {};
    vertexInputState.vertexAttributeDescriptionCount = {};
    vertexInputState.pVertexAttributeDescriptions    = {};


    viewportState.flags         = {};
    viewportState.viewportCount = {};
    viewportState.pViewports    = {};
    viewportState.scissorCount  = {};
    viewportState.pScissors     = {};


    depthStencilState.flags            = {};
    depthStencilState.depthTestEnable  = VK_TRUE;
    depthStencilState.depthWriteEnable = VK_TRUE;
    setValue(depthStencilState.depthCompareOp, VK_COMPARE_OP_LESS_OR_EQUAL);
    depthStencilState.depthBoundsTestEnable = {};
    depthStencilState.stencilTestEnable     = {};
    setValue(depthStencilState.front, VkStencilOpState());
    setValue(depthStencilState.back, VkStencilOpState());
    depthStencilState.minDepthBounds = {};
    depthStencilState.maxDepthBounds = {};
  }

  GraphicsPipelineState(const GraphicsPipelineState& src) = default;

  // Attach the pointer values of the structures to the internal arrays
  void update()
  {
    colorBlendState.attachmentCount = (uint32_t)blendAttachmentStates.size();
    colorBlendState.pAttachments    = blendAttachmentStates.data();

    dynamicState.dynamicStateCount = (uint32_t)dynamicStateEnables.size();
    dynamicState.pDynamicStates    = dynamicStateEnables.data();

    vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputState.vertexBindingDescriptionCount   = static_cast<uint32_t>(bindingDescriptions.size());
    vertexInputState.pVertexBindingDescriptions      = bindingDescriptions.data();
    vertexInputState.pVertexAttributeDescriptions    = attributeDescriptions.data();

    if(viewports.empty())
    {
      viewportState.viewportCount = 1;
      viewportState.pViewports    = nullptr;
    }
    else
    {
      viewportState.viewportCount = (uint32_t)viewports.size();
      viewportState.pViewports    = viewports.data();
    }

    if(scissors.empty())
    {
      viewportState.scissorCount = 1;
      viewportState.pScissors    = nullptr;
    }
    else
    {
      viewportState.scissorCount = (uint32_t)scissors.size();
      viewportState.pScissors    = scissors.data();
    }
  }

#ifdef VULKAN_HPP
  static inline VkPipelineColorBlendAttachmentState makePipelineColorBlendAttachmentState(
      vk::ColorComponentFlags colorWriteMask_ = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
                                                | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
      vk::Bool32      blendEnable_         = 0,
      vk::BlendFactor srcColorBlendFactor_ = vk::BlendFactor::eZero,
      vk::BlendFactor dstColorBlendFactor_ = vk::BlendFactor::eZero,
      vk::BlendOp     colorBlendOp_        = vk::BlendOp::eAdd,
      vk::BlendFactor srcAlphaBlendFactor_ = vk::BlendFactor::eZero,
      vk::BlendFactor dstAlphaBlendFactor_ = vk::BlendFactor::eZero,
      vk::BlendOp     alphaBlendOp_        = vk::BlendOp::eAdd)
  {
    vk::PipelineColorBlendAttachmentState res;
#else
  static inline VkPipelineColorBlendAttachmentState makePipelineColorBlendAttachmentState(
      VkColorComponentFlags colorWriteMask_ = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                              | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
      VkBool32      blendEnable_         = 0,
      VkBlendFactor srcColorBlendFactor_ = VK_BLEND_FACTOR_ZERO,
      VkBlendFactor dstColorBlendFactor_ = VK_BLEND_FACTOR_ZERO,
      VkBlendOp     colorBlendOp_        = VK_BLEND_OP_ADD,
      VkBlendFactor srcAlphaBlendFactor_ = VK_BLEND_FACTOR_ZERO,
      VkBlendFactor dstAlphaBlendFactor_ = VK_BLEND_FACTOR_ZERO,
      VkBlendOp     alphaBlendOp_        = VK_BLEND_OP_ADD)
  {
    VkPipelineColorBlendAttachmentState res;
#endif

    res.blendEnable         = blendEnable_;
    res.srcColorBlendFactor = srcColorBlendFactor_;
    res.dstColorBlendFactor = dstColorBlendFactor_;
    res.colorBlendOp        = colorBlendOp_;
    res.srcAlphaBlendFactor = srcAlphaBlendFactor_;
    res.dstAlphaBlendFactor = dstAlphaBlendFactor_;
    res.alphaBlendOp        = alphaBlendOp_;
    res.colorWriteMask      = colorWriteMask_;
    return res;
  }

  static inline VkVertexInputBindingDescription makeVertexInputBinding(uint32_t binding, uint32_t stride, VkVertexInputRate rate = VK_VERTEX_INPUT_RATE_VERTEX)
  {
    VkVertexInputBindingDescription vertexBinding;
    vertexBinding.binding   = binding;
    vertexBinding.inputRate = rate;
    vertexBinding.stride    = stride;
    return vertexBinding;
  }

  static inline VkVertexInputAttributeDescription makeVertexInputAttribute(uint32_t location, uint32_t binding, VkFormat format, uint32_t offset)
  {
    VkVertexInputAttributeDescription attrib;
    attrib.binding  = binding;
    attrib.location = location;
    attrib.format   = format;
    attrib.offset   = offset;
    return attrib;
  }


  void clearBlendAttachmentStates() { blendAttachmentStates.clear(); }
  void setBlendAttachmentCount(uint32_t attachmentCount) { blendAttachmentStates.resize(attachmentCount); }
  void setBlendAttachmentState(uint32_t attachment, VkPipelineColorBlendAttachmentState blendState)
  {
    assert(attachment < blendAttachmentStates.size());
    if(attachment <= blendAttachmentStates.size())
    {
      blendAttachmentStates[attachment] = blendState;
    }
  }
  uint32_t addBlendAttachmentState(VkPipelineColorBlendAttachmentState blendState)
  {
    blendAttachmentStates.push_back(blendState);
    return (uint32_t)(blendAttachmentStates.size() - 1);
  }


  void clearDynamicStateEnables() { dynamicStateEnables.clear(); }
  void setDynamicStateEnablesCount(uint32_t dynamicStateCount) { dynamicStateEnables.resize(dynamicStateCount); }
#ifdef VULKAN_HPP
  void setDynamicStateEnable(uint32_t state, vk::DynamicState dynamicState)
#else
  void     setDynamicStateEnable(uint32_t state, VkDynamicState dynamicState)
#endif
  {
    assert(state < dynamicStateEnables.size());
    if(state <= dynamicStateEnables.size())
    {
      dynamicStateEnables[state] = dynamicState;
    }
  }

#ifdef VULKAN_HPP
  uint32_t addDynamicStateEnable(vk::DynamicState dynamicState)
#else
  uint32_t addDynamicStateEnable(VkDynamicState dynamicState)
#endif
  {
    dynamicStateEnables.push_back(dynamicState);
    return (uint32_t)(dynamicStateEnables.size() - 1);
  }


  void clearBindingDescriptions() { bindingDescriptions.clear(); }
  void setBindingDescriptionsCount(uint32_t bindingDescriptionCount)
  {
    bindingDescriptions.resize(bindingDescriptionCount);
  }
  void setBindingDescription(uint32_t binding, VkVertexInputBindingDescription bindingDescription)
  {
    assert(binding < bindingDescriptions.size());
    if(binding <= bindingDescriptions.size())
    {
      bindingDescriptions[binding] = bindingDescription;
    }
  }
  uint32_t addBindingDescription(VkVertexInputBindingDescription bindingDescription)
  {
    bindingDescriptions.push_back(bindingDescription);
    return (uint32_t)(bindingDescriptions.size() - 1);
  }

#ifdef VULKAN_HPP
  void addBindingDescriptions(const std::vector<vk::VertexInputBindingDescription>& bindingDescriptions_)
#else
  void     addBindingDescriptions(const std::vector<VkVertexInputBindingDescription>& bindingDescriptions_)
#endif
  {
    bindingDescriptions.insert(bindingDescriptions.end(), bindingDescriptions_.begin(), bindingDescriptions_.end());
  }

  void clearAttributeDescriptions() { attributeDescriptions.clear(); }
  void setAttributeDescriptionsCount(uint32_t attributeDescriptionCount)
  {
    attributeDescriptions.resize(attributeDescriptionCount);
  }
  void setAttributeDescription(uint32_t attribute, VkVertexInputAttributeDescription attributeDescription)
  {
    assert(attribute < attributeDescriptions.size());
    if(attribute <= attributeDescriptions.size())
    {
      attributeDescriptions[attribute] = attributeDescription;
    }
  }
  uint32_t addAttributeDescription(VkVertexInputAttributeDescription attributeDescription)
  {
    attributeDescriptions.push_back(attributeDescription);
    return (uint32_t)(attributeDescriptions.size() - 1);
  }

#ifdef VULKAN_HPP
  void addAttributeDescriptions(const std::vector<vk::VertexInputAttributeDescription>& attributeDescriptions_)
#else
  void     addAttributeDescriptions(const std::vector<VkVertexInputAttributeDescription>& attributeDescriptions_)
#endif
  {
    attributeDescriptions.insert(attributeDescriptions.end(), attributeDescriptions_.begin(), attributeDescriptions_.end());
  }


  void clearViewports() { viewports.clear(); }
  void setViewportsCount(uint32_t viewportCount) { viewports.resize(viewportCount); }
  void setViewport(uint32_t attribute, VkViewport viewport)
  {
    assert(attribute < viewports.size());
    if(attribute <= viewports.size())
    {
      viewports[attribute] = viewport;
    }
  }
  uint32_t addViewport(VkViewport viewport)
  {
    viewports.push_back(viewport);
    return (uint32_t)(viewports.size() - 1);
  }


  void clearScissors() { scissors.clear(); }
  void setScissorsCount(uint32_t scissorCount) { scissors.resize(scissorCount); }
  void setScissor(uint32_t attribute, VkRect2D scissor)
  {
    assert(attribute < scissors.size());
    if(attribute <= scissors.size())
    {
      scissors[attribute] = scissor;
    }
  }
  uint32_t addScissor(VkRect2D scissor)
  {
    scissors.push_back(scissor);
    return (uint32_t)(scissors.size() - 1);
  }


#ifdef VULKAN_HPP
  vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState;
  vk::PipelineRasterizationStateCreateInfo rasterizationState;
  vk::PipelineMultisampleStateCreateInfo   multisampleState;
  vk::PipelineDepthStencilStateCreateInfo  depthStencilState;
  vk::PipelineViewportStateCreateInfo      viewportState;
  vk::PipelineDynamicStateCreateInfo       dynamicState;
  vk::PipelineColorBlendStateCreateInfo    colorBlendState;
  vk::PipelineVertexInputStateCreateInfo   vertexInputState;

private:
  std::vector<vk::PipelineColorBlendAttachmentState> blendAttachmentStates{makePipelineColorBlendAttachmentState()};
  std::vector<vk::DynamicState> dynamicStateEnables = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

  std::vector<vk::VertexInputBindingDescription>   bindingDescriptions;
  std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;

  std::vector<vk::Viewport> viewports;
  std::vector<vk::Rect2D>   scissors;


#else
  VkPipelineInputAssemblyStateCreateInfo inputAssemblyState{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  VkPipelineRasterizationStateCreateInfo rasterizationState{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  VkPipelineMultisampleStateCreateInfo   multisampleState{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  VkPipelineDepthStencilStateCreateInfo  depthStencilState{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  VkPipelineViewportStateCreateInfo      viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  VkPipelineDynamicStateCreateInfo       dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  VkPipelineColorBlendStateCreateInfo    colorBlendState{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  VkPipelineVertexInputStateCreateInfo   vertexInputState{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

private:
  std::vector<VkPipelineColorBlendAttachmentState> blendAttachmentStates{makePipelineColorBlendAttachmentState()};
  std::vector<VkDynamicState> dynamicStateEnables = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

  std::vector<VkVertexInputBindingDescription>   bindingDescriptions;
  std::vector<VkVertexInputAttributeDescription> attributeDescriptions;

  std::vector<VkViewport>          viewports;
  std::vector<VkRect2D>            scissors;

#endif

  // Helper to set objects for either C and C++
  template <class T, class U>
  void setValue(T& target, const U& val)
  {
    target = (T)(val);
  }
};


//--------------------------------------------------------------------------------------------------
/** 
# class nvvk::GraphicsPipelineGenerator

The graphics pipeline generator takes a GraphicsPipelineState object and pipeline-specific information such as 
the render pass and pipeline layout to generate the final pipeline. 

This structure is instantiated using C++ Vulkan objects if VULKAN_HPP is defined, and C otherwise.

Example of usage :
~~~~ c++
nvvk::GraphicsPipelineState pipelineState();
...
nvvk::GraphicsPipelineGenerator pipelineGenerator(m_device, m_pipelineLayout, m_renderPass, pipelineState);
pipelineGenerator.addShader(readFile("shaders/vert_shader.vert.spv"), VkShaderStageFlagBits::eVertex);
pipelineGenerator.addShader(readFile("shaders/frag_shader.frag.spv"), VkShaderStageFlagBits::eFragment);

m_pipeline = pipelineGenerator.createPipeline();
~~~~
*/

struct GraphicsPipelineGenerator
{
public:
  GraphicsPipelineGenerator(GraphicsPipelineState& pipelineState_)
      : pipelineState(pipelineState_)
  {
    init();
  }

  GraphicsPipelineGenerator(const GraphicsPipelineGenerator& src)
      : createInfo(src.createInfo)
      , device(src.device)
      , pipelineCache(src.pipelineCache)
      , pipelineState(src.pipelineState)
  {
    init();
  }

  GraphicsPipelineGenerator(VkDevice device_, const VkPipelineLayout& layout, const VkRenderPass& renderPass, GraphicsPipelineState& pipelineState_)
      : device(device_)
      , pipelineState(pipelineState_)
  {
    createInfo.layout     = layout;
    createInfo.renderPass = renderPass;
    init();
  }

  const GraphicsPipelineGenerator& operator=(const GraphicsPipelineGenerator& src)
  {
    device        = src.device;
    pipelineState = src.pipelineState;
    createInfo    = src.createInfo;
    pipelineCache = src.pipelineCache;

    init();
    return *this;
  }

  void setDevice(VkDevice device_) { device = device_; }

  void setRenderPass(VkRenderPass renderPass) { createInfo.renderPass = renderPass; }

  void setLayout(VkPipelineLayout layout) { createInfo.layout = layout; }

  ~GraphicsPipelineGenerator() { destroyShaderModules(); }

#ifdef VULKAN_HPP
  vk::PipelineShaderStageCreateInfo& addShader(const std::string&      code,
                                               vk::ShaderStageFlagBits stage,
                                               const char*             entryPoint = "main")
#else
  VkPipelineShaderStageCreateInfo& addShader(const std::string&    code,
                                             VkShaderStageFlagBits stage,
                                             const char*           entryPoint = "main")
#endif
  {
    std::vector<char> v;
    std::copy(code.begin(), code.end(), std::back_inserter(v));
    return addShader(v, stage, entryPoint);
  }

#ifdef VULKAN_HPP
  template <typename T>
  vk::PipelineShaderStageCreateInfo& addShader(const std::vector<T>&   code,
                                               vk::ShaderStageFlagBits stage,
                                               const char*             entryPoint = "main")
#else
  template <typename T>
  VkPipelineShaderStageCreateInfo& addShader(const std::vector<T>& code,
                                             VkShaderStageFlagBits stage,
                                             const char*           entryPoint = "main")
#endif

  {
    VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    createInfo.codeSize = sizeof(T) * code.size();
    createInfo.pCode    = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule shaderModule;
    vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
    temporaryModules.push_back(shaderModule);

    return addShader(shaderModule, stage, entryPoint);
  }
#ifdef VULKAN_HPP
  vk::PipelineShaderStageCreateInfo& addShader(vk::ShaderModule        shaderModule,
                                               vk::ShaderStageFlagBits stage,
                                               const char*             entryPoint = "main")
#else
  VkPipelineShaderStageCreateInfo& addShader(VkShaderModule        shaderModule,
                                             VkShaderStageFlagBits stage,
                                             const char*           entryPoint = "main")
#endif
  {
    VkPipelineShaderStageCreateInfo shaderStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    shaderStage.stage  = (VkShaderStageFlagBits)stage;
    shaderStage.module = shaderModule;
    shaderStage.pName  = entryPoint;

    shaderStages.push_back(shaderStage);
    return shaderStages.back();
  }

  void clearShaders()
  {
    shaderStages.clear();
    destroyShaderModules();
  }

  VkShaderModule getShaderModule(size_t index) const
  {
    if(index < shaderStages.size())
      return shaderStages[index].module;
    return VK_NULL_HANDLE;
  }

  VkPipeline createPipeline(const VkPipelineCache& cache)
  {
    update();
    VkPipeline pipeline;
    vkCreateGraphicsPipelines(device, cache, 1, (VkGraphicsPipelineCreateInfo*)&createInfo, nullptr, &pipeline);
    return pipeline;
  }

  VkPipeline createPipeline() { return createPipeline(pipelineCache); }

  void destroyShaderModules()
  {
    for(const auto& shaderModule : temporaryModules)
    {
      vkDestroyShaderModule(device, shaderModule, nullptr);
    }
    temporaryModules.clear();
  }
  void update()
  {
    createInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    createInfo.pStages    = shaderStages.data();
    pipelineState.update();
  }

#ifdef VULKAN_HPP
  vk::GraphicsPipelineCreateInfo createInfo;
#else
  VkGraphicsPipelineCreateInfo     createInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
#endif


private:
#ifdef VULKAN_HPP
  vk::Device        device;
  vk::PipelineCache pipelineCache;

  std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;
  std::vector<vk::ShaderModule>                  temporaryModules;

#else
  VkDevice                         device;
  VkPipelineCache                  pipelineCache{};

  std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
  std::vector<VkShaderModule>                  temporaryModules;
#endif
  GraphicsPipelineState& pipelineState;


  void init()
  {
    setValue(pipelineState.multisampleState.rasterizationSamples, VK_SAMPLE_COUNT_1_BIT);
    createInfo.pRasterizationState = &pipelineState.rasterizationState;
    createInfo.pInputAssemblyState = &pipelineState.inputAssemblyState;
    createInfo.pColorBlendState    = &pipelineState.colorBlendState;
    createInfo.pMultisampleState   = &pipelineState.multisampleState;
    createInfo.pViewportState      = &pipelineState.viewportState;
    createInfo.pDepthStencilState  = &pipelineState.depthStencilState;
    createInfo.pDynamicState       = &pipelineState.dynamicState;
    createInfo.pVertexInputState   = &pipelineState.vertexInputState;
  }

  // Helper to set objects for either C and C++
  template <class T, class U>
  void setValue(T& target, const U& val)
  {
    target = (T)(val);
  }
};


//--------------------------------------------------------------------------------------------------
/** 
# class nvvk::GraphicsPipelineGeneratorCombined

In some cases the application may have each state associated to a single pipeline. For convenience, 
GraphicsPipelineGeneratorCombined combines both the state and generator into a single object.

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

pipelineGenerator.addShader(readFile("shaders/vert_shader.vert.spv"), VkShaderStageFlagBits::eVertex);
pipelineGenerator.addShader(readFile("shaders/frag_shader.frag.spv"), VkShaderStageFlagBits::eFragment);

m_pipeline = pipelineGenerator.createPipeline();
~~~~
*/


struct GraphicsPipelineGeneratorCombined : public GraphicsPipelineState, public GraphicsPipelineGenerator
{
  GraphicsPipelineGeneratorCombined(VkDevice device_, const VkPipelineLayout& layout, const VkRenderPass& renderPass)
      : GraphicsPipelineState()
      , GraphicsPipelineGenerator(device_, layout, renderPass, *this)
  {
  }
};
}  // namespace nvvk
