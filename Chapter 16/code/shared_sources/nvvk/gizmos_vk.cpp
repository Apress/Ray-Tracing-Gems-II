/* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "gizmos_vk.hpp"

namespace nvvk {


// glsl_shader.vert, compiled with:
// # glslangValidator -o axis.vert.h -V axis.vert -vn s_vert_spv
/*************************************************
#version 450 core

layout(push_constant) uniform uPushConstant
{
  mat4 transform;
}
pc;

out gl_PerVertex
{
  vec4 gl_Position;
};

layout(location = 0) out interpolant
{
  vec4 Color;
} Out;

const float asize = 1.0f;
const float atip  = 0.2f;
const float abase = 0.75f;

vec3 arrow_vert[7];

void main()
{
  arrow_vert[0] = vec3(asize, 0, 0);
  arrow_vert[1] = vec3(abase, atip, 0);
  arrow_vert[2] = vec3(abase, -atip * 0.5f, -atip * 0.87f);
  arrow_vert[3] = vec3(abase, -atip * 0.5f, atip * 0.87f);
  arrow_vert[4] = vec3(abase, atip, 0);
  arrow_vert[5] = vec3(asize, 0, 0);  // To draw the line
  arrow_vert[6] = vec3(0, 0, 0);      // ...

 vec3 pos = arrow_vert[gl_VertexIndex];
  //Out.Color = aColor;
  if(gl_InstanceIndex == 0)
  {
    Out.Color   = vec4(1, 0, 0, 1);
    gl_Position = pc.transform * vec4(pos.xyz, 1);
  }
  else if(gl_InstanceIndex == 1)
  {
    Out.Color   = vec4(0, 1, 0, 1);
    gl_Position = pc.transform * vec4(pos.yxz, 1);
  }
  else
  {
    Out.Color   = vec4(0, 0, 1, 1);
    gl_Position = pc.transform * vec4(pos.yzx, 1);
  }
}
 
*********************/


// glsl_shader.frag, compiled with:
// # glslangValidator -V -x -o glsl_shader.frag.u32 glsl_shader.frag
/*************************************************
#version 450 core
layout(location = 0) out vec4 fColor;


layout(location = 0) in interpolant
{
  vec4 Color;
} In;

void main()
{
  fColor = In.Color;
}


*********************/

static std::vector<uint32_t> s_axis_vert_spv = {
    0x07230203, 0x00010000, 0x00080007, 0x0000006c, 0x00000000, 0x00020011, 0x00000001, 0x0006000b, 0x00000001,
    0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0009000f, 0x00000000,
    0x00000004, 0x6e69616d, 0x00000000, 0x0000002c, 0x00000030, 0x00000039, 0x0000003f, 0x00030003, 0x00000002,
    0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00050005, 0x0000000c, 0x6f727261, 0x65765f77,
    0x00007472, 0x00030005, 0x0000002a, 0x00736f70, 0x00060005, 0x0000002c, 0x565f6c67, 0x65747265, 0x646e4978,
    0x00007865, 0x00070005, 0x00000030, 0x495f6c67, 0x6174736e, 0x4965636e, 0x7865646e, 0x00000000, 0x00030005,
    0x00000037, 0x00000000, 0x00050006, 0x00000037, 0x00000000, 0x6f6c6f43, 0x00000072, 0x00030005, 0x00000039,
    0x0074754f, 0x00060005, 0x0000003d, 0x505f6c67, 0x65567265, 0x78657472, 0x00000000, 0x00060006, 0x0000003d,
    0x00000000, 0x505f6c67, 0x7469736f, 0x006e6f69, 0x00030005, 0x0000003f, 0x00000000, 0x00060005, 0x00000041,
    0x73755075, 0x6e6f4368, 0x6e617473, 0x00000074, 0x00060006, 0x00000041, 0x00000000, 0x6e617274, 0x726f6673,
    0x0000006d, 0x00030005, 0x00000043, 0x00006370, 0x00040047, 0x0000002c, 0x0000000b, 0x0000002a, 0x00040047,
    0x00000030, 0x0000000b, 0x0000002b, 0x00040047, 0x00000039, 0x0000001e, 0x00000000, 0x00050048, 0x0000003d,
    0x00000000, 0x0000000b, 0x00000000, 0x00030047, 0x0000003d, 0x00000002, 0x00040048, 0x00000041, 0x00000000,
    0x00000005, 0x00050048, 0x00000041, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000041, 0x00000000,
    0x00000007, 0x00000010, 0x00030047, 0x00000041, 0x00000002, 0x00020013, 0x00000002, 0x00030021, 0x00000003,
    0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000003, 0x00040015,
    0x00000008, 0x00000020, 0x00000000, 0x0004002b, 0x00000008, 0x00000009, 0x00000007, 0x0004001c, 0x0000000a,
    0x00000007, 0x00000009, 0x00040020, 0x0000000b, 0x00000006, 0x0000000a, 0x0004003b, 0x0000000b, 0x0000000c,
    0x00000006, 0x00040015, 0x0000000d, 0x00000020, 0x00000001, 0x0004002b, 0x0000000d, 0x0000000e, 0x00000000,
    0x0004002b, 0x00000006, 0x0000000f, 0x3f800000, 0x0004002b, 0x00000006, 0x00000010, 0x00000000, 0x0006002c,
    0x00000007, 0x00000011, 0x0000000f, 0x00000010, 0x00000010, 0x00040020, 0x00000012, 0x00000006, 0x00000007,
    0x0004002b, 0x0000000d, 0x00000014, 0x00000001, 0x0004002b, 0x00000006, 0x00000015, 0x3f400000, 0x0004002b,
    0x00000006, 0x00000016, 0x3e4ccccd, 0x0006002c, 0x00000007, 0x00000017, 0x00000015, 0x00000016, 0x00000010,
    0x0004002b, 0x0000000d, 0x00000019, 0x00000002, 0x0004002b, 0x00000006, 0x0000001a, 0xbdcccccd, 0x0004002b,
    0x00000006, 0x0000001b, 0xbe322d0e, 0x0006002c, 0x00000007, 0x0000001c, 0x00000015, 0x0000001a, 0x0000001b,
    0x0004002b, 0x0000000d, 0x0000001e, 0x00000003, 0x0004002b, 0x00000006, 0x0000001f, 0x3e322d0e, 0x0006002c,
    0x00000007, 0x00000020, 0x00000015, 0x0000001a, 0x0000001f, 0x0004002b, 0x0000000d, 0x00000022, 0x00000004,
    0x0004002b, 0x0000000d, 0x00000024, 0x00000005, 0x0004002b, 0x0000000d, 0x00000026, 0x00000006, 0x0006002c,
    0x00000007, 0x00000027, 0x00000010, 0x00000010, 0x00000010, 0x00040020, 0x00000029, 0x00000007, 0x00000007,
    0x00040020, 0x0000002b, 0x00000001, 0x0000000d, 0x0004003b, 0x0000002b, 0x0000002c, 0x00000001, 0x0004003b,
    0x0000002b, 0x00000030, 0x00000001, 0x00020014, 0x00000032, 0x00040017, 0x00000036, 0x00000006, 0x00000004,
    0x0003001e, 0x00000037, 0x00000036, 0x00040020, 0x00000038, 0x00000003, 0x00000037, 0x0004003b, 0x00000038,
    0x00000039, 0x00000003, 0x0007002c, 0x00000036, 0x0000003a, 0x0000000f, 0x00000010, 0x00000010, 0x0000000f,
    0x00040020, 0x0000003b, 0x00000003, 0x00000036, 0x0003001e, 0x0000003d, 0x00000036, 0x00040020, 0x0000003e,
    0x00000003, 0x0000003d, 0x0004003b, 0x0000003e, 0x0000003f, 0x00000003, 0x00040018, 0x00000040, 0x00000036,
    0x00000004, 0x0003001e, 0x00000041, 0x00000040, 0x00040020, 0x00000042, 0x00000009, 0x00000041, 0x0004003b,
    0x00000042, 0x00000043, 0x00000009, 0x00040020, 0x00000044, 0x00000009, 0x00000040, 0x0007002c, 0x00000036,
    0x00000053, 0x00000010, 0x0000000f, 0x00000010, 0x0000000f, 0x0007002c, 0x00000036, 0x00000060, 0x00000010,
    0x00000010, 0x0000000f, 0x0000000f, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8,
    0x00000005, 0x0004003b, 0x00000029, 0x0000002a, 0x00000007, 0x00050041, 0x00000012, 0x00000013, 0x0000000c,
    0x0000000e, 0x0003003e, 0x00000013, 0x00000011, 0x00050041, 0x00000012, 0x00000018, 0x0000000c, 0x00000014,
    0x0003003e, 0x00000018, 0x00000017, 0x00050041, 0x00000012, 0x0000001d, 0x0000000c, 0x00000019, 0x0003003e,
    0x0000001d, 0x0000001c, 0x00050041, 0x00000012, 0x00000021, 0x0000000c, 0x0000001e, 0x0003003e, 0x00000021,
    0x00000020, 0x00050041, 0x00000012, 0x00000023, 0x0000000c, 0x00000022, 0x0003003e, 0x00000023, 0x00000017,
    0x00050041, 0x00000012, 0x00000025, 0x0000000c, 0x00000024, 0x0003003e, 0x00000025, 0x00000011, 0x00050041,
    0x00000012, 0x00000028, 0x0000000c, 0x00000026, 0x0003003e, 0x00000028, 0x00000027, 0x0004003d, 0x0000000d,
    0x0000002d, 0x0000002c, 0x00050041, 0x00000012, 0x0000002e, 0x0000000c, 0x0000002d, 0x0004003d, 0x00000007,
    0x0000002f, 0x0000002e, 0x0003003e, 0x0000002a, 0x0000002f, 0x0004003d, 0x0000000d, 0x00000031, 0x00000030,
    0x000500aa, 0x00000032, 0x00000033, 0x00000031, 0x0000000e, 0x000300f7, 0x00000035, 0x00000000, 0x000400fa,
    0x00000033, 0x00000034, 0x0000004e, 0x000200f8, 0x00000034, 0x00050041, 0x0000003b, 0x0000003c, 0x00000039,
    0x0000000e, 0x0003003e, 0x0000003c, 0x0000003a, 0x00050041, 0x00000044, 0x00000045, 0x00000043, 0x0000000e,
    0x0004003d, 0x00000040, 0x00000046, 0x00000045, 0x0004003d, 0x00000007, 0x00000047, 0x0000002a, 0x00050051,
    0x00000006, 0x00000048, 0x00000047, 0x00000000, 0x00050051, 0x00000006, 0x00000049, 0x00000047, 0x00000001,
    0x00050051, 0x00000006, 0x0000004a, 0x00000047, 0x00000002, 0x00070050, 0x00000036, 0x0000004b, 0x00000048,
    0x00000049, 0x0000004a, 0x0000000f, 0x00050091, 0x00000036, 0x0000004c, 0x00000046, 0x0000004b, 0x00050041,
    0x0000003b, 0x0000004d, 0x0000003f, 0x0000000e, 0x0003003e, 0x0000004d, 0x0000004c, 0x000200f9, 0x00000035,
    0x000200f8, 0x0000004e, 0x0004003d, 0x0000000d, 0x0000004f, 0x00000030, 0x000500aa, 0x00000032, 0x00000050,
    0x0000004f, 0x00000014, 0x000300f7, 0x00000052, 0x00000000, 0x000400fa, 0x00000050, 0x00000051, 0x0000005f,
    0x000200f8, 0x00000051, 0x00050041, 0x0000003b, 0x00000054, 0x00000039, 0x0000000e, 0x0003003e, 0x00000054,
    0x00000053, 0x00050041, 0x00000044, 0x00000055, 0x00000043, 0x0000000e, 0x0004003d, 0x00000040, 0x00000056,
    0x00000055, 0x0004003d, 0x00000007, 0x00000057, 0x0000002a, 0x0008004f, 0x00000007, 0x00000058, 0x00000057,
    0x00000057, 0x00000001, 0x00000000, 0x00000002, 0x00050051, 0x00000006, 0x00000059, 0x00000058, 0x00000000,
    0x00050051, 0x00000006, 0x0000005a, 0x00000058, 0x00000001, 0x00050051, 0x00000006, 0x0000005b, 0x00000058,
    0x00000002, 0x00070050, 0x00000036, 0x0000005c, 0x00000059, 0x0000005a, 0x0000005b, 0x0000000f, 0x00050091,
    0x00000036, 0x0000005d, 0x00000056, 0x0000005c, 0x00050041, 0x0000003b, 0x0000005e, 0x0000003f, 0x0000000e,
    0x0003003e, 0x0000005e, 0x0000005d, 0x000200f9, 0x00000052, 0x000200f8, 0x0000005f, 0x00050041, 0x0000003b,
    0x00000061, 0x00000039, 0x0000000e, 0x0003003e, 0x00000061, 0x00000060, 0x00050041, 0x00000044, 0x00000062,
    0x00000043, 0x0000000e, 0x0004003d, 0x00000040, 0x00000063, 0x00000062, 0x0004003d, 0x00000007, 0x00000064,
    0x0000002a, 0x0008004f, 0x00000007, 0x00000065, 0x00000064, 0x00000064, 0x00000001, 0x00000002, 0x00000000,
    0x00050051, 0x00000006, 0x00000066, 0x00000065, 0x00000000, 0x00050051, 0x00000006, 0x00000067, 0x00000065,
    0x00000001, 0x00050051, 0x00000006, 0x00000068, 0x00000065, 0x00000002, 0x00070050, 0x00000036, 0x00000069,
    0x00000066, 0x00000067, 0x00000068, 0x0000000f, 0x00050091, 0x00000036, 0x0000006a, 0x00000063, 0x00000069,
    0x00050041, 0x0000003b, 0x0000006b, 0x0000003f, 0x0000000e, 0x0003003e, 0x0000006b, 0x0000006a, 0x000200f9,
    0x00000052, 0x000200f8, 0x00000052, 0x000200f9, 0x00000035, 0x000200f8, 0x00000035, 0x000100fd, 0x00010038};


static std::vector<uint32_t> s_axis_frag_spv = {
    0x07230203, 0x00010000, 0x00080007, 0x00000012, 0x00000000, 0x00020011, 0x00000001, 0x0006000b, 0x00000001,
    0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0007000f, 0x00000004,
    0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x0000000c, 0x00030010, 0x00000004, 0x00000007, 0x00030003,
    0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00040005, 0x00000009, 0x6c6f4366,
    0x0000726f, 0x00030005, 0x0000000a, 0x00000000, 0x00050006, 0x0000000a, 0x00000000, 0x6f6c6f43, 0x00000072,
    0x00030005, 0x0000000c, 0x00006e49, 0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00040047, 0x0000000c,
    0x0000001e, 0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006,
    0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000003, 0x00000007,
    0x0004003b, 0x00000008, 0x00000009, 0x00000003, 0x0003001e, 0x0000000a, 0x00000007, 0x00040020, 0x0000000b,
    0x00000001, 0x0000000a, 0x0004003b, 0x0000000b, 0x0000000c, 0x00000001, 0x00040015, 0x0000000d, 0x00000020,
    0x00000001, 0x0004002b, 0x0000000d, 0x0000000e, 0x00000000, 0x00040020, 0x0000000f, 0x00000001, 0x00000007,
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x00050041, 0x0000000f,
    0x00000010, 0x0000000c, 0x0000000e, 0x0004003d, 0x00000007, 0x00000011, 0x00000010, 0x0003003e, 0x00000009,
    0x00000011, 0x000100fd, 0x00010038};

//--------------------------------------------------------------------------------------------------
//
//
void AxisVK::display(VkCommandBuffer cmdBuf, const nvmath::mat4f& transform, const VkExtent2D& screenSize)
{
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineTriangleFan);

  // Setup viewport:
  VkViewport viewport{};
  viewport.width    = float(screenSize.width);
  viewport.height   = float(screenSize.height);
  viewport.minDepth = 0;
  viewport.maxDepth = 1;
  VkRect2D rect;
  rect.offset = VkOffset2D{0, 0};
  rect.extent = VkExtent2D{screenSize.width, screenSize.height};
  vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
  vkCmdSetScissor(cmdBuf, 0, 1, &rect);


  // Set the orthographic matrix in the bottom left corner
  {
    const float         pixelW   = m_axisSize / screenSize.width;
    const float         pixelH   = m_axisSize / screenSize.height;
    const nvmath::mat4f matOrtho = {pixelW * .8f,  0.0f,          0.0f,  0.0f,  //
                                    0.0f,          -pixelH * .8f, 0.0f,  0.0f,  //
                                    0.0f,          0.0f,          -0.5f, 0.0f,  //
                                    -1.f + pixelW, 1.f - pixelH,  0.5f,  1.0f};

    nvmath::mat4f modelView = transform;
    modelView.set_translate({0, 0, 0});
    modelView = matOrtho * modelView;
    // Push the matrix to the shader
    vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(nvmath::mat4f), &modelView.a00);
  }

  // Draw 3 times the tip of the arrow, the shader is flipping the orientation and setting the color
  vkCmdDraw(cmdBuf, 6, 3, 0, 0);
  // Now draw the line of the arrow using the last 2 vertex of the buffer (offset 5)
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLines);
  vkCmdDraw(cmdBuf, 2, 3, 5, 0);
}

void AxisVK::createAxisObject(VkRenderPass renderPass, uint32_t subpass)
{
  // The shader need Push Constants: the transformation matrix
  const VkPushConstantRange push_constants{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(nvmath::mat4f)};

  VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges    = &push_constants;
  vkCreatePipelineLayout(m_device, &layout_info, nullptr, &m_pipelineLayout);

  // Creation of the pipeline
  VkShaderModule smVertex;
  VkShaderModule smFrag;

  VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  createInfo.codeSize = sizeof(uint32_t) * s_axis_vert_spv.size();
  createInfo.pCode    = reinterpret_cast<const uint32_t*>(s_axis_vert_spv.data());
  vkCreateShaderModule(m_device, &createInfo, nullptr, &smVertex);
  createInfo.codeSize = sizeof(uint32_t) * s_axis_frag_spv.size();
  createInfo.pCode    = reinterpret_cast<const uint32_t*>(s_axis_frag_spv.data());
  vkCreateShaderModule(m_device, &createInfo, nullptr, &smFrag);

  // Pipeline state
  nvvk::GraphicsPipelineState gps;
  gps.inputAssemblyState.topology         = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
  gps.rasterizationState.cullMode         = VK_CULL_MODE_NONE;
  gps.depthStencilState.depthTestEnable   = VK_TRUE;
  gps.depthStencilState.stencilTestEnable = VK_FALSE;
  gps.depthStencilState.depthCompareOp    = VK_COMPARE_OP_LESS_OR_EQUAL;

  // Creating the tips
  nvvk::GraphicsPipelineGenerator gpg(m_device, m_pipelineLayout, renderPass, gps);
  gpg.addShader(smVertex, VK_SHADER_STAGE_VERTEX_BIT);
  gpg.addShader(smFrag, VK_SHADER_STAGE_FRAGMENT_BIT);
  m_pipelineTriangleFan = gpg.createPipeline();

  // Creating the lines
  gps.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
  m_pipelineLines                 = gpg.createPipeline();

  vkDestroyShaderModule(m_device, smVertex, nullptr);
  vkDestroyShaderModule(m_device, smFrag, nullptr);
}

}  // namespace nvvk


// glsl_shader.vert, compiled with:
// # glslangValidator -o axis.vert.h -V axis.vert -vn s_vert_spv
/*************************************************
#version 450 core

layout(push_constant) uniform uPushConstant
{
  mat4 transform;
}
pc;

out gl_PerVertex
{
  vec4 gl_Position;
};

layout(location = 0) out interpolant
{
  vec4 Color;
} Out;

const float asize = 1.0f;
const float atip  = 0.2f;
const float abase = 0.75f;

vec3 arrow_vert[7];

void main()
{
  arrow_vert[0] = vec3(asize, 0, 0);
  arrow_vert[1] = vec3(abase, atip, 0);
  arrow_vert[2] = vec3(abase, -atip * 0.5f, -atip * 0.87f);
  arrow_vert[3] = vec3(abase, -atip * 0.5f, atip * 0.87f);
  arrow_vert[4] = vec3(abase, atip, 0);
  arrow_vert[5] = vec3(asize, 0, 0);  // To draw the line
  arrow_vert[6] = vec3(0, 0, 0);      // ...

 vec3 pos = arrow_vert[gl_VertexIndex];
  //Out.Color = aColor;
  if(gl_InstanceIndex == 0)
  {
    Out.Color   = vec4(1, 0, 0, 1);
    gl_Position = pc.transform * vec4(pos.xyz, 1);
  }
  else if(gl_InstanceIndex == 1)
  {
    Out.Color   = vec4(0, 1, 0, 1);
    gl_Position = pc.transform * vec4(pos.yxz, 1);
  }
  else
  {
    Out.Color   = vec4(0, 0, 1, 1);
    gl_Position = pc.transform * vec4(pos.yzx, 1);
  }
}
 
*********************/


// glsl_shader.frag, compiled with:
// # glslangValidator -V -x -o glsl_shader.frag.u32 glsl_shader.frag
/*************************************************
#version 450 core
layout(location = 0) out vec4 fColor;


layout(location = 0) in interpolant
{
  vec4 Color;
} In;

void main()
{
  fColor = In.Color;
}


*********************/
