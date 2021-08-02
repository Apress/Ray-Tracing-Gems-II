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

#include "shadermodulemanager_vk.hpp"
#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdarg.h>
#include <stdio.h>

#include <nvh/fileoperations.hpp>
#include <nvh/nvprint.hpp>

#define NV_LINE_MARKERS 1


namespace nvvk {

const VkShaderModule ShaderModuleManager::PREPROCESS_ONLY_MODULE = (VkShaderModule)~0;
#if USESHADERC
shaderc_compiler_t ShaderModuleManager::s_shadercCompiler      = nullptr;
uint32_t           ShaderModuleManager::s_shadercCompilerUsers = 0;
#endif

std::string ShaderModuleManager::DefaultInterface::getTypeDefine(uint32_t type) const
{
  switch(type)
  {
    case VK_SHADER_STAGE_VERTEX_BIT:
      return "#define _VERTEX_SHADER_ 1\n";
    case VK_SHADER_STAGE_FRAGMENT_BIT:
      return "#define _FRAGMENT_SHADER_ 1\n";
    case VK_SHADER_STAGE_COMPUTE_BIT:
      return "#define _COMPUTE_SHADER_ 1\n";
    case VK_SHADER_STAGE_GEOMETRY_BIT:
      return "#define _GEOMETRY_SHADER_ 1\n";
    case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
      return "#define _TESS_CONTROL_SHADER_ 1\n";
    case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
      return "#define _TESS_EVALUATION_SHADER_ 1\n";
#if VK_NV_mesh_shader
    case VK_SHADER_STAGE_MESH_BIT_NV:
      return "#define _MESH_SHADER_ 1\n";
    case VK_SHADER_STAGE_TASK_BIT_NV:
      return "#define _TASK_SHADER_ 1\n";
#endif
#if VK_NV_ray_tracing
    case VK_SHADER_STAGE_RAYGEN_BIT_NV:
      return "#define _RAY_GENERATION_SHADER_ 1\n";
    case VK_SHADER_STAGE_ANY_HIT_BIT_NV:
      return "#define _RAY_ANY_HIT_SHADER_ 1\n";
    case VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV:
      return "#define _RAY_CLOSEST_HIT_SHADER_ 1\n";
    case VK_SHADER_STAGE_MISS_BIT_NV:
      return "#define _RAY_MISS_SHADER_ 1\n";
    case VK_SHADER_STAGE_INTERSECTION_BIT_NV:
      return "#define _RAY_INTERSECTION_SHADER_ 1\n";
    case VK_SHADER_STAGE_CALLABLE_BIT_NV:
      return "#define _RAY_CALLABLE_BIT_SHADER_ 1\n";
#endif
  }

  return std::string();
}

uint32_t ShaderModuleManager::DefaultInterface::getTypeShadercKind(uint32_t type) const
{
#if USESHADERC
  switch(type)
  {
    case VK_SHADER_STAGE_VERTEX_BIT:
      return shaderc_glsl_vertex_shader;
    case VK_SHADER_STAGE_FRAGMENT_BIT:
      return shaderc_glsl_fragment_shader;
    case VK_SHADER_STAGE_COMPUTE_BIT:
      return shaderc_glsl_compute_shader;
    case VK_SHADER_STAGE_GEOMETRY_BIT:
      return shaderc_glsl_geometry_shader;
    case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
      return shaderc_glsl_tess_control_shader;
    case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
      return shaderc_glsl_tess_evaluation_shader;

#if VK_NV_mesh_shader
    case VK_SHADER_STAGE_MESH_BIT_NV:
      return shaderc_glsl_mesh_shader;
    case VK_SHADER_STAGE_TASK_BIT_NV:
      return shaderc_glsl_task_shader;
#endif
#if VK_NV_ray_tracing
    case VK_SHADER_STAGE_RAYGEN_BIT_NV:
      return shaderc_glsl_raygen_shader;
    case VK_SHADER_STAGE_ANY_HIT_BIT_NV:
      return shaderc_glsl_anyhit_shader;
    case VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV:
      return shaderc_glsl_closesthit_shader;
    case VK_SHADER_STAGE_MISS_BIT_NV:
      return shaderc_glsl_miss_shader;
    case VK_SHADER_STAGE_INTERSECTION_BIT_NV:
      return shaderc_glsl_intersection_shader;
    case VK_SHADER_STAGE_CALLABLE_BIT_NV:
      return shaderc_glsl_callable_shader;
#endif
  }

  return shaderc_glsl_infer_from_source;
#else
  return 0;
#endif
}

bool ShaderModuleManager::setupShaderModule(ShaderModule& module)
{
  Definition& definition = module.definition;

  module.module = VK_NULL_HANDLE;
  if(definition.filetype == FILETYPE_DEFAULT)
  {
    definition.filetype = m_filetype;
  }

  std::string combinedPrepend = m_prepend;
  std::string combinedFilenames;
  combinedPrepend += definition.prepend;
  combinedFilenames += definition.filename;

  if(definition.filetype == FILETYPE_SPIRV)
  {
    std::string filenameFound;
    definition.content = nvh::loadFile(definition.filename, true, m_directories, filenameFound);
  }
  else
  {
    std::string prepend = m_usedSetupIF->getTypeDefine(definition.type);

    definition.content =
        manualInclude(definition.filename, definition.filenameFound, prepend + m_prepend + definition.prepend, false);
  }

  if(definition.content.empty())
  {
    return false;
  }

  if(m_preprocessOnly)
  {
    module.module = PREPROCESS_ONLY_MODULE;
    return true;
  }
  else
  {
    VkResult                 vkresult         = VK_ERROR_INVALID_SHADER_NV;
    VkShaderModuleCreateInfo shaderModuleInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};

#if USESHADERC
    shaderc_compilation_result_t result = nullptr;
    if(definition.filetype == FILETYPE_GLSL)
    {
      shaderc_shader_kind shaderkind = (shaderc_shader_kind)m_usedSetupIF->getTypeShadercKind(definition.type);
      shaderc_compile_options_t options = (shaderc_compile_options_t)m_usedSetupIF->getShadercCompileOption(s_shadercCompiler);
      if(!options)
      {
        if(m_apiMajor == 1 && m_apiMinor == 0)
        {
          shaderc_compile_options_set_target_env(m_shadercOptions, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_0);
        }
        else if(m_apiMajor == 1 && m_apiMinor == 1)
        {
          shaderc_compile_options_set_target_env(m_shadercOptions, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_1);
        }
        else if(m_apiMajor == 1 && m_apiMinor == 2)
        {
          shaderc_compile_options_set_target_env(m_shadercOptions, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
        }

        shaderc_compile_options_set_optimization_level(m_shadercOptions, m_shadercOptimizationLevel);
        
        // Keep debug info, doesn't cost shader execution perf, only compile-time and memory size.
        // Improves usage for debugging tools, not recommended for shipping application,
        // but good for developmenent builds.
        shaderc_compile_options_set_generate_debug_info(m_shadercOptions);

        options = m_shadercOptions;
      }

      result = shaderc_compile_into_spv(s_shadercCompiler, definition.content.c_str(), definition.content.size(),
                                        shaderkind, definition.filename.c_str(), "main", options);

      if(!result)
      {
        return false;
      }

      if(shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success)
      {
        bool failedToOptimize = strstr(shaderc_result_get_error_message(result), "failed to optimize");
        int  level            = failedToOptimize ? LOGLEVEL_WARNING : LOGLEVEL_ERROR;
        nvprintfLevel(level, "%s: optimization_level_performance\n", definition.filename.c_str());
        nvprintfLevel(level, "  %s\n", definition.prepend.c_str());
        nvprintfLevel(level, "  %s\n", shaderc_result_get_error_message(result));
        shaderc_result_release(result);

        if(!failedToOptimize || options != m_shadercOptions)
        {
          return false;
        }

        // try again without optimization
        shaderc_compile_options_set_optimization_level(m_shadercOptions, shaderc_optimization_level_zero);

        result = shaderc_compile_into_spv(s_shadercCompiler, definition.content.c_str(), definition.content.size(),
                                          shaderkind, definition.filename.c_str(), "main", options);
      }

      if(shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success)
      {
        LOGE("%s: optimization_level_zero\n", definition.filename.c_str());
        LOGE("  %s\n", definition.prepend.c_str());
        LOGE("  %s\n", shaderc_result_get_error_message(result));
        shaderc_result_release(result);
        return false;
      }

      shaderModuleInfo.codeSize = shaderc_result_get_length(result);
      shaderModuleInfo.pCode    = (const uint32_t*)shaderc_result_get_bytes(result);
    }
    else
#else
    if(definition.filetype == FILETYPE_GLSL)
    {
      LOGW("No direct GLSL support\n");
      return false;
    }
    else
#endif
    {
      shaderModuleInfo.codeSize = definition.content.size();
      shaderModuleInfo.pCode    = (const uint32_t*)definition.content.c_str();
    }

    vkresult = ::vkCreateShaderModule(m_device, &shaderModuleInfo, nullptr, &module.module);

    if (vkresult == VK_SUCCESS && m_keepModuleSPIRV) {
      module.moduleSPIRV = std::string((const char*)shaderModuleInfo.pCode, shaderModuleInfo.codeSize);
    }

#if USESHADERC
    if(result)
    {
      shaderc_result_release(result);
    }
#endif

    return vkresult == VK_SUCCESS;
  }
}

void ShaderModuleManager::init(VkDevice device, int apiMajor, int apiMinor)
{
  assert(!m_device);
  m_device   = device;
  m_apiMajor = apiMajor;
  m_apiMinor = apiMinor;
}

void ShaderModuleManager::deinit()
{
  deleteShaderModules();
  m_device = VK_NULL_HANDLE;
}

ShaderModuleID ShaderModuleManager::createShaderModule(const Definition& definition)
{
  ShaderModule module;
  module.definition     = definition;

  setupShaderModule(module);

  // find unused
  for(size_t i = 0; i < m_shadermodules.size(); i++)
  {
    if(m_shadermodules[i].definition.type == 0)
    {
      m_shadermodules[i] = module;
      return i;
    }
  }

  m_shadermodules.push_back(module);
  return m_shadermodules.size() - 1;
}

ShaderModuleID ShaderModuleManager::createShaderModule(uint32_t           type,
                                                       std::string const& filename,
                                                       std::string const& prepend,
                                                       FileType           fileType /*= FILETYPE_DEFAULT*/,
                                                       std::string const& entryname /*= "main"*/)
{
  Definition def;
  def.type     = type;
  def.filename = filename;
  def.prepend  = prepend;
  def.filetype = fileType;
  def.entry    = entryname;
  return createShaderModule(def);
}

bool ShaderModuleManager::areShaderModulesValid()
{
  bool valid = true;
  for(size_t i = 0; i < m_shadermodules.size(); i++)
  {
    valid = valid && isValid(i);
  }
  return valid;
}

void ShaderModuleManager::deleteShaderModules()
{
  for(size_t i = 0; i < m_shadermodules.size(); i++)
  {
    destroyShaderModule((ShaderModuleID)i);
  }
  m_shadermodules.clear();
}

void ShaderModuleManager::reloadModule(ShaderModuleID idx)
{
  if(!isValid(idx))
    return;

  ShaderModule& module = getShaderModule(idx);

  bool old         = m_preprocessOnly;
  m_preprocessOnly = module.module == PREPROCESS_ONLY_MODULE;
  if(module.module && module.module != PREPROCESS_ONLY_MODULE)
  {
    vkDestroyShaderModule(m_device, module.module, nullptr);
    module.module = nullptr;
  }
  if(module.definition.type != 0)
  {
    setupShaderModule(module);
  }

  m_preprocessOnly = old;
}

void ShaderModuleManager::reloadShaderModules()
{
  LOGI("Reloading programs...\n");

  for(size_t i = 0; i < m_shadermodules.size(); i++)
  {
    reloadModule((ShaderModuleID)i);
  }

  LOGI("done\n");
}

bool ShaderModuleManager::isValid(ShaderModuleID idx) const
{
  return (m_shadermodules[idx].definition.type && m_shadermodules[idx].module != 0) || !m_shadermodules[idx].definition.type;
}

VkShaderModule ShaderModuleManager::get(ShaderModuleID idx) const
{
  return m_shadermodules[idx].module;
}

ShaderModuleManager::ShaderModule& ShaderModuleManager::getShaderModule(ShaderModuleID idx)
{
  return m_shadermodules[idx];
}

const ShaderModuleManager::ShaderModule& ShaderModuleManager::getShaderModule(ShaderModuleID idx) const
{
  return m_shadermodules[idx];
}

void ShaderModuleManager::destroyShaderModule(ShaderModuleID idx)
{
  if(!isValid(idx))
    return;

  ShaderModule& module = getShaderModule(idx);

  if(module.module && module.module != PREPROCESS_ONLY_MODULE)
  {
    vkDestroyShaderModule(m_device, module.module, nullptr);
    module.module = 0;
  }
  module.definition = Definition();
}

const char* ShaderModuleManager::getCode(ShaderModuleID idx, size_t* len) const
{
  return m_shadermodules[idx].definition.content.c_str();
}
const size_t ShaderModuleManager::getCodeLen(ShaderModuleID idx) const
{
  return m_shadermodules[idx].definition.content.size();
}


bool ShaderModuleManager::dumpSPIRV(ShaderModuleID idx, const char* filename) const 
{
  if (m_shadermodules[idx].moduleSPIRV.empty()) return false;

  FILE* f = fopen(filename, "wb");
  if (f) {
    fwrite(m_shadermodules[idx].moduleSPIRV.data(), m_shadermodules[idx].moduleSPIRV.size(), 1, f);
    fclose(f);
    return true;
  }

  return false;
}

}  // namespace nvvk
