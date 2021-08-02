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

#ifndef NV_SHADERMODULEMANAGER_INCLUDED
#define NV_SHADERMODULEMANAGER_INCLUDED


#include <stdio.h>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

#if USESHADERC
#define NV_EXTENSIONS
#include <shaderc/shaderc.h>
#undef NV_EXTENSIONS
#endif

#include <nvh/shaderfilemanager.hpp>


namespace nvvk {

//////////////////////////////////////////////////////////////////////////
/**
  # class nvvk::ShaderModuleManager

  The ShaderModuleManager manages VkShaderModules stored in files (SPIR-V or GLSL)

  Using ShaderFileManager it will find the files and resolve #include for GLSL.
  You must add include directories to the base-class for this.

  It also comes with some convenience functions to reload shaders etc.
  That is why we pass out the ShaderModuleID rather than a VkShaderModule directly.

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
*/

class ShaderModuleID
{
public:
  size_t m_value;

  ShaderModuleID()
      : m_value(size_t(~0))
  {
  }

  ShaderModuleID(size_t b)
      : m_value(b)
  {
  }
  ShaderModuleID& operator=(size_t b)
  {
    m_value = b;
    return *this;
  }

  bool isValid() const { return m_value != size_t(~0); }

  operator bool() const { return isValid(); }
  operator size_t() const { return m_value; }

  friend bool operator==(const ShaderModuleID& lhs, const ShaderModuleID& rhs) { return rhs.m_value == lhs.m_value; }
};

class ShaderModuleManager : public nvh::ShaderFileManager
{
public:
  struct ShaderModule
  {
    ShaderModule()
        : module(0)
    {
    }

    VkShaderModule module;
    std::string    moduleSPIRV;
    Definition     definition;
  };

  void init(VkDevice device, int apiMajor = 1, int apiMinor = 1);

  // also calls deleteShaderModules
  void deinit();

  ShaderModuleID createShaderModule(uint32_t           type,
                                    std::string const& filename,
                                    std::string const& prepend   = "",
                                    FileType           fileType  = FILETYPE_DEFAULT,
                                    std::string const& entryname = "main");

  void destroyShaderModule(ShaderModuleID idx);
  void reloadModule(ShaderModuleID idx);

  void reloadShaderModules();
  void deleteShaderModules();
  bool areShaderModulesValid();

#if USESHADERC
  void setOptimizationLevel(shaderc_optimization_level level) { m_shadercOptimizationLevel = level; }
#endif


  bool                isValid(ShaderModuleID idx) const;
  VkShaderModule      get(ShaderModuleID idx) const;
  ShaderModule&       getShaderModule(ShaderModuleID idx);
  const ShaderModule& getShaderModule(ShaderModuleID idx) const;
  const char*         getCode(ShaderModuleID idx, size_t* len = NULL) const;
  const size_t        getCodeLen(ShaderModuleID idx) const;
  bool                dumpSPIRV(ShaderModuleID idx, const char * filename) const;


  // state will affect the next created shader module
  // also keep m_filetype in mind!
  bool m_preprocessOnly  = false;
  bool m_keepModuleSPIRV = false;

  //////////////////////////////////////////////////////////////////////////
  //
  // for internal development, useful when we have new shader types that
  // are not covered by public VulkanSDK

  struct SetupInterface
  {
    // This class is to aid using a shaderc library version that is not
    // provided by the Vulkan SDK, but custom. Therefore it allows custom settings etc.
    // Useful for driver development of new shader stages, otherwise can be pretty much ignored.

    virtual std::string getTypeDefine(uint32_t type) const      = 0;
    virtual uint32_t    getTypeShadercKind(uint32_t type) const = 0;
    virtual void*       getShadercCompileOption(void* shadercCompiler) { return nullptr; }
  };

  void setSetupIF(SetupInterface* setupIF);


  ShaderModuleManager(ShaderModuleManager const&) = delete;
  ShaderModuleManager& operator=(ShaderModuleManager const&) = delete;

  ShaderModuleManager()
  {
    m_usedSetupIF             = &m_defaultSetupIF;
    m_supportsExtendedInclude = true;
#if USESHADERC
    s_shadercCompilerUsers++;
    if(!s_shadercCompiler)
    {
      s_shadercCompiler = shaderc_compiler_initialize();
    }
    m_shadercOptions = shaderc_compile_options_initialize();
#endif
  }
  ShaderModuleManager(VkDevice device)
  {
    m_usedSetupIF             = &m_defaultSetupIF;
    m_supportsExtendedInclude = true;
#if USESHADERC
    s_shadercCompilerUsers++;
    if(!s_shadercCompiler)
    {
      s_shadercCompiler = shaderc_compiler_initialize();
    }
    m_shadercOptions = shaderc_compile_options_initialize();
#endif
    init(device);
  }

  ~ShaderModuleManager()
  {
    deinit();
#if USESHADERC
    s_shadercCompilerUsers--;
    if(s_shadercCompiler && s_shadercCompilerUsers == 0)
    {
      shaderc_compiler_release(s_shadercCompiler);
      s_shadercCompiler = nullptr;
    }
    if(m_shadercOptions)
    {
      shaderc_compile_options_release(m_shadercOptions);
    }
#endif
  }

private:
  ShaderModuleID createShaderModule(const Definition& def);
  bool           setupShaderModule(ShaderModule& prog);


  struct DefaultInterface : public SetupInterface
  {
    std::string getTypeDefine(uint32_t type) const override;
    uint32_t    getTypeShadercKind(uint32_t type) const override;
  };


  static const VkShaderModule PREPROCESS_ONLY_MODULE;

  VkDevice         m_device = VK_NULL_HANDLE;
  DefaultInterface m_defaultSetupIF;
  SetupInterface*  m_usedSetupIF = nullptr;

  int m_apiMajor = 1;
  int m_apiMinor = 1;

#if USESHADERC
  static shaderc_compiler_t  s_shadercCompiler;
  static uint32_t            s_shadercCompilerUsers;
  shaderc_compile_options_t  m_shadercOptions           = nullptr;
  shaderc_optimization_level m_shadercOptimizationLevel = shaderc_optimization_level_performance;
#endif

  std::vector<ShaderModule> m_shadermodules;
};

}  // namespace nvvk


#endif  //NV_PROGRAM_INCLUDED
