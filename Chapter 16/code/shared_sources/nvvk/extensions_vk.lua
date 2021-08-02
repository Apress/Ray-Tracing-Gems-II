--[[
/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
]]

-- HOW TO USE
--
-- 1. Setup environment variable NVVK_VULKAN_XML pointing to vk.xml
--    or use VULKAN_SDK >= 1.2.135.0
--
-- 2. Modify the enablelist of extensions you may want to use.
--    Only those extensions that have functions will get a loader
--    (load_VK_BLAH_whatever) so not all extensions in this list
--    may get a function in the end.
--
-- 3. Check out this and the other extensions_vk files for write access
--
-- 4. Run with a lua5.1 compatible lua runtime and the lua2xml project
--    https://github.com/manoelcampos/xml2lua
--    (shared_internal has all the files).
--
--    lua extensions_vk.lua
--
--    within this directory.

local VULKAN_XML = os.getenv("NVVK_VULKAN_XML") or os.getenv("VULKAN_SDK").."/share/vulkan/registry/vk.xml"
local extensionSubset = [[
    VK_KHR_ray_tracing
    VK_KHR_push_descriptor
    VK_KHR_8bit_storage
    VK_KHR_create_renderpass2
    VK_KHR_depth_stencil_resolve
    VK_KHR_draw_indirect_count
    VK_KHR_driver_properties
    VK_KHR_pipeline_executable_properties
    VK_KHR_buffer_device_address
    
    VK_NV_compute_shader_derivatives
    VK_NV_cooperative_matrix
    VK_NV_corner_sampled_image
    VK_NV_coverage_reduction_mode
    VK_NV_dedicated_allocation_image_aliasing
    VK_NV_mesh_shader
    VK_NV_ray_tracing
    VK_NV_representative_fragment_test
    VK_NV_shading_rate_image
    VK_NV_viewport_array2
    VK_NV_viewport_swizzle
    VK_NV_scissor_exclusive
    VK_NV_device_generated_commands
    
    VK_EXT_buffer_device_address
    VK_EXT_debug_marker
    VK_EXT_calibrated_timestamps
    VK_EXT_conservative_rasterization
    VK_EXT_descriptor_indexing
    VK_EXT_depth_clip_enable
    VK_EXT_memory_budget
    VK_EXT_memory_priority
    VK_EXT_pci_bus_info
    VK_EXT_sample_locations
    VK_EXT_sampler_filter_minmax
    VK_EXT_texel_buffer_alignment
    VK_EXT_debug_utils
    VK_EXT_host_query_reset
    
    VK_KHR_external_memory_win32
    VK_KHR_external_semaphore_win32
    VK_KHR_external_fence_win32

    VK_KHR_external_memory_fd
    VK_KHR_external_semaphore_fd
	
	VK_KHR_acceleration_structure
    VK_KHR_ray_tracing_pipeline
	VK_KHR_pipeline_library
    ]]

local function extractFeatureDefs(features, filename, enablelist)
  assert(filename, "filename not provided")
  
  local xml2lua = require("xml2lua")
  local handler = require("xmlhandler.tree")
  
  local f = io.open(filename,"rt")
  assert(f, filename.." not found")
  
  local xml = f:read("*a")
  f:close()
  
  -- Bug workaround https://github.com/manoelcampos/xml2lua/issues/35
  xml = xml:gsub("(<param>)(<type>[%w_]+</type>)%* ", function(p,typ)
        -- add _ dummy symbol
        return "<param>_"..typ.."* "
      end)
    
  local parser = xml2lua.parser(handler)
  parser:parse(xml)
  
  local version = xml:match("VK_HEADER_VERSION</name> (%d+)")
  assert(version)
  
  xml = nil

  --[[
          <command>
              <proto><type>VkDeviceAddress</type> <name>vkGetBufferDeviceAddress</name></proto>
              <param><type>VkDevice</type> <name>device</name></param>
              <param>const <type>VkBufferDeviceAddressInfo</type>* <name>pInfo</name></param>
          </command>
          <command name="vkGetBufferDeviceAddressKHR"        alias="vkGetBufferDeviceAddress"/>
          <command name="vkGetBufferDeviceAddressEXT"        alias="vkGetBufferDeviceAddress"/>
          
          
          <extension name="VK_EXT_buffer_device_address" number="245" type="device" requires="VK_KHR_get_physical_device_properties2" author="NV" contact="Jeff Bolz @jeffbolznv"  deprecatedby="VK_KHR_buffer_device_address" supported="vulkan">
              <require>
                  ...
                  <command name="vkGetBufferDeviceAddressEXT"/>
              </require>
              
          <extension name="VK_KHR_win32_surface" number="10" type="instance" requires="VK_KHR_surface" platform="win32" author="KHR" contact="Jesse Hall @critsec,Ian Elliott @ianelliottus" supported="vulkan">
              <require>
                  ...
                  <command name="vkCreateWin32SurfaceKHR"/>
                  <command name="vkGetPhysicalDeviceWin32PresentationSupportKHR"/>
              </require>
          </extension>
  ]]
  
  local types      = handler.root.registry.types
  local commands   = handler.root.registry.commands
  local extensions = handler.root.registry.extensions.extension
  
  -- debugging
  if (false) then
    local serpent = require "serpent"
    local f = io.open(filename..".cmds.lua", "wt")
    f:write(serpent.block(commands))
    local f = io.open(filename..".exts.lua", "wt")
    f:write(serpent.block(extensions))
  end
  
  -- build list of alias'ed types
  local lktypes = {}
  for _,v in ipairs(types.type) do
    local alias = v._attr.alias
    if (alias) then
      lktypes[v._attr.name] = alias
    end
  end

  -- build list of commands, and aliased commands
  local lkcmds = {}
  for _,v in ipairs(commands.command) do
    if (v.proto) then
      lkcmds[v.proto.name] = v
    elseif (v._attr.alias) then
      lkcmds[v._attr.name] = lkcmds[v._attr.alias]
      assert(lkcmds[v._attr.name].proto.name)
    end
  end
  
  
  local platforms = {
    ggp = "VK_USE_PLATFORM_GGP",
    win32 = "VK_USE_PLATFORM_WIN32_KHR",
    vi = "VK_USE_PLATFORM_VI_NN",
    ios = "VK_USE_PLATFORM_IOS_MVK",
    macos = "VK_USE_PLATFORM_MACOS_MVK",
    android = "VK_USE_PLATFORM_ANDROID_KHR",
    fuchsia = "VK_USE_PLATFORM_FUCHSIA",
    metal = "VK_USE_PLATFORM_METAL_EXT",
    xlib = "VK_USE_PLATFORM_XLIB_KHR",
    xcb = "VK_USE_PLATFORM_XCB_KHR",
    wayland = "VK_USE_PLATFORM_WAYLAND_KHR",
    xlib_xrandr = "VK_USE_PLATFORM_XLIB_XRANDR_EXT",
  }
  for _,v in ipairs(extensions) do
    local extname = v._attr.name
    local extrequires = v.require
    if (extrequires and not extrequires[1]) then
      extrequires = {extrequires}
    end
    
    if (enablelist[extname]) then
      local alias = {}
      local cmds = {}
      local hascmds = false
      local addedcmds = {}
      for _,r in ipairs(extrequires) do
        local extcmds = r.command
        local exttypes = r.type
        if (extcmds) then
          -- convert single entry to array
          if (not extcmds[1]) then
            extcmds = {extcmds}
          end
          -- extract commands used by extension
          for _,c in ipairs(extcmds) do
            local cname = c._attr.name
            local cmd = lkcmds[cname]
            
            assert(cmd, extname..":"..cname)
            if (not addedcmds[cname]) then
              table.insert(cmds, {name=cname, cmd=cmd})
              addedcmds[cname] = true
            end
          end
          -- extract type aliasing 
          -- we prefer to use the extensions original types
          -- which may have been replaced through aliasing
          if (exttypes) then
            if (not exttypes[1]) then
              exttypes = {exttypes}
            end
            for _,t in ipairs(exttypes) do
              local tname = t._attr.name
              local aname = lktypes[tname]
              if (aname) then
                alias[aname] = tname
              end
            end
          end
        end
      end
      if (cmds[1]) then
        print(extname)
        table.insert(features, {feature=extname, typ=v._attr.type, cmds=cmds, alias=alias, platform=platforms[v._attr.platform or "_"]})
      end
    end
  end
  
  return version
end

local function generate(outfilename, header, enablelist)

  local function toTab(str)
    local tab = {}
    for name in str:gmatch("[%w_]+") do 
      tab[name] = true
    end
    return tab
  end
  
  local enablelist = enablelist and toTab(enablelist)

  local xmlfile = VULKAN_XML
  
  local vk_features = {}
  local vk_version = extractFeatureDefs(vk_features, xmlfile, enablelist)
  
  local availables_header = ""
  local availables_source = ""
  
  local loaders_header = ""
  local loaders_source = ""
  local loaders_all = ""
  local reset_all = ""
  
  local function process(f, api, apientry) 
    local func      = ""
    local initfunc  = ""
    local success   = ""
    local variable  = ""
    local reset     = ""
    
    for _,c in ipairs(f.cmds) do
      assert(c.cmd and c.name, f.feature)
      local cmd = c.cmd
      local name = c.name
      local dbgname = f.feature..":"..name
      
      local prot = "PFN_"..name
      local mangled = "pfn_"..name
      local ret = cmd.proto.type
      
      local params = cmd.param
      assert(params, dbgname)
      if (not (params[1] and params[1].type))  then
        params = {params}
      end
      
      local isInstance = params[1].type == "VkInstance"
      
      -- argument definition
      local args = "(\n"
      -- argument passing
      local exec = (ret ~= "void" and "return " or "")..mangled.."("
      
      for _,p in ipairs(params) do
        assert(p.type and p.name, dbgname)
        -- convert promoted types back to extension types
        local typ = f.alias[p.type] or p.type
        -- handle qualifiers
        local p1 = p[1] == "const" and "const " or ""
        local function pointer(a,b)
          --local convert = {["*"] = "*", ["**"] = "**", ["* const*"] = "* const*",}
          return (a or b or ""):find("*",0, true) and (a or b) or ""
        end
        local p2 = pointer(p[2],p[1])
        args = args.."    "..p1..typ..p2.." "..p.name..",\n"
        exec = exec..p.name..","
      end
      
      args = args:sub(1,-3)..")"
      exec = exec:sub(1,-2)..");"
      
      variable = variable.."static "..prot.." "..mangled.." = 0;\n"
      reset    = reset.."  "..prot.." "..mangled.." = 0;\n"
      
      func = func..api..ret.." "..apientry..name..args.."\n{\n"
      func = func.."  assert("..mangled..");\n"
      func = func.."  "..exec.."\n"
      func = func.."}\n"
      
      if (isInstance) then
        initfunc  = initfunc.."  "..mangled..' = ('..prot..')getInstanceProcAddr(instance, "'..name..'");\n'
      else
        initfunc  = initfunc.."  "..mangled..' = ('..prot..')getDeviceProcAddr(device, "'..name..'");\n'
      end
      success  = success.."  success = success && ("..mangled..' != 0);\n'
    end
    
    if (initfunc ~= "") then
      local preproc_condition = "#if "..f.feature..(f.platform and " && defined("..f.platform..")" or "").."\n"
      
      loaders_header = loaders_header..preproc_condition
      loaders_header = loaders_header.."#define LOADER_"..f.feature.." 1\n"
      loaders_header = loaders_header.."int load_"..f.feature.."(VkInstance instance, PFN_vkGetInstanceProcAddr getInstanceProcAddr, VkDevice device, PFN_vkGetDeviceProcAddr getDeviceProcAddr);\n"
      loaders_header = loaders_header.."extern int has_"..f.feature..";\n"
      loaders_header = loaders_header.."#endif\n\n"
      
      loaders_source = loaders_source.."/* /////////////////////////////////// */\n"
      loaders_source = loaders_source..preproc_condition
      loaders_source = loaders_source..variable.."\n"
      loaders_source = loaders_source..func.."\n"
      loaders_source = loaders_source.."int has_"..f.feature.." = 0;\n"
      loaders_source = loaders_source.."int load_"..f.feature.."(VkInstance instance, PFN_vkGetInstanceProcAddr getInstanceProcAddr, VkDevice device, PFN_vkGetDeviceProcAddr getDeviceProcAddr)\n{\n"
      loaders_source = loaders_source..initfunc
      loaders_source = loaders_source.."  int success = 1;\n"
      loaders_source = loaders_source..success
      loaders_source = loaders_source.."  has_"..f.feature.." = success;\n"
      loaders_source = loaders_source.."  return success;\n}\n"
      loaders_source = loaders_source.."#endif\n\n"
      
      loaders_all = loaders_all.."  "..preproc_condition
      loaders_all = loaders_all.."  load_"..f.feature.."(instance, getInstanceProcAddr, device, getDeviceProcAddr);\n"
      loaders_all = loaders_all.."  #endif\n"
      
      reset_all = reset_all.."  "..preproc_condition
      reset_all = reset_all.."  has_"..f.feature.." = 0;\n"
      reset_all = reset_all..reset
      reset_all = reset_all.."  #endif\n\n"
    else
      error("nofuncs "..f.feature)
    end
  end
  
  for i,f in ipairs(vk_features) do
    process(f, "VKAPI_ATTR ", "VKAPI_CALL ")
  end

  local fheader = io.open(outfilename..".hpp", "wt")
  local fsource = io.open(outfilename..".cpp", "wt")
  assert(fheader, "could not open "..outfilename..".hpp for writing")
  assert(fsource, "could not open "..outfilename..".cpp for writing")
    
fheader:write("/* based on VK_HEADER_VERSION "..vk_version.." */\n"..header..[[

/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

//////////////////////////////////////////////////////////////////////////
/**
  # Vulkan Extension Loader

  The extensions_vk files takes care of loading and providing the symbols of
  Vulkan C Api extensions.
  It is generated by `extensions_vk.lua` which contains a enablelist of
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

*/

#pragma once

#include <vulkan/vulkan.h>

/* super load/reset */
void load_VK_EXTENSION_SUBSET(VkInstance instance, PFN_vkGetInstanceProcAddr getInstanceProcAddr, VkDevice device, PFN_vkGetDeviceProcAddr getDeviceProcAddr);
void reset_VK_EXTENSION_SUBSET();

/* loaders */
]]..loaders_header..[[

]])

fsource:write("/* based on VK_HEADER_VERSION "..vk_version.." */\n"..header..[[


/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include <assert.h>
#include "]]..outfilename..[[.hpp"

/* loaders */
]]..loaders_source..[[

/* super load/reset */
void load_VK_EXTENSION_SUBSET(VkInstance instance, PFN_vkGetInstanceProcAddr getInstanceProcAddr, VkDevice device, PFN_vkGetDeviceProcAddr getDeviceProcAddr) {
]]..loaders_all..[[
}
void reset_VK_EXTENSION_SUBSET() {
]]..reset_all..[[
}

]])




end

do
  generate("extensions_vk",
    "/* auto generated by extensions_vk.lua */",
    extensionSubset)
end

