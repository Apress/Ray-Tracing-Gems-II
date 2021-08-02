/* Copyright (c) 2014-2019, NVIDIA CORPORATION. All rights reserved.
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

/**
  # namespace nvh::gltf

  These utilities are for loading glTF models in a
  canonical scene representation. From this representation
  you would create the appropriate 3D API resources (buffers
  and textures).
 
  ~~~ C++
  // Typical Usage
  // Load the GLTF Scene using TinyGLTF
 
  tinygltf::Model    gltfModel;
  tinygltf::TinyGLTF gltfContext;
  fileLoaded = gltfContext.LoadASCIIFromFile(&gltfModel, &error, &warn, m_filename);
 
  // Fill the data in the gltfScene
  gltfScene.getMaterials(tmodel);
  gltfScene.getDrawableNodes(tmodel, GltfAttributes::Normal | GltfAttributes::Texcoord_0);

  // Todo in App:
  //   create buffers for vertices and indices, from gltfScene.m_position, gltfScene.m_index
  //   create textures from images: using tinygltf directly
  //   create descriptorSet for material using directly gltfScene.m_materials
  ~~~

*/

#pragma once
#pragma once
#include "fileformats/tiny_gltf.h"
#include "nvmath/nvmath.h"
#include "nvmath/nvmath_glsltypes.h"
#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

using namespace nvmath;
#define KHR_LIGHTS_PUNCTUAL_EXTENSION_NAME "KHR_lights_punctual"
#define KHR_TEXTURE_TRANSFORM_EXTENSION_NAME "KHR_texture_transform"
#define KHR_MATERIALS_PBRSPECULARGLOSSINESS_EXTENSION_NAME "KHR_materials_pbrSpecularGlossiness"
#define KHR_MATERIALS_UNLIT_EXTENSION_NAME "KHR_materials_unlit"
#define KHR_MATERIALS_ANISOTROPY_EXTENSION_NAME "KHR_materials_anisotropy"

namespace nvh {

struct GltfMaterial
{
  int shadingModel{0};  // 0: metallic-roughness, 1: specular-glossiness

  // PbrMetallicRoughness
  vec4  pbrBaseColorFactor{1, 1, 1, 1};
  int   pbrBaseColorTexture{-1};
  float pbrMetallicFactor{1.f};
  float pbrRoughnessFactor{1.f};
  int   pbrMetallicRoughnessTexture{-1};

  // KHR_materials_pbrSpecularGlossiness
  vec4  khrDiffuseFactor{1, 1, 1, 1};
  int   khrDiffuseTexture{-1};
  vec3  khrSpecularFactor{1, 1, 1};
  float khrGlossinessFactor{1};
  int   khrSpecularGlossinessTexture{-1};

  int   emissiveTexture{-1};
  vec3  emissiveFactor{0, 0, 0};
  int   alphaMode{0};
  float alphaCutoff{0.5f};
  int   doubleSided{0};

  int   normalTexture{-1};
  float normalTextureScale{1.f};
  int   occlusionTexture{-1};
  float occlusionTextureStrength{1};

  // KHR_texture_transform : https://github.com/KhronosGroup/glTF/tree/master/extensions/2.0/Khronos/KHR_texture_transform
  mat3 uvTransform{1};

  // KHR_materials_unlit : https://github.com/KhronosGroup/glTF/tree/master/extensions/2.0/Khronos/KHR_materials_unlit
  int unlit{0};

  // KHR_materials_anisotropy
  float anisotropy{0};
  vec3  anisotropyDirection{1, 0, 0};
  int   anisotropyTexture{-1};
};


struct GltfNode
{
  nvmath::mat4f worldMatrix{1};
  int           primMesh{0};
};

struct GltfPrimMesh
{
  uint32_t firstIndex{0};
  uint32_t indexCount{0};
  uint32_t vertexOffset{0};
  uint32_t vertexCount{0};
  int      materialIndex{0};

  nvmath::vec3f posMin{0, 0, 0};
  nvmath::vec3f posMax{0, 0, 0};
};

struct GltfStats
{
  uint32_t nbCameras{0};
  uint32_t nbImages{0};
  uint32_t nbTextures{0};
  uint32_t nbMaterials{0};
  uint32_t nbSamplers{0};
  uint32_t nbNodes{0};
  uint32_t nbMeshes{0};
  uint32_t nbLights{0};
  uint32_t imageMem{0};
  uint32_t nbUniqueTriangles{0};
  uint32_t nbTriangles{0};
};

struct GltfCamera
{
  nvmath::mat4f worldMatrix{1};
  nvmath::vec3f eye{0, 0, 0};
  nvmath::vec3f center{0, 0, 0};
  nvmath::vec3f up{0, 1, 0};

  tinygltf::Camera cam;
};

struct GltfLight
{
  nvmath::mat4f   worldMatrix{1};
  tinygltf::Light light;
};


enum class GltfAttributes : uint8_t
{
  Position   = 0,
  Normal     = 1,
  Texcoord_0 = 2,
  Texcoord_1 = 4,
  Tangent    = 8,
  Color_0    = 16,
  //Joints_0   = 32, // #TODO - Add support for skinning
  //Weights_0  = 64,
};
using GltfAttributes_t = std::underlying_type_t<GltfAttributes>;

inline GltfAttributes operator|(GltfAttributes lhs, GltfAttributes rhs)
{
  return static_cast<GltfAttributes>(static_cast<GltfAttributes_t>(lhs) | static_cast<GltfAttributes_t>(rhs));
}

inline GltfAttributes operator&(GltfAttributes lhs, GltfAttributes rhs)
{
  return static_cast<GltfAttributes>(static_cast<GltfAttributes_t>(lhs) & static_cast<GltfAttributes_t>(rhs));
}

//--------------------------------------------------------------------------------------------------
// Class to convert gltfScene in simple draw-able format
//
struct GltfScene
{
  void importMaterials(const tinygltf::Model& tmodel);
  void importDrawableNodes(const tinygltf::Model& tmodel, GltfAttributes attributes);
  void computeSceneDimensions();
  void destroy();

  static GltfStats getStatistics(const tinygltf::Model& tinyModel);

  // Scene data
  std::vector<GltfMaterial> m_materials;   // Material for shading
  std::vector<GltfNode>     m_nodes;       // Drawable nodes, flat hierarchy
  std::vector<GltfPrimMesh> m_primMeshes;  // Primitive promoted to meshes
  std::vector<GltfCamera>   m_cameras;
  std::vector<GltfLight>    m_lights;

  // Attributes, all same length if valid
  std::vector<nvmath::vec3f> m_positions;
  std::vector<uint32_t>      m_indices;
  std::vector<nvmath::vec3f> m_normals;
  std::vector<nvmath::vec4f> m_tangents;
  std::vector<nvmath::vec2f> m_texcoords0;
  std::vector<nvmath::vec2f> m_texcoords1;
  std::vector<nvmath::vec4f> m_colors0;

  // #TODO - Adding support for Skinning
  //using vec4us = vector4<unsigned short>;
  //std::vector<vec4us>        m_joints0;
  //std::vector<nvmath::vec4f> m_weights0;

  // Size of the scene
  struct Dimensions
  {
    nvmath::vec3f min = nvmath::vec3f(std::numeric_limits<float>::max());
    nvmath::vec3f max = nvmath::vec3f(std::numeric_limits<float>::min());
    nvmath::vec3f size{0.f};
    nvmath::vec3f center{0.f};
    float         radius{0};
  } m_dimensions;


private:
  void          processNode(const tinygltf::Model& tmodel, int& nodeIdx, const nvmath::mat4f& parentMatrix);
  void          processMesh(const tinygltf::Model& tmodel, const tinygltf::Primitive& tmesh, GltfAttributes attributes);
  nvmath::mat4f getLocalMatrix(const tinygltf::Node& tnode);


  // Temporary data
  std::unordered_map<int, std::vector<uint32_t>> m_meshToPrimMeshes;
  std::vector<uint32_t>                          primitiveIndices32u;
  std::vector<uint16_t>                          primitiveIndices16u;
  std::vector<uint8_t>                           primitiveIndices8u;

  // Return a vector of data for a tinygltf::Value
  template <typename T>
  static inline std::vector<T> getVector(const tinygltf::Value& value)
  {
    std::vector<T> result{0};
    if(!value.IsArray())
      return result;
    result.resize(value.ArrayLen());
    for(int i = 0; i < value.ArrayLen(); i++)
    {
      result[i] = static_cast<T>(value.Get(i).IsNumber() ? value.Get(i).Get<double>() : value.Get(i).Get<int>());
    }
    return result;
  }

  // Appending to \p attribVec, all the values of \p attribName
  // Return false if the attribute is missing
  template <typename T>
  bool getAttribute(const tinygltf::Model& tmodel, const tinygltf::Primitive& primitive, std::vector<T>& attribVec, const std::string& attribName)
  {
    if(primitive.attributes.find(attribName) == primitive.attributes.end())
      return false;

    // Retrieving the data of the attribute
    const auto& accessor = tmodel.accessors[primitive.attributes.find(attribName)->second];
    const auto& bufView  = tmodel.bufferViews[accessor.bufferView];
    const auto& buffer   = tmodel.buffers[bufView.buffer];
    const auto  bufData  = reinterpret_cast<const T*>(&(buffer.data[accessor.byteOffset + bufView.byteOffset]));
    const auto  nbElems  = accessor.count;

    // Supporting KHR_mesh_quantization
    assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);

    // Copying the attributes
    if(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
    {
      if(bufView.byteStride == 0)
      {
        attribVec.insert(attribVec.end(), bufData, bufData + nbElems);
      }
      else
      {
        // With stride, need to add one by one the element
        auto bufferByte = reinterpret_cast<const uint8_t*>(bufData);
        for(size_t i = 0; i < nbElems; i++)
        {
          attribVec.push_back(*reinterpret_cast<const T*>(bufferByte));
          bufferByte += bufView.byteStride;
        }
      }
    }
    else
    {
      // The component is smaller than float and need to be converted

      // VEC3 or VEC4
      int nbComponents = accessor.type == TINYGLTF_TYPE_VEC2 ? 2 : (accessor.type == TINYGLTF_TYPE_VEC3) ? 3 : 4;
      // UNSIGNED_BYTE or UNSIGNED_SHORT
      int strideComponent = accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE ? 1 : 2;

      size_t byteStride = bufView.byteStride > 0 ? bufView.byteStride : nbComponents * strideComponent;
      auto   bufferByte = reinterpret_cast<const uint8_t*>(bufData);
      for(size_t i = 0; i < nbElems; i++)
      {
        T vecValue;

        auto bufferByteData = bufferByte;
        for(int c = 0; c < nbComponents; c++)
        {
          float value = *reinterpret_cast<const float*>(bufferByteData);
          switch(accessor.componentType)
          {
            case TINYGLTF_COMPONENT_TYPE_BYTE:
              vecValue[c] = std::max(value / 127.0f, -1.f);
              break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
              vecValue[c] = value / 255.0f;
              break;
            case TINYGLTF_COMPONENT_TYPE_SHORT:
              vecValue[c] = std::max(value / 32767.0f, -1.f);
              break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
              vecValue[c] = value / 65535.0f;
              break;
            default:
              assert(!"KHR_mesh_quantization unsupported format");
              break;
          }
          bufferByteData += strideComponent;
        }
        bufferByte += byteStride;
        attribVec.push_back(vecValue);
      }
    }


    return true;
  }

  inline bool hasExtension(const tinygltf::ExtensionMap& extensions, const std::string& name)
  {
    return extensions.find(name) != extensions.end();
  }
  void computeCamera();
  void checkRequiredExtensions(const tinygltf::Model& tmodel);
};


}  // namespace nvh
