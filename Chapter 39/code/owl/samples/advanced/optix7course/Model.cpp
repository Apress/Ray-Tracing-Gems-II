// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "Model.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

//std
#include <set>

namespace std {
  inline bool operator<(const tinyobj::index_t &a,
                        const tinyobj::index_t &b)
  {
    if (a.vertex_index < b.vertex_index) return true;
    if (a.vertex_index > b.vertex_index) return false;
    
    if (a.normal_index < b.normal_index) return true;
    if (a.normal_index > b.normal_index) return false;
    
    if (a.texcoord_index < b.texcoord_index) return true;
    if (a.texcoord_index > b.texcoord_index) return false;
    
    return false;
  }
}

/*! \namespace osc - Optix Siggraph Course */
namespace osc {
  


  /*! find vertex with given position, normal, texcoord, and return
      its vertex ID, or, if it doesn't exit, add it to the mesh, and
      its just-created index */
  int addVertex(TriangleMesh *mesh,
                tinyobj::attrib_t &attributes,
                const tinyobj::index_t &idx,
                std::map<tinyobj::index_t,int> &knownVertices)
  {
    if (knownVertices.find(idx) != knownVertices.end())
      return knownVertices[idx];

    const vec3f *vertex_array   = (const vec3f*)attributes.vertices.data();
    const vec3f *normal_array   = (const vec3f*)attributes.normals.data();
    const vec2f *texcoord_array = (const vec2f*)attributes.texcoords.data();
    
    int newID = (int)mesh->vertex.size();
    knownVertices[idx] = newID;

    mesh->vertex.push_back(vertex_array[idx.vertex_index]);
    if (idx.normal_index >= 0) {
      while (mesh->normal.size() < mesh->vertex.size())
        mesh->normal.push_back(normal_array[idx.normal_index]);
    }
    if (idx.texcoord_index >= 0) {
      while (mesh->texcoord.size() < mesh->vertex.size())
        mesh->texcoord.push_back(texcoord_array[idx.texcoord_index]);
    }
    
    return newID;
  }

  /*! load a texture (if not already loaded), and return its ID in the
      model's textures[] vector. Textures that could not get loaded
      return -1 */
  int loadTexture(Model *model,
                  std::map<std::string,int> &knownTextures,
                  const std::string &inFileName,
                  const std::string &modelPath)
  {
    if (inFileName == "")
      return -1;
    
    if (knownTextures.find(inFileName) != knownTextures.end())
      return knownTextures[inFileName];

    std::string fileName = inFileName;
    // first, fix backspaces:
    for (auto &c : fileName)
      if (c == '\\') c = '/';
    fileName = modelPath+"/"+fileName;

    vec2i res;
    int   comp;
    unsigned char* image = stbi_load(fileName.c_str(),
                                     &res.x, &res.y, &comp, STBI_rgb_alpha);
    int textureID = -1;
    if (image) {
      textureID = (int)model->textures.size();
      Texture *texture = new Texture;
      texture->resolution = res;
      texture->pixel      = (uint32_t*)image;

      /* iw - actually, it seems that stbi loads the pictures
         mirrored along the y axis - mirror them here */
      for (int y=0;y<res.y/2;y++) {
        uint32_t *line_y = texture->pixel + y * res.x;
        uint32_t *mirrored_y = texture->pixel + (res.y-1-y) * res.x;
        for (int x=0;x<res.x;x++) {
          std::swap(line_y[x],mirrored_y[x]);
        }
      }
      
      model->textures.push_back(texture);
    } else {
      std::cout << OWL_TERMINAL_RED
                << "Could not load texture from " << fileName << "!"
                << OWL_TERMINAL_DEFAULT << std::endl;
    }
    
    knownTextures[inFileName] = textureID;
    return textureID;
  }
  
  Model *loadOBJ(const std::string &objFile)
  {
    Model *model = new Model;

    const std::string modelDir
      = objFile.substr(0,objFile.rfind('/')+1);
    
    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err = "";

    bool readOK
      = tinyobj::LoadObj(&attributes,
                         &shapes,
                         &materials,
                         &err,
						 &err,
                         objFile.c_str(),
                         modelDir.c_str(),
                         /* triangulate */true);
    if (!readOK) {
      throw std::runtime_error("Could not read OBJ model from "+objFile+" : "+err);
    }

    if (materials.empty())
      throw std::runtime_error("could not parse materials ...");

    std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;
    for (int shapeID=0;shapeID<(int)shapes.size();shapeID++) {
      tinyobj::shape_t &shape = shapes[shapeID];

      std::set<int> materialIDs;
      for (auto faceMatID : shape.mesh.material_ids)
        materialIDs.insert(faceMatID);
      
      std::map<tinyobj::index_t,int> knownVertices;
      std::map<std::string,int>      knownTextures;
      
      for (int materialID : materialIDs) {
        TriangleMesh *mesh = new TriangleMesh;
        
        for (size_t faceID=0;faceID<shape.mesh.material_ids.size();faceID++) {
          if (shape.mesh.material_ids[faceID] != materialID) continue;
          if (shape.mesh.num_face_vertices[faceID] != 3)
            throw std::runtime_error("not properly tessellated");
          tinyobj::index_t idx0 = shape.mesh.indices[3*faceID+0];
          tinyobj::index_t idx1 = shape.mesh.indices[3*faceID+1];
          tinyobj::index_t idx2 = shape.mesh.indices[3*faceID+2];
          
          vec3i idx(addVertex(mesh, attributes, idx0, knownVertices),
                    addVertex(mesh, attributes, idx1, knownVertices),
                    addVertex(mesh, attributes, idx2, knownVertices));
          mesh->index.push_back(idx);
          if (materialID < 0) {
            mesh->diffuse = vec3f(1,0,0);
            mesh->diffuseTextureID = -1;
          } else {
            mesh->diffuse = (const vec3f&)materials[materialID].diffuse;
            mesh->diffuseTextureID = loadTexture(model,
                                                 knownTextures,
                                                 materials[materialID].diffuse_texname,
                                                 modelDir);
          }
        }

        if (mesh->vertex.empty())
          delete mesh;
        else {
          // just for sanity's sake:
          if (mesh->texcoord.size() > 0)
            mesh->texcoord.resize(mesh->vertex.size());
          // just for sanity's sake:
          if (mesh->normal.size() > 0)
            mesh->normal.resize(mesh->vertex.size());

          for (auto idx : mesh->index) {
            if (idx.x < 0 || idx.x >= (int)mesh->vertex.size() ||
                idx.y < 0 || idx.y >= (int)mesh->vertex.size() ||
                idx.z < 0 || idx.z >= (int)mesh->vertex.size())
              { PING; PRINT(idx); PRINT(mesh->vertex.size()); }
          }
          model->meshes.push_back(mesh);
        }
      }
    }

    // of course, you should be using tbb::parallel_for for stuff
    // like this:
    for (auto mesh : model->meshes)
      for (auto vtx : mesh->vertex)
        model->bounds.extend(vtx);
    
    return model;
  }
}
