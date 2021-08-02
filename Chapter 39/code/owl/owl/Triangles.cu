// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
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

#include "Triangles.h"
#include "Context.h"

namespace owl {

  __device__ static float atomicMax(float* address, float val)
  {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
      assumed = old;
      old = ::atomicCAS(address_as_i, assumed,
                        __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
  }
  
  __device__ static float atomicMin(float* address, float val)
  {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
      assumed = old;
      old = ::atomicCAS(address_as_i, assumed,
                        __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
  }

  /*! device kernel to compute bounding box ov vertex array (and thus,
      bbox of triangle mesh, for motion blur (which for instances
      requies knowing the bboxes of its objects */
  __global__ void computeBoundsOfVertices(box3f *d_bounds,
                                          const void *d_vertices,
                                          size_t count,
                                          size_t stride,
                                          size_t offset)
  {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= count) return;

    const uint8_t *ptr = (const uint8_t *)d_vertices;
    ptr += tid*stride;
    ptr += offset;

    vec3f vtx = *(const vec3f*)ptr;
    atomicMin(&d_bounds->lower.x,vtx.x);
    atomicMin(&d_bounds->lower.y,vtx.y);
    atomicMin(&d_bounds->lower.z,vtx.z);
    atomicMax(&d_bounds->upper.x,vtx.x);
    atomicMax(&d_bounds->upper.y,vtx.y);
    atomicMax(&d_bounds->upper.z,vtx.z);
  }
                                          
  // ------------------------------------------------------------------
  // TrianglesGeomType
  // ------------------------------------------------------------------
  
  TrianglesGeomType::TrianglesGeomType(Context *const context,
                                       size_t varStructSize,
                                       const std::vector<OWLVarDecl> &varDecls)
    : GeomType(context,varStructSize,varDecls)
  {}

  // ------------------------------------------------------------------
  // TrianglesGeom::DeviceData
  // ------------------------------------------------------------------
  
  TrianglesGeom::DeviceData::DeviceData(const DeviceContext::SP &device)
    : Geom::DeviceData(device)
  {}
  
  
  // ------------------------------------------------------------------
  // TrianglesGeom
  // ------------------------------------------------------------------
  
  TrianglesGeom::TrianglesGeom(Context *const context,
                               GeomType::SP geometryType)
    : Geom(context,geometryType)
  {}
  
  /*! pretty-print */
  std::string TrianglesGeom::toString() const
  {
    return "TrianglesGeom";
  }

  /*! call a cuda kernel that computes the bounds of the vertex buffers */
  void TrianglesGeom::computeBounds(box3f bounds[2])
  {
    assert(vertex.buffers.size() == 1 || vertex.buffers.size() == 2);

    int numThreads = 1024;
    int numBlocks = int((vertex.count + numThreads - 1) / numThreads);

    DeviceContext::SP device = context->getDevice(0);
    assert(device);
    SetActiveGPU forLifeTime(device);
      
    DeviceMemory d_bounds;
    d_bounds.alloc(2*sizeof(box3f));
    bounds[0] = bounds[1] = box3f();
    d_bounds.upload(bounds);
    computeBoundsOfVertices<<<numBlocks,numThreads>>>
      (((box3f*)d_bounds.get())+0,
       vertex.buffers[0]->getPointer(device),
       vertex.count,vertex.stride,vertex.offset);
    if (vertex.buffers.size() == 2)
      computeBoundsOfVertices<<<numBlocks,numThreads>>>
        (((box3f*)d_bounds.get())+1,
         vertex.buffers[1]->getPointer(device),
         vertex.count,vertex.stride,vertex.offset);
    CUDA_SYNC_CHECK();
    d_bounds.download(&bounds[0]);
    d_bounds.free();
    CUDA_SYNC_CHECK();
    if (vertex.buffers.size() == 1)
      bounds[1] = bounds[0];
  }

  RegisteredObject::DeviceData::SP TrianglesGeom::createOn(const DeviceContext::SP &device) 
  { return std::make_shared<DeviceData>(device); }


  /*! set the vertex array (if vector size is 1), or set/enable
    motion blur via multiple time steps, if vector size >= 0 */
  void TrianglesGeom::setVertices(const std::vector<Buffer::SP> &vertexArrays,
                                  /*! the number of vertices in each time step */
                                  size_t count,
                                  size_t stride,
                                  size_t offset)
  {
    vertex.buffers = vertexArrays;
    vertex.count   = count;
    vertex.stride  = stride;
    vertex.offset  = offset;

    for (auto device : context->getDevices()) {
      DeviceData &dd = getDD(device);
      dd.vertexPointers.clear();
      for (auto va : vertexArrays)
        dd.vertexPointers.push_back((CUdeviceptr)va->getPointer(device));
    }
  }
  
  void TrianglesGeom::setIndices(Buffer::SP indices,
                                 size_t count,
                                 size_t stride,
                                 size_t offset)
  {
    index.buffer = indices;
    index.count  = count;
    index.stride = stride;
    index.offset = offset;
    
    for (auto device : context->getDevices()) {
      DeviceData &dd = getDD(device);
      dd.indexPointer = (CUdeviceptr)indices->getPointer(device);
    }
  }

} // ::owl
