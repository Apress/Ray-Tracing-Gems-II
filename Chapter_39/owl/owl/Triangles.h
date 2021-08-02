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

#pragma once

#include "Geometry.h"

namespace owl {

  /*! a geometry *type* that uses triangular primitives, and that
      captures the anyhit and closesthit programs, variable types, SBT
      layout, etc, associated with all instances of this type */
  struct TrianglesGeomType : public GeomType {
    typedef std::shared_ptr<TrianglesGeomType> SP;
    
    TrianglesGeomType(Context *const context,
                      size_t varStructSize,
                      const std::vector<OWLVarDecl> &varDecls);
    
    /*! pretty-print */
    std::string toString() const override { return "TriangleGeomType"; }
    
    std::shared_ptr<Geom> createGeom() override;
  };

  /*! an actual *instance* of a given triangle-primitmives type; this
      actual trianlge mesh uses the programs and SBT data associated
      with its associated TrianglesGeomType, and will "instantiate"
      these with the vertex arrays, index array, etc, specified in
      this Geom */
  struct TrianglesGeom : public Geom {

    typedef std::shared_ptr<TrianglesGeom> SP;

    /*! any device-specific data, such as optix handles, cuda device
        pointers, etc */
    struct DeviceData : public Geom::DeviceData {
      DeviceData(const DeviceContext::SP &device);

      /*! this is a *vector* of vertex arrays, for motion blur
          purposes. ie, for static meshes only one entry is used, for
          motion blur two (and eventually, maybe more) will be used */
      std::vector<CUdeviceptr> vertexPointers;

      /*! device poiner to array of indices - the memory for the
          indices will live in some sort of buffer; this only points
          to that buffer */
      CUdeviceptr indexPointer  = (CUdeviceptr)0;
    };

    /*! constructor - create a new (as yet without vertices, indices,
        etc) instance of given triangles geom type */
    TrianglesGeom(Context *const context,
                  GeomType::SP geometryType);

    /*! creates the device-specific data for this group */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override;
    /*! creates the device-specific data for this group */

    /*! get reference to given device-specific data for this object */
    inline DeviceData &getDD(const DeviceContext::SP &device) const;

    /*! set the vertex array (if vector size is 1), or set/enable
        motion blur via multiple time steps, if vector size >= 0 */
    void setVertices(const std::vector<Buffer::SP> &vertices,
                     /*! the number of vertices in each time step */
                     size_t count,
                     size_t stride,
                     size_t offset);

    /*! set the index buffer; this remains one buffer even if motion blur is enabled. */
    void setIndices(Buffer::SP indices,
                    size_t count,
                    size_t stride,
                    size_t offset);

    /*! call a cuda kernel that computes the bounds of the vertex buffers */
    void computeBounds(box3f bounds[2]);

    /*! pretty-print */
    std::string toString() const override;

    struct {
      size_t count  = 0;
      size_t stride = 0;
      size_t offset = 0;
      Buffer::SP buffer;
    } index;
    struct {
      size_t count  = 0;
      size_t stride = 0;
      size_t offset = 0;
      std::vector<Buffer::SP> buffers;
    } vertex;
  };

  // ------------------------------------------------------------------
  // implementation section
  // ------------------------------------------------------------------
  
  /*! get reference to given device-specific data for this object */
  inline TrianglesGeom::DeviceData &TrianglesGeom::getDD(const DeviceContext::SP &device) const
  {
    assert(device->ID < (int)deviceData.size());
    return deviceData[device->ID]->as<TrianglesGeom::DeviceData>();
  }
  
} // ::owl
