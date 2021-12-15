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

#include "Object.h"

namespace owl {

  // ------------------------------------------------------------------
  // non class specific and global stuff
  // ------------------------------------------------------------------
  
  /*! atomic counter that always describes the next not yet used
    unique ID, which we can use to fill in the Object::uniquID
    values */
  std::atomic<uint64_t> Object::nextAvailableID;

  
  /*! returns number of bytes for given data type (where applicable) */
  size_t sizeOf(OWLDataType type)
  {
    if ((size_t)type >= (size_t)OWL_USER_TYPE_BEGIN)
      return (size_t)type - (size_t)OWL_USER_TYPE_BEGIN;
        
    switch(type) {
      
    case OWL_BOOL:
      return sizeof(bool);
    case OWL_BOOL2:
      return 2*sizeof(bool);
    case OWL_BOOL3:
      return 3*sizeof(bool);
    case OWL_BOOL4:
      return 4*sizeof(bool);
      
    case OWL_UCHAR:
      return sizeof(uint8_t);
    case OWL_UCHAR2:
      return 2*sizeof(uint8_t);
    case OWL_UCHAR3:
      return 3*sizeof(uint8_t);
    case OWL_UCHAR4:
      return 4*sizeof(uint8_t);
      
    case OWL_CHAR:
      return sizeof(int8_t);
    case OWL_CHAR2:
      return 2*sizeof(int8_t);
    case OWL_CHAR3:
      return 3*sizeof(int8_t);
    case OWL_CHAR4:
      return 4*sizeof(int8_t);
      
    case OWL_USHORT:
      return sizeof(uint16_t);
    case OWL_USHORT2:
      return 2*sizeof(uint16_t);
    case OWL_USHORT3:
      return 3*sizeof(uint16_t);
    case OWL_USHORT4:
      return 4*sizeof(uint16_t);
      
    case OWL_SHORT:
      return sizeof(int16_t);
    case OWL_SHORT2:
      return 2*sizeof(int16_t);
    case OWL_SHORT3:
      return 3*sizeof(int16_t);
    case OWL_SHORT4:
      return 4*sizeof(int16_t);
      
    case OWL_INT:
      return sizeof(int32_t);
    case OWL_INT2:
      return 2*sizeof(int32_t);
    case OWL_INT3:
      return 3*sizeof(int32_t);
    case OWL_INT4:
      return 4*sizeof(int32_t);
      
    case OWL_UINT:
      return sizeof(uint32_t);
    case OWL_UINT2:
      return 2*sizeof(uint32_t);
    case OWL_UINT3:
      return 3*sizeof(uint32_t);
    case OWL_UINT4:
      return 4*sizeof(uint32_t);
      
    case OWL_LONG:
      return sizeof(int64_t);
    case OWL_LONG2:
      return 2*sizeof(int64_t);
    case OWL_LONG3:
      return 3*sizeof(int64_t);
    case OWL_LONG4:
      return 4*sizeof(int64_t);
      
    case OWL_ULONG:
      return sizeof(uint64_t);
    case OWL_ULONG2:
      return 2*sizeof(uint64_t);
    case OWL_ULONG3:
      return 3*sizeof(uint64_t);
    case OWL_ULONG4:
      return 4*sizeof(uint64_t);
      
    case OWL_FLOAT:
      return sizeof(float);
    case OWL_FLOAT2:
      return 2*sizeof(float);
    case OWL_FLOAT3:
      return 3*sizeof(float);
    case OWL_FLOAT4:
      return 4*sizeof(float);

    case OWL_AFFINE3F:
      return sizeof(affine3f);

    case OWL_BUFFER_SIZE:
      return sizeof(size_t);
      
    case OWL_BUFFER:
      //      return sizeof();
      OWL_RAISE("device code for OWL_BUFFER type not yet implemented");
    case OWL_BUFFER_POINTER:
      return sizeof(void *);
    case OWL_GROUP:
      return sizeof(OptixTraversableHandle);
    case OWL_DEVICE:
      return sizeof(int32_t);
    default:
      OWL_RAISE(std::string(__PRETTY_FUNCTION__)
                +": not yet implemented for type #"
                +std::to_string((int)type));
      return 0;
    }
  }

  
  /*! convert a OWLDataType enum into a strict that represents the name of that type */
  std::string typeToString(OWLDataType type)
  {
    if (type >= OWL_USER_TYPE_BEGIN)
      return "(user defined type)";
    switch(type) {
      
      // ------------------------------------------------------------------
      // bool
      // ------------------------------------------------------------------
    case OWL_BOOL:
      return "bool";
    case OWL_BOOL2:
      return "bool2";
    case OWL_BOOL3:
      return "bool3";
    case OWL_BOOL4:
      return "bool4";
      
      // ------------------------------------------------------------------
      // 8 bit
      // ------------------------------------------------------------------
    case OWL_CHAR:
      return "char";
    case OWL_CHAR2:
      return "char2";
    case OWL_CHAR3:
      return "char3";
    case OWL_CHAR4:
      return "char4";
      
    case OWL_UCHAR:
      return "uchar";
    case OWL_UCHAR2:
      return "uchar2";
    case OWL_UCHAR3:
      return "uchar3";
    case OWL_UCHAR4:
      return "uchar4";
      
      // ------------------------------------------------------------------
      // 16 bit
      // ------------------------------------------------------------------
    case OWL_SHORT:
      return "short";
    case OWL_SHORT2:
      return "short2";
    case OWL_SHORT3:
      return "short3";
    case OWL_SHORT4:
      return "short4";
      
    case OWL_USHORT:
      return "ushort";
    case OWL_USHORT2:
      return "ushort2";
    case OWL_USHORT3:
      return "ushort3";
    case OWL_USHORT4:
      return "ushort4";
      
      // ------------------------------------------------------------------
      // 32 bit
      // ------------------------------------------------------------------
    case OWL_INT:
      return "int";
    case OWL_INT2:
      return "int2";
    case OWL_INT3:
      return "int3";
    case OWL_INT4:
      return "int4";
      
    case OWL_UINT:
      return "uint";
    case OWL_UINT2:
      return "uint2";
    case OWL_UINT3:
      return "uint3";
    case OWL_UINT4:
      return "uint4";
      
    case OWL_FLOAT:
      return "float";
    case OWL_FLOAT2:
      return "float2";
    case OWL_FLOAT3:
      return "float3";
    case OWL_FLOAT4:
      return "float4";
      
      // ------------------------------------------------------------------
      // 64 bit
      // ------------------------------------------------------------------
    case OWL_LONG:
      return "long";
    case OWL_LONG2:
      return "long2";
    case OWL_LONG3:
      return "long3";
    case OWL_LONG4:
      return "long4";
      
    case OWL_ULONG:
      return "ulong";
    case OWL_ULONG2:
      return "ulong2";
    case OWL_ULONG3:
      return "ulong3";
    case OWL_ULONG4:
      return "ulong4";
      
      // ------------------------------------------------------------------
      // other copable
      // ------------------------------------------------------------------
    case OWL_AFFINE3F:
      return "affine3f";
      
      // ------------------------------------------------------------------
      // "meta" types
      // ------------------------------------------------------------------
    case OWL_BUFFER:
      return "OWLBuffer";
    case OWL_BUFFER_POINTER:
      return "OWLBufferPointer";
    case OWL_BUFFER_SIZE:
      return "OWLBufferSize";
    case OWL_BUFFER_ID:
      return "OWLBufferID";
    case OWL_GROUP:
      return "OWLGroup";
    case OWL_DEVICE:
      return "OWLDevice";
    case OWL_TEXTURE:
      return "OWLTexture";
    default:
      if ((size_t)type >= (size_t)OWL_USER_TYPE_BEGIN)
        return "OWL_USER_TYPE(sz="
          +std::to_string((size_t)type-(size_t)OWL_USER_TYPE_BEGIN)+")";
      else
        OWL_RAISE(std::string(__PRETTY_FUNCTION__)
                  +": not yet implemented for type #"
                  +std::to_string((int)type));
      return "";
    }
  }


  
  // ------------------------------------------------------------------
  // Object::DeviceData
  // ------------------------------------------------------------------
  
  /*! creates the device-specific data for this group */
  Object::DeviceData::SP Object::createOn(const DeviceContext::SP &device)
  {
    return std::make_shared<DeviceData>(device);
  }
  
  /*! creates the actual device data for all devies,by calling \see
    createOn() for each device */
  void Object::createDeviceData(const std::vector<DeviceContext::SP> &devices)
  {
    if (!deviceData.empty())
      OWL_RAISE
        ("trying to create device data on object "+toString()
         +", but it already exists!?");
    assert(deviceData.empty());
    for (auto device : devices)
      deviceData.push_back(createOn(device));
  }
  
  // ------------------------------------------------------------------
  // Object
  // ------------------------------------------------------------------
  
  Object::Object()
    : uniqueID(nextAvailableID++)
  {}

  std::string Object::toString() const
  { return "Object"; }




  // ------------------------------------------------------------------
  // Object
  // ------------------------------------------------------------------

  /*! pretty-printer, for printf-debugging */
  std::string ContextObject::toString() const 
  {
    return "ContextObject";
  }
  
} // ::owl
