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

#include "Variable.h"
#include "Context.h"
#include "InstanceGroup.h"
// device buffer representation that we'll write for Buffer variables
#include "owl/owl_device_buffer.h"
 
namespace owl { 
  
  /*! throw an exception that the type the user tried to set doesn't
    math the type he/she declared*/
  void Variable::mismatchingType(const std::string &attemptedType)
  {
    assert(varDecl);
    throw std::runtime_error
      ("trying to set variable to value of wrong type.\n"
       "note: name of variable is " + std::string(varDecl->name) + "\n"
       "note: type of variable is " + typeToString(varDecl->type) + "\n"
       "note: type of value attempted to be set is " + attemptedType
       );
  }

  void Variable::set(const std::shared_ptr<Buffer>  &value)
  { mismatchingType("Buffer"); }
  void Variable::set(const std::shared_ptr<Group>   &value)
  { mismatchingType("Group"); }
  void Variable::set(const std::shared_ptr<Texture> &value)
  { mismatchingType("Texture"); }
    
  void Variable::setRaw(const void *ptr)
  { mismatchingType("void*"); }


  void Variable::set(const bool &value)
  { mismatchingType("bool"); }
  void Variable::set(const vec2b &value)
  { mismatchingType("bool2"); }
  void Variable::set(const vec3b &value)
  { mismatchingType("bool3"); }
  void Variable::set(const vec4b &value)
  { mismatchingType("bool4"); }

  
  void Variable::set(const int8_t &value)
  { mismatchingType("char"); }
  void Variable::set(const vec2c &value)
  { mismatchingType("char2"); }
  void Variable::set(const vec3c &value)
  { mismatchingType("char3"); }
  void Variable::set(const vec4c &value)
  { mismatchingType("char4"); }
    
  void Variable::set(const uint8_t &value)
  { mismatchingType("uchar"); }
  void Variable::set(const vec2uc &value)
  { mismatchingType("uchar2"); }
  void Variable::set(const vec3uc &value)
  { mismatchingType("uchar3"); }
  void Variable::set(const vec4uc &value)
  { mismatchingType("uchar4"); }
    


  void Variable::set(const int16_t &value)
  { mismatchingType("short"); }
  void Variable::set(const vec2s &value)
  { mismatchingType("short2"); }
  void Variable::set(const vec3s &value)
  { mismatchingType("short3"); }
  void Variable::set(const vec4s &value)
  { mismatchingType("short4"); }
    
  void Variable::set(const uint16_t &value)
  { mismatchingType("ushort"); }
  void Variable::set(const vec2us &value)
  { mismatchingType("ushort2"); }
  void Variable::set(const vec3us &value)
  { mismatchingType("ushort3"); }
  void Variable::set(const vec4us &value)
  { mismatchingType("ushort4"); }
    

  void Variable::set(const int32_t &value)
  { mismatchingType("int"); }
  void Variable::set(const vec2i &value)
  { mismatchingType("int2"); }
  void Variable::set(const vec3i &value)
  { mismatchingType("int3"); }
  void Variable::set(const vec4i &value)
  { mismatchingType("int4"); }
    
  void Variable::set(const uint32_t &value)
  { mismatchingType("uint"); }
  void Variable::set(const vec2ui &value)
  { mismatchingType("uint2"); }
  void Variable::set(const vec3ui &value)
  { mismatchingType("uint3"); }
  void Variable::set(const vec4ui &value)
  { mismatchingType("uint4"); }
    
  void Variable::set(const int64_t &value)
  { mismatchingType("long"); }
  void Variable::set(const vec2l &value)
  { mismatchingType("long2"); }
  void Variable::set(const vec3l &value)
  { mismatchingType("long3"); }
  void Variable::set(const vec4l &value)
  { mismatchingType("long4"); }
    
  void Variable::set(const uint64_t &value)
  { mismatchingType("ulong"); }
  void Variable::set(const vec2ul &value)
  { mismatchingType("ulong2"); }
  void Variable::set(const vec3ul &value)
  { mismatchingType("ulong3"); }
  void Variable::set(const vec4ul &value)
  { mismatchingType("ulong4"); }
    
  void Variable::set(const float  &value)
  { mismatchingType("float"); }
  void Variable::set(const vec2f  &value)
  { mismatchingType("float2"); }
  void Variable::set(const vec3f  &value)
  { mismatchingType("float3"); }
  void Variable::set(const vec4f  &value)
  { mismatchingType("float4"); }

  void Variable::set(const double &value)
  { mismatchingType("double"); }
  void Variable::set(const vec2d  &value)
  { mismatchingType("double2"); }
  void Variable::set(const vec3d  &value)
  { mismatchingType("double3"); }
  void Variable::set(const vec4d  &value)
  { mismatchingType("double4"); }

  void Variable::set(const affine3f &value)
  { mismatchingType("affine3f"); }
  
  /*! Variable type for ray "user yypes". User types have a
      user-specified size in bytes, and get set by passing a pointer
      to 'raw' data that then gets copied in binary form */
  struct UserTypeVariable : public Variable
  {
    UserTypeVariable(const OWLVarDecl *const varDecl)
      : Variable(varDecl),
        data(/* actual size is 'type' - constant */varDecl->type - OWL_USER_TYPE_BEGIN)
    {}
    
    void setRaw(const void *ptr) override
    {
      memcpy(data.data(),ptr,data.size());
    }

    /*! writes the device specific representation of the given type */
    void writeToSBT(uint8_t *sbtEntry,
                    const DeviceContext::SP &device) const override
    {
      memcpy(sbtEntry,data.data(),data.size());
    }
    
    std::vector<uint8_t> data;
  };

  /*! Variable type for basic and compound-basic data types such as
      float, vec3f, etc */
  template<typename T>
  struct VariableT : public Variable {
    typedef std::shared_ptr<VariableT<T>> SP;

    VariableT(const OWLVarDecl *const varDecl)
      : Variable(varDecl)
    {}
    
    void set(const T &value) override { this->value = value; }

    /*! writes the device specific representation of the given type */
    void writeToSBT(uint8_t *sbtEntry,
                    const DeviceContext::SP &device) const override
    {
      *(T*)sbtEntry = value;
    }

    T value;
  };

  /*! Variable type that accepts owl buffer types, and on the
      device-side writes just the raw device pointer into the SBT */
  struct BufferPointerVariable : public Variable {
    typedef std::shared_ptr<BufferPointerVariable> SP;

    BufferPointerVariable(const OWLVarDecl *const varDecl)
      : Variable(varDecl)
    {}
    void set(const Buffer::SP &value) override { this->buffer = value; }

    /*! writes the device specific representation of the given type */
    void writeToSBT(uint8_t *sbtEntry,
                    const DeviceContext::SP &device) const override
    {
      const void *value
        = buffer
        ? buffer->getPointer(device)
        : nullptr;
      *(const void**)sbtEntry = value;
    }
    
    Buffer::SP buffer;
  };

  /*! Variable type that accepts owl buffer types, and on the
      device-side writes just the raw device pointer into the SBT */
  struct BufferSizeVariable : public Variable {
    typedef std::shared_ptr<BufferSizeVariable> SP;

    BufferSizeVariable(const OWLVarDecl *const varDecl)
      : Variable(varDecl)
    {}
    void set(const Buffer::SP &value) override { this->buffer = value; }

    /*! writes the device specific representation of the given type */
    void writeToSBT(uint8_t *sbtEntry,
                    const DeviceContext::SP &device) const override
    {
      const size_t value
        = buffer
        ? buffer->elementCount
        : 0ull;
      *(size_t*)sbtEntry = value;
    }

    Buffer::SP buffer;
  };

  /*! Variable type that accepts owl buffer types, and on the
      device-side writes just the raw device pointer into the SBT */
  struct BufferIDVariable : public Variable {
    typedef std::shared_ptr<BufferIDVariable> SP;

    BufferIDVariable(const OWLVarDecl *const varDecl)
      : Variable(varDecl)
    {}
    void set(const Buffer::SP &value) override { this->buffer = value; }

    /*! writes the device specific representation of the given type */
    void writeToSBT(uint8_t *sbtEntry,
                    const DeviceContext::SP &device) const override
    {
      const int32_t value
        = buffer
        ? buffer->ID
        : -1;
      *(int32_t*)sbtEntry = value;
    }

    Buffer::SP buffer;
  };

  /*! Fully-implicit Variable type that doesn't actually take _any_
      user data, but instead always writes the currently active
      device's device ID into the SBT */
  struct DeviceIndexVariable : public Variable {
    typedef std::shared_ptr<BufferPointerVariable> SP;

    DeviceIndexVariable(const OWLVarDecl *const varDecl)
      : Variable(varDecl)
    {}
    void set(const Buffer::SP &value) override
    {
      OWL_RAISE("cannot _set_ a device index variable; it is purely implicit");
    }

    /*! writes the device specific representation of the given type */
    void writeToSBT(uint8_t *sbtEntry,
                    const DeviceContext::SP &device) const override
    {
      *(int*)sbtEntry = device->ID;
    }
    
    Buffer::SP buffer;
  };
  
  /*! Buffer variable that takes owl buffer types, and on the device
      writes full device::Buffer types with size, pointer, etc */
  struct BufferVariable : public Variable {
    typedef std::shared_ptr<BufferVariable> SP;

    BufferVariable(const OWLVarDecl *const varDecl)
      : Variable(varDecl)
    {}
    void set(const Buffer::SP &value) override { this->buffer = value; }

    /*! writes the device specific representation of the given type */
    void writeToSBT(uint8_t *sbtEntry,
                    const DeviceContext::SP &device) const override
    {
      device::Buffer *devRep = (device::Buffer *)sbtEntry;
      if (!buffer) {
        devRep->data  = 0;
        devRep->count = 0;
        devRep->type  = OWL_INVALID_TYPE;
      } else {
        devRep->data  = (void *)buffer->getPointer(device);
        devRep->count = buffer->elementCount;
        devRep->type  = buffer->type;
      }
    }
    
    Buffer::SP buffer;
  };

  /*! Variable type that accepts owl Group types on the host, and
      writes the groups' respective OptixTraversableHandle into the
      SBT */
  struct GroupVariable : public Variable {
    typedef std::shared_ptr<GroupVariable> SP;

    GroupVariable(const OWLVarDecl *const varDecl)
      : Variable(varDecl)
    {}
    void set(const Group::SP &value) override
    {
      if (value && !std::dynamic_pointer_cast<InstanceGroup>(value))
        OWL_RAISE("OWL currently supports only instance groups to be passed to traversal; if you do want to trace rays into a single User or Triangle group, please put them into a single 'dummy' instance with jsut this one child and a identity transform");
      this->group = value;
    }

    /*! writes the device specific representation of the given type */
    void writeToSBT(uint8_t *sbtEntry,
                    const DeviceContext::SP &device) const override
    {
      const OptixTraversableHandle value
        = group
        ? group->getTraversable(device)//context->llo->groupGetTraversable(group->ID,deviceID)
        : 0;
      *(OptixTraversableHandle*)sbtEntry = value;
    }
    
    
    Group::SP group;
  };
  
  /*! Variable type that manages textures; accepting owl::Texture
      objects on the host, and writing their corresponding cuda
      texture obejct handles into the SBT */
  struct TextureVariable : public Variable {
    typedef std::shared_ptr<TextureVariable> SP;

    TextureVariable(const OWLVarDecl *const varDecl)
      : Variable(varDecl)
    {}
    void set(const Texture::SP &value) override
    {
      this->texture = value;
    }

    /*! writes the device specific representation of the given type */
    void writeToSBT(uint8_t *sbtEntry,
                    const DeviceContext::SP &device) const override
    {
      cudaTextureObject_t to = {};
      if (texture) {
        assert(device->ID < (int)texture->textureObjects.size());
        to = texture->textureObjects[device->ID];
      }
      *(cudaTextureObject_t*)sbtEntry = to;
    }
    
    Texture::SP texture;
  };
  
  /*! creates a variable type that matches the given variable
      declaration */
  Variable::SP Variable::createInstanceOf(const OWLVarDecl *decl)
  {
    assert(decl);
    assert(decl->name);
    if (decl->type >= OWL_USER_TYPE_BEGIN)
      return std::make_shared<UserTypeVariable>(decl);
    switch(decl->type) {

      // ------------------------------------------------------------------
      // bool
      // ------------------------------------------------------------------
    case OWL_BOOL:
      return std::make_shared<VariableT<bool>>(decl);
    case OWL_BOOL2:
      return std::make_shared<VariableT<vec2b>>(decl);
    case OWL_BOOL3:
      return std::make_shared<VariableT<vec3b>>(decl);
    case OWL_BOOL4:
      return std::make_shared<VariableT<vec4b>>(decl);

      // ------------------------------------------------------------------
      // 8 bit
      // ------------------------------------------------------------------
    case OWL_CHAR:
      return std::make_shared<VariableT<int8_t>>(decl);
    case OWL_CHAR2:
      return std::make_shared<VariableT<vec2c>>(decl);
    case OWL_CHAR3:
      return std::make_shared<VariableT<vec3c>>(decl);
    case OWL_CHAR4:
      return std::make_shared<VariableT<vec4c>>(decl);

    case OWL_UCHAR:
      return std::make_shared<VariableT<uint8_t>>(decl);
    case OWL_UCHAR2:
      return std::make_shared<VariableT<vec2uc>>(decl);
    case OWL_UCHAR3:
      return std::make_shared<VariableT<vec3uc>>(decl);
    case OWL_UCHAR4:
      return std::make_shared<VariableT<vec4uc>>(decl);

      // ------------------------------------------------------------------
      // 16 bit
      // ------------------------------------------------------------------
    case OWL_SHORT:
      return std::make_shared<VariableT<int16_t>>(decl);
    case OWL_SHORT2:
      return std::make_shared<VariableT<vec2s>>(decl);
    case OWL_SHORT3:
      return std::make_shared<VariableT<vec3s>>(decl);
    case OWL_SHORT4:
      return std::make_shared<VariableT<vec4s>>(decl);

    case OWL_USHORT:
      return std::make_shared<VariableT<uint16_t>>(decl);
    case OWL_USHORT2:
      return std::make_shared<VariableT<vec2us>>(decl);
    case OWL_USHORT3:
      return std::make_shared<VariableT<vec3us>>(decl);
    case OWL_USHORT4:
      return std::make_shared<VariableT<vec4us>>(decl);
      
      // ------------------------------------------------------------------
      // 32 bit
      // ------------------------------------------------------------------
    case OWL_INT:
      return std::make_shared<VariableT<int32_t>>(decl);
    case OWL_INT2:
      return std::make_shared<VariableT<vec2i>>(decl);
    case OWL_INT3:
      return std::make_shared<VariableT<vec3i>>(decl);
    case OWL_INT4:
      return std::make_shared<VariableT<vec4i>>(decl);

    case OWL_UINT:
      return std::make_shared<VariableT<uint32_t>>(decl);
    case OWL_UINT2:
      return std::make_shared<VariableT<vec2ui>>(decl);
    case OWL_UINT3:
      return std::make_shared<VariableT<vec3ui>>(decl);
    case OWL_UINT4:
      return std::make_shared<VariableT<vec4ui>>(decl);

    case OWL_FLOAT:
      return std::make_shared<VariableT<float>>(decl);
    case OWL_FLOAT2:
      return std::make_shared<VariableT<vec2f>>(decl);
    case OWL_FLOAT3:
      return std::make_shared<VariableT<vec3f>>(decl);
    case OWL_FLOAT4:
      return std::make_shared<VariableT<vec4f>>(decl);
      
      // ------------------------------------------------------------------
      // 64 bit
      // ------------------------------------------------------------------
    case OWL_LONG:
      return std::make_shared<VariableT<int64_t>>(decl);
    case OWL_LONG2:
      return std::make_shared<VariableT<vec2l>>(decl);
    case OWL_LONG3:
      return std::make_shared<VariableT<vec3l>>(decl);
    case OWL_LONG4:
      return std::make_shared<VariableT<vec4l>>(decl);

    case OWL_ULONG:
      return std::make_shared<VariableT<uint64_t>>(decl);
    case OWL_ULONG2:
      return std::make_shared<VariableT<vec2ul>>(decl);
    case OWL_ULONG3:
      return std::make_shared<VariableT<vec3ul>>(decl);
    case OWL_ULONG4:
      return std::make_shared<VariableT<vec4ul>>(decl);

    case OWL_DOUBLE:
      return std::make_shared<VariableT<double>>(decl);
    case OWL_DOUBLE2:
      return std::make_shared<VariableT<vec2d>>(decl);
    case OWL_DOUBLE3:
      return std::make_shared<VariableT<vec3d>>(decl);
    case OWL_DOUBLE4:
      return std::make_shared<VariableT<vec4d>>(decl);

    case OWL_AFFINE3F:
      return std::make_shared<VariableT<affine3f>>(decl);
      
      // ------------------------------------------------------------------
      // meta
      // ------------------------------------------------------------------
    case OWL_GROUP:
      return std::make_shared<GroupVariable>(decl);
    case OWL_TEXTURE:
      return std::make_shared<TextureVariable>(decl);
    case OWL_BUFFER:
      return std::make_shared<BufferVariable>(decl);
    case OWL_BUFFER_POINTER:
      return std::make_shared<BufferPointerVariable>(decl);
    case OWL_BUFFER_ID:
      return std::make_shared<BufferIDVariable>(decl);
    case OWL_BUFFER_SIZE:
      return std::make_shared<BufferSizeVariable>(decl);
    case OWL_DEVICE:
      return std::make_shared<DeviceIndexVariable>(decl);
    case OWL_INVALID_TYPE:
      throw std::runtime_error("Tried to create an instance of OWL_INVALID_TYPE");
    default:
      throw std::runtime_error(std::string(__PRETTY_FUNCTION__)
                               +": not yet implemented for type "
                               +typeToString(decl->type));
    }
  }
    
} // ::owl
