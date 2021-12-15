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

#include "Group.h"

namespace owl {

  /*! a OWL Group / BVH over instances (i.e., a IAS) */
  struct InstanceGroup : public Group {
    typedef std::shared_ptr<InstanceGroup> SP;
    
    /*! any device-specific data, such as optix handles, cuda device
      pointers, etc */
    struct DeviceData : public Group::DeviceData {
      typedef std::shared_ptr<DeviceData> SP;
      
      /*! constructor */
      DeviceData(const DeviceContext::SP &device);
      
      DeviceMemory optixInstanceBuffer;

      /*! if we use motion blur, this is used to store all the motoin transforms */
      DeviceMemory motionTransformsBuffer;
      DeviceMemory motionAABBsBuffer;
      DeviceMemory outputBuffer;
    };
    
    /*! construct with given array of groups - transforms can be specified later */
    InstanceGroup(Context *const context,
                  size_t numChildren,
                  Group::SP *groups,
                  unsigned int buildFlags);

    /*! pretty-printer, for printf-debugging */
    std::string toString() const override;
    
    /*! set given child to given group */
    void setChild(size_t childID, Group::SP child);
                  
    /*! set transformation matrix of given child */
    void setTransform(size_t childID, const affine3f &xfm);

    /*! set transformation matrix of given child */
    void setTransforms(uint32_t timeStep,
                       const float *floatsForThisStimeStep,
                       OWLMatrixFormat matrixFormat);

    /* set instance IDs to use for the children - MUST be an array of
       children.size() items */
    void setInstanceIDs(const uint32_t *instanceIDs);

    /* set visibility masks to use for the children - MUST be an array of
       children.size() items */
    void setVisibilityMasks(const uint8_t *visibilityMasks);
      
    void buildAccel() override;
    void refitAccel() override;

    /*! creates the device-specific data for this group */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override;
    
    /*! get reference to given device-specific data for this object */
    inline DeviceData &getDD(const DeviceContext::SP &device) const;

    template<bool FULL_REBUILD>
    void staticBuildOn(const DeviceContext::SP &device);
    template<bool FULL_REBUILD>
    void motionBlurBuildOn(const DeviceContext::SP &device);

    /*! return the SBT offset to use for this group - SBT offsets for
      instnace groups are always 0 */
    int getSBTOffset() const override { return 0; }
    
    /*! the list of children - note we do have to keep them both in
      the ll layer _and_ here for the refcounting to work; the
      transforms are only stored once, on the ll layer */
    std::vector<Group::SP>  children;
    
    /*! set of transform matrices for t=0 and t=1, respectively. if we
      don't use motion blur, the second one may be unused */
    std::vector<affine3f>   transforms[2];

    /*! vector of instnace IDs to use for these instances - if not
      specified we/optix will fill in automatically using
      instanceID=childID */
    std::vector<uint32_t>   instanceIDs;

    /*! vector of visibility masks to use for these instances - if not
      specified we/optix will fill in automatically using
      visibility=255 */
    std::vector<uint8_t> visibilityMasks;

    constexpr static unsigned int defaultBuildFlags = 
        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;

    protected:
    const unsigned int buildFlags;

  };

  // ------------------------------------------------------------------
  // implementation section
  // ------------------------------------------------------------------
  
  /*! get reference to given device-specific data for this object */
  inline InstanceGroup::DeviceData &InstanceGroup::getDD(const DeviceContext::SP &device) const
  {
    assert(device && device->ID < (int)deviceData.size());
    return deviceData[device->ID]->as<DeviceData>();
  }
  
} // ::owl
