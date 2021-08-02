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

#pragma once

#include "owl/common/math/box.h"
#include "owl/common/math/LinearSpace.h"

#include <vector>
#include <memory>
#ifdef _GNUC_
    #include <unistd.h>
#endif

namespace owl {
  namespace viewer {

    inline float toRadian(float deg) { return deg * float(M_PI/180.f); }
    inline float toDegrees(float rad) { return rad / float(M_PI/180.f); }

    struct OWLViewer;

    /*! the entire state for someting that can 'control' a camera -
        ie, that can rotate, move, focus, force-up, etc, a
        camera... for which it needs way more information than the
        simple camera.

        Note this uses a RIGHT HANDED camera as follows:
        - logical "up" is y axis
        - right is x axis
        - depth is _NEGATIVE_ z axis
    */
    struct Camera {
      Camera()
      {}

      vec3f getPOI() const
      {
        return position - poiDistance * frame.vz;
      }
      float getFovyInDegrees() const { return fovyInDegrees; }
      float getCosFovy() const { return cosf(toRadian(fovyInDegrees)); }
      vec3f getFrom() const;
      vec3f getAt()   const;
      vec3f getUp()   const;
      
      void setFovy(const float fovy);

      void setFocalDistance(float focalDistance);

      /*! set given aspect ratio */
      void setAspect(const float aspect);

      /*! re-compute all orientation related fields from given
          'user-style' camera parameters */
      void setOrientation(/* camera origin    : */const vec3f &origin,
                          /* point of interest: */const vec3f &interest,
                          /* up-vector        : */const vec3f &up,
                          /* fovy, in degrees : */float fovyInDegrees,
                          /* set focal dist?  : */bool  setFocalDistance=true);

      /*! tilt the frame around the z axis such that the y axis is "facing upwards" */
      void forceUpFrame();

      void setUpVector(const vec3f &up)
      { upVector = up; forceUpFrame(); }

      linear3f      frame         { one };
      vec3f         position      { 0,-1,0 };
      /*! distance to the 'point of interst' (poi); e.g., the point we
          will rotate around */
      float         poiDistance   { 1.f };
      float         focalDistance { 1.f };
      vec3f         upVector      { 0,1,0 };
      /* if set to true, any change to the frame will always use to
         upVector to 'force' the frame back upwards; if set to false,
         the upVector will be ignored */
      bool          forceUp       { true };

      /*! multiplier how fast the camera should move in world space
          for each unit of "user specifeid motion" (ie, pixel
          count). Initial value typically should depend on the world
          size, but can also be adjusted. This is actually something
          that should be more part of the manipulator viewer(s), but
          since that same value is shared by multiple such viewers
          it's easiest to attach it to the camera here ...*/
      float         motionSpeed   { 1.f };
      float         aspect        { 1.f };
      float         fovyInDegrees { 60.f };

      double        lastModified = 0.;
    };

    // ------------------------------------------------------------------
    /*! abstract base class that allows to manipulate a renderable
      camera */
    struct CameraManipulator {
      CameraManipulator(OWLViewer *viewer) : viewer(viewer) {}

      /*! this gets called when the user presses a key on the keyboard ... */
      virtual void key(char key, const vec2i &where);

      /*! this gets called when the user presses a key on the keyboard ... */
      virtual void special(int key, const vec2i &where) { };

      /*! mouse got dragged with left button pressedn, by 'delta'
          pixels, at last position where */
      virtual void mouseDragLeft  (const vec2i &where, const vec2i &delta) {}

      /*! mouse got dragged with left button pressedn, by 'delta'
          pixels, at last position where */
      virtual void mouseDragCenter(const vec2i &where, const vec2i &delta) {}

      /*! mouse got dragged with left button pressedn, by 'delta'
          pixels, at last position where */
      virtual void mouseDragRight (const vec2i &where, const vec2i &delta) {}

      /*! mouse button got either pressed or released at given location */
      virtual void mouseButtonLeft  (const vec2i &where, bool pressed) {}

      /*! mouse button got either pressed or released at given location */
      virtual void mouseButtonCenter(const vec2i &where, bool pressed) {}

      /*! mouse button got either pressed or released at given location */
      virtual void mouseButtonRight (const vec2i &where, bool pressed) {}

    protected:
      OWLViewer *const viewer;
    };

  } // ::owl::viewer
} // ::owl

