// ======================================================================== //
// Copyright 2018-2020 Ingo Wald                                            //
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

#include "Camera.h"

namespace owl {
  namespace viewer {

        // ------------------------------------------------------------------
    /*! camera manipulator with the following traits
     
     - Rotate mode ROI:

        - there is a "point of interest" (POI) that the camera rotates
        around.  (we track this via poiDistance, the point is then
        thta distance along the fame's z axis)
        
        - we can restrict the minimum and maximum distance camera can be
        away from this point
        
        - we can specify a max bounding box that this poi can never
        exceed (validPoiBounds).
        
        - we can restrict whether that point can be moved (by using a
        single-point valid poi bounds box
      
        - left drag rotates around the object

      - Rotate mode Arcball:

        * on left drag, translate mouse movement to arcball rotation

      - right drag moved forward, backward (within min/max distance
      bounds)

      - middle drag strafes left/right/up/down (within valid poi
      bounds)
      
    */
    struct CameraInspectMode : public CameraManipulator {

      enum RotateMode { POI, Arcball };

      CameraInspectMode(OWLViewer *widget,
                       const box3f &validPoiBounds,
                       const float minDistance,
                       const float maxDistance,
                       const RotateMode rotateMode = POI);

      /*! set to rotation around ROI or arcball rotation */
      void setRotateMode(RotateMode rm);

      /*! this gets called when the user presses a key on the keyboard ... */
      virtual void key(char key, const vec2i &where) override;
      /*! this gets called when the user presses a key on the keyboard ... */
      virtual void special(int key, const vec2i &where) override;
      
    private:
      /*! helper function: rotate camera frame by given degrees, then
          make sure the frame, poidistance etc are all properly set,
          the widget gets notified, etc */
      void rotate(const float deg_x, const float deg_y);

      /*! helper function: move forward/backwards by given multiple of
          motion speed, then make sure the frame, poidistance etc are
          all properly set, the widget gets notified, etc */
      void move(const float step);

      /*! move the POI itself */
      void movePOI(const vec2f delta);

      /*! project x/y window coordinate to (unit) arcball coordinates */
      vec3f arcballProject(const vec2i &where);

      void kbd_up();
      void kbd_down();
      void kbd_left();
      void kbd_right();
      void kbd_forward();
      void kbd_back();

      /*! mouse got dragged with left button pressedn, by 'delta'
          pixels, at last position where */
      virtual void mouseDragLeft  (const vec2i &where, const vec2i &delta) override;

      /*! mouse got dragged with left button pressedn, by 'delta'
          pixels, at last position where */
      virtual void mouseDragRight (const vec2i &where, const vec2i &delta) override;

      /*! mouse got dragged with left button pressedn, by 'delta'
          pixels, at last position where */
      virtual void mouseDragCenter(const vec2i &where, const vec2i &delta) override;
      
      /*! mouse button got either pressed or released at given location */
      virtual void mouseButtonLeft(const vec2i &where, bool pressed) override;


      /*! if non-empty, bounding box of region the poi is not allowed
        to leave */
      const box3f validPoiBounds;
      const float minDistance;
      const float maxDistance;
      
      /*! either around ROI or on arcball */
      RotateMode rotateMode;

      /*! either use arcball rotation or rotation around POI */

      struct {
        /*! the current rotation
            zero-length signifies that the arcball wasn't
            initialized yet */
        Quaternion3f rotation = Quaternion3f(0.f);

        /*! the rotation when the mouse button was pressed */
        Quaternion3f down_rotation = Quaternion3f(1.f);

        /*! the position projected to the unit arcball when
            the dragging mouse button was pressed */
        vec3f down_pos = vec3f(0.f);
      } arcball;
    };

  } // ::owl::viewer
} // ::owl
