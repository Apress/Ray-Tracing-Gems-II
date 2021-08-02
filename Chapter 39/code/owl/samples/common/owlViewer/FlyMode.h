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

      - left button rotates the camera around the viewer position

      - middle button strafes in camera plane
      
      - right buttton moves forward/backwards
      
    */
    struct CameraFlyMode : public CameraManipulator {

      CameraFlyMode(OWLViewer *widget)
        : CameraManipulator(widget)
      {}

      /*! this gets called when the user presses a key on the keyboard ... */
      virtual void key(char key, const vec2i &where) override;
      /*! this gets called when the user presses a key on the keyboard ... */
      virtual void special(int key, const vec2i &where) override;
      
    private:
      /*! helper function: rotate camera frame by given degrees, then
          make sure the frame, poidistance etc are all properly set,
          the widget gets notified, etc */
      void rotate(const float deg_x, const float deg_y);
      
      /*! move forward/backward */
      void move(const float step);

      /*! strafe in camera plane */
      void strafe(const vec2f delta);

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
      

    };

  } // ::owl::viewer
} // ::owl
