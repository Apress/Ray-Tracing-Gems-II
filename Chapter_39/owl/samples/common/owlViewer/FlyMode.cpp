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

#include "FlyMode.h"
#include "OWLViewer.h"

namespace owl {
  namespace viewer {

    const float kbd_rotate_degrees = 10.f;
    const float degrees_per_drag_fraction = 150;
    const float pixels_per_move = 10.f;
    
    // ##################################################################
    // actual motion functions that do the actual work
    // ##################################################################

    void CameraFlyMode::move(const float step)
    {
      Camera &fc = viewer->camera;
      // negative z axis: 'subtract' step
      fc.position = fc.position - step*fc.motionSpeed * fc.frame.vz;
      viewer->updateCamera();
    }

    void CameraFlyMode::strafe(const vec2f step)
    {
      Camera &fc = viewer->camera;
      fc.position = fc.position
        - step.x*fc.motionSpeed * fc.frame.vx
        + step.y*fc.motionSpeed * fc.frame.vy;
      viewer->updateCamera();
    }

    void CameraFlyMode::rotate(const float deg_u,
                                  const float deg_v)
    {
      float rad_u = -(float)M_PI/180.f*deg_u;
      float rad_v = -(float)M_PI/180.f*deg_v;

      assert(viewer);
      Camera &fc = viewer->camera;
      
      fc.frame
        = linear3f::rotate(fc.frame.vy,rad_u)
        * linear3f::rotate(fc.frame.vx,rad_v)
        * fc.frame;

      if (fc.forceUp) fc.forceUpFrame();

      viewer->updateCamera();
    }

    // ##################################################################
    // MOUSE interaction
    // ##################################################################

    /*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
    void CameraFlyMode::mouseDragLeft(const vec2i &where, const vec2i &delta)
    {
      const vec2f fraction = vec2f(delta) / vec2f(viewer->getWindowSize());
      rotate(fraction.x * degrees_per_drag_fraction,
             fraction.y * degrees_per_drag_fraction);
    }

    /*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
    void CameraFlyMode::mouseDragCenter(const vec2i &where, const vec2i &delta)
    {
      const vec2f fraction = vec2f(delta) / vec2f(viewer->getWindowSize());
      strafe(fraction*pixels_per_move);
    }

    /*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
    void CameraFlyMode::mouseDragRight(const vec2i &where, const vec2i &delta)
    {
      const vec2f fraction = vec2f(delta) / vec2f(viewer->getWindowSize());
      move(fraction.y*pixels_per_move);
    }

    // ##################################################################
    // KEYBOARD interaction
    // ##################################################################

    void CameraFlyMode::kbd_up()
    {
      rotate(0,+kbd_rotate_degrees);
    }
    
    void CameraFlyMode::kbd_down()
    {
      rotate(0,-kbd_rotate_degrees);
    }

    /*! keyboard left/right: note this works _exactly_ the other way
        around than the camera does: moving camera to the right
        'drags' the model to the right (ie, the camer to the left),
        but _typing_ right 'moves' the viewer/camera, so rotates
        camera to the right). This _reads_ counter-intuitive, but
        feels more natural, so is intentional */
    void CameraFlyMode::kbd_right()
    {
      rotate(-kbd_rotate_degrees,0);
    }
    
    /*! keyboard left/right: note this works _exactly_ the other way
        around than the camera does: moving camera to the right
        'drags' the model to the right (ie, the camer to the left),
        but _typing_ right 'moves' the viewer/camera, so rotates
        camera to the right). This _reads_ counter-intuitive, but
        feels more natural, so is intentional */
    void CameraFlyMode::kbd_left()
    {
      rotate(+kbd_rotate_degrees,0);
    }
    
    void CameraFlyMode::kbd_forward()
    {
      move(+1.f);
    }
    
    void CameraFlyMode::kbd_back()
    {
      move(-1.f);
      // Camera &fc = viewer->camera;
      // float step = 1.f;
      
      // const vec3f poi  = fc.position - fc.poiDistance * fc.frame.vz;
      // fc.poiDistance   = max(maxDistance,fc.poiDistance+step);
      // fc.focalDistance = fc.poiDistance;
      // fc.position = poi + fc.poiDistance * fc.frame.vz;
      // viewer->updateCamera();
    }
    
    
    /*! this gets called when the user presses a key on the keyboard ... */
    void CameraFlyMode::key(char key, const vec2i &where) 
    {
      Camera &fc = viewer->camera;
      
      switch(key) {
      case 'w':
        kbd_up();
        break;
      case 's':
        kbd_down();
        break;
      case 'd':
        kbd_right();
        break;
      case 'a':
        kbd_left();
        break;
      case 'e':
        kbd_forward();
        break;
      case 'c':
        kbd_back();
        break;
      default:
        CameraManipulator::key(key,where);
      }
    }

    /*! this gets called when the user presses a key on the keyboard ... */
    void CameraFlyMode::special(int key, const vec2i &where) 
    {
      switch (key) {
      case GLFW_KEY_UP:
        kbd_up();
        break;
      case GLFW_KEY_DOWN:
        kbd_down();
        break;
      case GLFW_KEY_RIGHT:
        kbd_right();
        break;
      case GLFW_KEY_LEFT:
        kbd_left();
        break;
      case GLFW_KEY_PAGE_UP:
        kbd_forward();
        break;
      case GLFW_KEY_PAGE_DOWN:
        kbd_back();
        break;
      }
    }

  } // ::owl::viewer
} // ::owl
