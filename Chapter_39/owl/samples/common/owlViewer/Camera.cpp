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

#include "Camera.h"
#include "OWLViewer.h"

namespace owl {
  namespace viewer {

    vec3f Camera::getFrom() const
    {
      return position;
    }
    
    vec3f Camera::getAt() const
    {
      return position - frame.vz;
    }

    vec3f Camera::getUp() const
    {
      return frame.vy;
    }
      
    // void Camera::digestInto(SimpleCamera &easy)
    // {
    //   easy.lens.center = position;
    //   easy.lens.radius = 0.f;
    //   easy.lens.du     = frame.vx;
    //   easy.lens.dv     = frame.vy;

    //   const float minFocalDistance
    //     = max(computeStableEpsilon(position),
    //                computeStableEpsilon(frame.vx));

    //   /*
    //     tan(fov/2) = (height/2) / dist
    //     -> height = 2*tan(fov/2)*dist
    //   */
    //   float screen_height
    //     = 2.f*tanf(fovyInDegrees/2 * (float)M_PI/180.f)
    //     * max(minFocalDistance,focalDistance);
    //   easy.screen.vertical   = screen_height * frame.vy;
    //   easy.screen.horizontal = screen_height * aspect * frame.vx;
    //   easy.screen.lower_left
    //     = //easy.lens.center
    //     /* NEGATIVE z axis! */
    //     - max(minFocalDistance,focalDistance) * frame.vz
    //     - 0.5f * easy.screen.vertical
    //     - 0.5f * easy.screen.horizontal;

    //   easy.lastModified = getCurrentTime();
    // }

    void Camera::setFovy(const float fovy)
    {
      this->fovyInDegrees = fovy;
    }

    void Camera::setAspect(const float aspect)
    {
      this->aspect = aspect;
    }

    void Camera::setFocalDistance(float focalDistance)
    {
      this->focalDistance = focalDistance;
    }

    /*! tilt the frame around the z axis such that the y axis is "facing upwards" */
    void Camera::forceUpFrame()
    {
      // frame.vz remains unchanged
      if (fabsf(dot(frame.vz,upVector)) < 1e-6f)
        // looking along upvector; not much we can do here ...
        return;
      frame.vx = normalize(cross(upVector,frame.vz));
      frame.vy = normalize(cross(frame.vz,frame.vx));
    }

    void Camera::setOrientation(/* camera origin    : */const vec3f &origin,
                                    /* point of interest: */const vec3f &interest,
                                    /* up-vector        : */const vec3f &up,
                                    /* fovy, in degrees : */float fovyInDegrees,
                                    /* set focal dist?  : */bool  setFocalDistance)
    {
      this->fovyInDegrees = fovyInDegrees;
      position = origin;
      upVector = up;
      frame.vz
        = (interest==origin)
        ? vec3f(0,0,1)
        : /* negative because we use NEGATIZE z axis */ - normalize(interest - origin);
      frame.vx = cross(up,frame.vz);
      if (dot(frame.vx,frame.vx) < 1e-8f)
        frame.vx = vec3f(0,1,0);
      else
        frame.vx = normalize(frame.vx);
      // frame.vx
      //   = (fabs(dot(up,frame.vz)) < 1e-6f)
      //   ? vec3f(0,1,0)
      //   : normalize(cross(up,frame.vz));
      frame.vy = normalize(cross(frame.vz,frame.vx));
      poiDistance = length(interest-origin);
      if (setFocalDistance) focalDistance = poiDistance;
      forceUpFrame();
    }


    /*! this gets called when the user presses a key on the keyboard ... */
    void CameraManipulator::key(char key, const vec2i &where)
    {
      Camera &fc = viewer->camera;

      switch(key) {
      case 'f':
      case 'F':
        if (viewer->flyModeManipulator)
          viewer->cameraManipulator = viewer->flyModeManipulator;
        break;
      case 'i':
      case 'I':
        if (viewer->inspectModeManipulator)
          viewer->cameraManipulator = viewer->inspectModeManipulator;
        break;
      case '+':
      case '=':
        fc.motionSpeed *= 2.f;
        std::cout << "# viewer: new motion speed is " << fc.motionSpeed << std::endl;
        break;
      case '-':
      case '_':
        fc.motionSpeed /= 2.f;
        std::cout << "# viewer: new motion speed is " << fc.motionSpeed << std::endl;
        break;
      case 'C': {
        // for anybody deriving from this class: if you do not like
        // this output, simply intercept the 'C' key in your derived
        // 'key()' method before calling this...
        std::cout << "(C)urrent camera:" << std::endl;
        std::cout << "- from :" << fc.position << std::endl;
        std::cout << "- poi  :" << fc.getPOI() << std::endl;
        std::cout << "- upVec:" << fc.upVector << std::endl;
        std::cout << "- frame:" << fc.frame << std::endl;

        const vec3f vp = fc.position;
        const vec3f vi = fc.getPOI();
        const vec3f vu = fc.upVector;
        const float fovy = fc.getFovyInDegrees();
        std::cout << "(suggested cmdline format, for apps that support this:) "
                  << " --camera"
                  << " " << vp.x << " " << vp.y << " " << vp.z
                  << " " << vi.x << " " << vi.y << " " << vi.z
                  << " " << vu.x << " " << vu.y << " " << vu.z
                  << " -fovy " << fovy
                  << std::endl;
      } break;
      case 'x':
      case 'X':
        fc.setUpVector(fc.upVector==vec3f(1,0,0)?vec3f(-1,0,0):vec3f(1,0,0));
        viewer->updateCamera();
        break;
      case 'y':
      case 'Y':
        fc.setUpVector(fc.upVector==vec3f(0,1,0)?vec3f(0,-1,0):vec3f(0,1,0));
        viewer->updateCamera();
        break;
      case 'z':
      case 'Z':
        fc.setUpVector(fc.upVector==vec3f(0,0,1)?vec3f(0,0,-1):vec3f(0,0,1));
        viewer->updateCamera();
        break;
      default:
        break;
      }
    }

  } // ::owl::viewer
} // ::owl

