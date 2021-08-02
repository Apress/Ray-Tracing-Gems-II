/* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef NV_CAMCONTROL_INCLUDED
#define NV_CAMCONTROL_INCLUDED

#include <nvmath/nvmath.h>
#include <algorithm>

namespace nvh
{
  //////////////////////////////////////////////////////////////////////////
  /**
    # class nvh::CameraControl

    CameraControl is a utility class to create a viewmatrix
    based on mouse inputs.
  
    It can operate in perspective or orthographic mode (`m_sceneOrtho==true`).
  
    perspective:
    - LMB: rotate
    - RMB or WHEEL: zoom via dolly movement
    - MMB: pan/move within camera plane
  
    ortho:
    - LMB: pan/move within camera plane
    - RMB or WHEEL: zoom via dolly movement, application needs to use `m_sceneOrthoZoom` for projection matrix adjustment
    - MMB: rotate
  
    The camera can be orbiting (`m_useOrbit==true`) around `m_sceneOrbit` or
    otherwise provide "first person/fly through"-like controls.
  
    Speed of movement/rotation etc. is influenced by `m_sceneDimension` as well as the 
    sensitivity values.
  */

  class CameraControl 
  {
  public:

    CameraControl()
      : m_lastButtonFlags(0)
      , m_lastWheel(0)
      , m_senseWheelZoom(0.05f/120.0f)
      , m_senseZoom(0.001f)
      , m_senseRotate((nv_pi*0.5f)/256.0f)
      , m_sensePan(1.0f)
      , m_sceneOrbit(0.0f)
      , m_sceneDimension(1.0f)
      , m_sceneOrtho(false)
      , m_sceneOrthoZoom(1.0f)
      , m_useOrbit(true)
      , m_sceneUp(0,1,0)
    {

    }

    inline void  processActions(const nvmath::vec2i &window, const nvmath::vec2f &mouse, int mouseButtonFlags, int wheel)
    {
      int changed  = m_lastButtonFlags ^ mouseButtonFlags;
      m_lastButtonFlags = mouseButtonFlags;

      int panFlag  = m_sceneOrtho ? 1<<0 : 1<<2;
      int zoomFlag = 1<<1;
      int rotFlag  = m_sceneOrtho ? 1<<2 : 1<<0;


      m_panning  = !!(mouseButtonFlags & panFlag);
      m_zooming  = !!(mouseButtonFlags & zoomFlag);
      m_rotating = !!(mouseButtonFlags & rotFlag);
      m_zoomingWheel = wheel != m_lastWheel;

      m_startZoomWheel = m_lastWheel;
      m_lastWheel = wheel;

      if (m_rotating){
        m_panning = false;
        m_zooming = false;
      }

      if (m_panning && (changed & panFlag)){
        // pan
        m_startPan = mouse;
        m_startMatrix = m_viewMatrix;
      }
      if (m_zooming && (changed & zoomFlag)){
        // zoom
        m_startMatrix = m_viewMatrix;
        m_startZoom = mouse;
        m_startZoomOrtho = m_sceneOrthoZoom;
      }
      if (m_rotating && (changed & rotFlag)){
        // rotate
        m_startRotate = mouse;
        m_startMatrix = m_viewMatrix;
      }

      if (m_zooming || m_zoomingWheel){
        float dist = 
          m_zooming ? -(nvmath::dot( mouse - m_startZoom ,nvmath::vec2f(-1,1)) * m_sceneDimension * m_senseZoom) 
          : (float(wheel - m_startZoomWheel) * m_sceneDimension * m_senseWheelZoom);

        if (m_zoomingWheel){
          m_startZoomOrtho = m_sceneOrthoZoom;
          m_startMatrix = m_viewMatrix;
        }

        if (m_sceneOrtho){
          float newzoom = m_startZoomOrtho - (dist);
          if (m_zoomingWheel){
            if (newzoom < 0){
              m_sceneOrthoZoom *= 0.5;
            }
            else if (m_sceneOrthoZoom < abs(dist)){
              m_sceneOrthoZoom *= 2.0;
            }
            else{
              m_sceneOrthoZoom = newzoom;
            }
          }
          else{
            m_sceneOrthoZoom = newzoom;
          }
          m_sceneOrthoZoom = std::max(0.0001f,m_sceneOrthoZoom);
        }
        else{
          nvmath::mat4f delta = nvmath::translation_mat4(nvmath::vec3f(0,0,dist * 2.0f));
          m_viewMatrix = delta * m_startMatrix;
        }

      }

      if (m_panning){
        float aspect = float(window.x)/float(window.y);

        nvmath::vec3f winsize(window.x, window.y, 1.0f);
        nvmath::vec3f ortho(m_sceneOrthoZoom * aspect, m_sceneOrthoZoom, 1.0f);
        nvmath::vec3f sub( mouse - m_startPan, 0.0f);
        sub /= winsize;
        sub *= ortho;
        sub.y *= -1.0;
        if (!m_sceneOrtho){
          sub *= m_sensePan * m_sceneDimension;
        }

        nvmath::mat4f delta;
        delta.as_translation(sub);
        m_viewMatrix = delta * m_startMatrix;
      }

      if (m_rotating){
        float aspect = float(window.x)/float(window.y);

        nvmath::vec2f angles = (mouse - m_startRotate) * m_senseRotate;
        

        if (m_useOrbit){
          nvmath::mat4f rot    = nvmath::rotation_yaw_pitch_roll(angles.x, angles.y, 0.0f);
          nvmath::vec3f center = nvmath::vec3f(m_startMatrix * nvmath::vec4f(m_sceneOrbit, 1.0f));
          nvmath::mat4f delta  = nvmath::translation_mat4(center) * rot * nvmath::translation_mat4(-center);

          m_viewMatrix = delta * m_startMatrix;
        }
        else{
          // FIXME use sceneUP
          nvmath::mat4f rot = nvmath::rotation_yaw_pitch_roll(angles.x, angles.y, 0.0f);

          m_viewMatrix = rot * m_startMatrix;
        }
      }
    }
    
    bool        m_useOrbit;
    bool        m_sceneOrtho;
    float       m_sceneOrthoZoom;
    float       m_sceneDimension;

    nvmath::vec3f   m_sceneUp;
    nvmath::vec3f   m_sceneOrbit;
    nvmath::mat4f   m_viewMatrix;

    float       m_senseWheelZoom;
    float       m_senseZoom;
    float       m_senseRotate;
    float       m_sensePan;

  private:
    bool        m_zooming;
    bool        m_zoomingWheel;
    bool        m_panning;
    bool        m_rotating;

    nvmath::vec2f   m_startPan;
    nvmath::vec2f   m_startZoom;
    nvmath::vec2f   m_startRotate;
    nvmath::mat4f   m_startMatrix;
    int         m_startZoomWheel;
    float       m_startZoomOrtho;

    int         m_lastButtonFlags;
    int         m_lastWheel;

  };
}

#endif
