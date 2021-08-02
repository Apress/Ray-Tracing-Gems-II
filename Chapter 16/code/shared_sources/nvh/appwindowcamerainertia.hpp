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
 */ //--------------------------------------------------------------------
#include <nvpwindow.hpp>
#ifdef WIN32
#include <windows.h>
#endif

#include "nvh/camerainertia.hpp"
#include "nvh/timesampler.hpp"
#include <imgui/imgui_helper.h>

#ifdef SUPPORT_NVTOOLSEXT
#include "nvh/nsightevents.h"
#else
// Note: they are defined inside "nsightevents.h"
// but let's define them again here as empty defines for the case when NSIGHT is not needed at all
#define NX_RANGE int
#define NX_MARK(name)
#define NX_RANGESTART(name) 0
#define NX_RANGEEND(id)
#define NX_RANGEPUSH(name)
#define NX_RANGEPUSHCOL(name, c)
#define NX_RANGEPOP()
#define NXPROFILEFUNC(name)
#define NXPROFILEFUNCCOL(name, c)
#define NXPROFILEFUNCCOL2(name, c, a)
#endif

#include <map>
using std::map;

#define KEYTAU 0.10f
//-----------------------------------------------------------------------------
// GLOBALS
//-----------------------------------------------------------------------------
#ifndef WIN32
struct POINT
{
  int x;
  int y;
};
#endif
struct ToggleInfo {
  bool * p;
  bool addToUI;
  std::string desc;
};
#ifdef WINDOWINERTIACAMERA_EXTERN
extern std::map<char, ToggleInfo> g_toggleMap;
#else
std::map<char, ToggleInfo> g_toggleMap;
#endif
inline void addToggleKey(char c, bool* target, const char* desc, bool addToUI = true)
{
  LOGI(desc);
  g_toggleMap[c].desc = desc;
  g_toggleMap[c].p = target;
  g_toggleMap[c].addToUI = addToUI;
}
//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------
inline void DrawToggles()
{
  for (auto &it : g_toggleMap) {
    if (!it.second.addToUI)
      continue;
    bool *pB = it.second.p;
    bool  prevValue = *pB;
    ImGui::Checkbox(it.second.desc.c_str(), pB);
  }
}
//-----------------------------------------------------------------------------
// Derive the Window for this sample
//-----------------------------------------------------------------------------
class AppWindowCameraInertia : public NVPWindow
{
public:
  AppWindowCameraInertia(const vec3f eye    = vec3f(0.0f, 1.0f, -3.0f),
                         const vec3f focus  = vec3f(0, 0, 0),
                         const vec3f object = vec3f(0, 0, 0),
                         float       fov_   = 50.0,
                         float       near_  = 0.01f,
                         float       far_   = 10.0)
      : m_camera(eye, focus, object)
  {
    m_renderCnt          = 1;
    m_bCameraMode        = true;
    m_bContinue          = true;
    m_moveStep           = 0.2f;
    m_ptLastMousePosit.x = m_ptLastMousePosit.y = 0;
    m_ptCurrentMousePosit.x = m_ptCurrentMousePosit.y = 0;
    m_ptOriginalMousePosit.x = m_ptOriginalMousePosit.y = 0;
    m_bMousing                                          = false;
    m_bRMousing                                         = false;
    m_bMMousing                                         = false;
    m_bNewTiming                                        = false;
    m_bAdjustTimeScale                                  = true;
    m_fov                                               = fov_;
    m_near                                              = near_;
    m_far                                               = far_;
  }

  bool  m_bCameraMode;
  bool  m_bContinue;
  float m_moveStep;
  POINT m_ptLastMousePosit;
  POINT m_ptCurrentMousePosit;
  POINT m_ptOriginalMousePosit;
  bool  m_bMousing;
  bool  m_bRMousing;
  bool  m_bMMousing;
  bool  m_bNewTiming;
  bool  m_bAdjustTimeScale;

  int           m_renderCnt;
  TimeSampler   m_realtime;
  bool          m_timingGlitch;
  InertiaCamera m_camera;
  mat4f         m_projection;
  float         m_fov, m_near, m_far;

public:
  inline mat4f& projMat() { return m_projection; }
  inline mat4f& viewMat() { return m_camera.m4_view; }
  inline bool&  nonStopRendering() { return m_realtime.bNonStopRendering; }

  bool open(int posX, int posY, int width, int height, const char* title, bool requireGLContext) override;

  virtual void onWindowClose() override;
  virtual void onWindowResize(int w, int h) override;
  virtual void onWindowRefresh() override;
  virtual void onMouseMotion(int x, int y) override;
  virtual void onMouseWheel(int delta) override;
  virtual void onMouseButton(NVPWindow::MouseButton button, ButtonAction action, int mods, int x, int y) override;
  virtual void onKeyboard(AppWindowCameraInertia::KeyCode key, ButtonAction action, int mods, int x, int y) override;
  virtual void onKeyboardChar(unsigned char key, int mods, int x, int y) override;
  
  virtual int idle();

  const char* getHelpText(int* lines = NULL)
  {
    if(lines)
      *lines = 7;
    return "Left mouse button: rotate around the target\n"
           "Right mouse button: translate target forward backward (+ Y axis rotate)\n"
           "Middle mouse button: Pan target along view plane\n"
           "Mouse wheel or PgUp/PgDn: zoom in/out\n"
           "Arrow keys: rotate around the target\n"
           "Ctrl+Arrow keys: Pan target\n"
           "Ctrl+PgUp/PgDn: translate target forward/backward\n";
  }
};
#ifndef WINDOWINERTIACAMERA_EXTERN

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------
bool AppWindowCameraInertia::open(int posX, int posY, int width, int height, const char* title, bool requireGLContext)
{
  m_realtime.bNonStopRendering = true;

  float r      = (float)width / (float)height;
  m_projection = perspective(m_fov, r, m_near, m_far);

  ImGuiH::Init(width, height, this);
  return NVPWindow::open(posX, posY, width, height, title, requireGLContext);
}

void AppWindowCameraInertia::onWindowClose() {}

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------
#define CAMERATAU 0.03f
void AppWindowCameraInertia::onMouseMotion(int x, int y)
{
  m_ptCurrentMousePosit.x = x;
  m_ptCurrentMousePosit.y = y;
  if(ImGuiH::mouse_pos(x, y))
    return;
  //---------------------------- LEFT
  if(m_bMousing)
  {
    float hval   = 2.0f * (float)(m_ptCurrentMousePosit.x - m_ptLastMousePosit.x) / (float)getWidth();
    float vval   = 2.0f * (float)(m_ptCurrentMousePosit.y - m_ptLastMousePosit.y) / (float)getHeight();
    m_camera.tau = CAMERATAU;
    m_camera.rotateH(hval);
    m_camera.rotateV(vval);
    m_renderCnt++;
  }
  //---------------------------- MIDDLE
  if(m_bMMousing)
  {
    float hval   = 2.0f * (float)(m_ptCurrentMousePosit.x - m_ptLastMousePosit.x) / (float)getWidth();
    float vval   = 2.0f * (float)(m_ptCurrentMousePosit.y - m_ptLastMousePosit.y) / (float)getHeight();
    m_camera.tau = CAMERATAU;
    m_camera.rotateH(hval, true);
    m_camera.rotateV(vval, true);
    m_renderCnt++;
  }
  //---------------------------- RIGHT
  if(m_bRMousing)
  {
    float hval   = 2.0f * (float)(m_ptCurrentMousePosit.x - m_ptLastMousePosit.x) / (float)getWidth();
    float vval   = -2.0f * (float)(m_ptCurrentMousePosit.y - m_ptLastMousePosit.y) / (float)getHeight();
    m_camera.tau = CAMERATAU;
    m_camera.rotateH(hval, !!(getKeyModifiers() & KMOD_CONTROL));
    m_camera.move(vval, !!(getKeyModifiers() & KMOD_CONTROL));
    m_renderCnt++;
  }

  m_ptLastMousePosit.x = m_ptCurrentMousePosit.x;
  m_ptLastMousePosit.y = m_ptCurrentMousePosit.y;
}

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------
void AppWindowCameraInertia::onMouseWheel(int delta)
{
  if(ImGuiH::mouse_wheel(delta))
    return;
  m_camera.tau = KEYTAU;
  m_camera.move(delta > 0 ? m_moveStep : -m_moveStep, !!(getKeyModifiers() & KMOD_CONTROL));
  m_renderCnt++;
}
//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------
void AppWindowCameraInertia::onMouseButton(NVPWindow::MouseButton button, NVPWindow::ButtonAction state, int mods, int x, int y)
{
  if(ImGuiH::mouse_button(button, state))
    return;
  switch(button)
  {
    case NVPWindow::MOUSE_BUTTON_LEFT:
      if(state == NVPWindow::BUTTON_PRESS)
      {
        m_renderCnt++;
        // TODO: equivalent of glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED/NORMAL);
        m_bMousing = true;
        m_renderCnt++;
        if(getKeyModifiers() & KMOD_CONTROL)
        {
        }
        else if(getKeyModifiers() & KMOD_SHIFT)
        {
        }
      }
      else
      {
        m_bMousing = false;
        m_renderCnt++;
      }
      break;
    case NVPWindow::MOUSE_BUTTON_RIGHT:
      if(state == NVPWindow::BUTTON_PRESS)
      {
        m_ptLastMousePosit.x = m_ptCurrentMousePosit.x = x;
        m_ptLastMousePosit.y = m_ptCurrentMousePosit.y = y;
        m_bRMousing                                    = true;
        m_renderCnt++;
        if(getKeyModifiers() & KMOD_CONTROL)
        {
        }
      }
      else
      {
        m_bRMousing = false;
        m_renderCnt++;
      }
      break;
    case NVPWindow::MOUSE_BUTTON_MIDDLE:
      if(state == NVPWindow::BUTTON_PRESS)
      {
        m_ptLastMousePosit.x = m_ptCurrentMousePosit.x = x;
        m_ptLastMousePosit.y = m_ptCurrentMousePosit.y = y;
        m_bMMousing                                    = true;
        m_renderCnt++;
      }
      else
      {
        m_bMMousing = false;
        m_renderCnt++;
      }
      break;
  }
}
//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------
void AppWindowCameraInertia::onKeyboard(NVPWindow::KeyCode key, NVPWindow::ButtonAction action, int mods, int x, int y)
{
  m_renderCnt++;
  if(ImGuiH::key_button(key, action, mods))
    return;
  if(action == NVPWindow::BUTTON_RELEASE)
    return;
  switch(key)
  {
    case NVPWindow::KEY_F1:
      break;
    case NVPWindow::KEY_F2:
      break;
    case NVPWindow::KEY_F3:
    case NVPWindow::KEY_F4:
    case NVPWindow::KEY_F5:
    case NVPWindow::KEY_F6:
    case NVPWindow::KEY_F7:
    case NVPWindow::KEY_F8:
    case NVPWindow::KEY_F9:
    case NVPWindow::KEY_F10:
    case NVPWindow::KEY_F11:
      break;
    case NVPWindow::KEY_F12:
      break;
    case NVPWindow::KEY_LEFT:
      m_camera.tau = KEYTAU;
      m_camera.rotateH(m_moveStep, !!(getKeyModifiers() & KMOD_CONTROL));
      break;
    case NVPWindow::KEY_UP:
      m_camera.tau = KEYTAU;
      m_camera.rotateV(m_moveStep, !!(getKeyModifiers() & KMOD_CONTROL));
      break;
    case NVPWindow::KEY_RIGHT:
      m_camera.tau = KEYTAU;
      m_camera.rotateH(-m_moveStep, !!(getKeyModifiers() & KMOD_CONTROL));
      break;
    case NVPWindow::KEY_DOWN:
      m_camera.tau = KEYTAU;
      m_camera.rotateV(-m_moveStep, !!(getKeyModifiers() & KMOD_CONTROL));
      break;
    case NVPWindow::KEY_PAGE_UP:
      m_camera.tau = KEYTAU;
      m_camera.move(m_moveStep, !!(getKeyModifiers() & KMOD_CONTROL));
      break;
    case NVPWindow::KEY_PAGE_DOWN:
      m_camera.tau = KEYTAU;
      m_camera.move(-m_moveStep, !!(getKeyModifiers() & KMOD_CONTROL));
      break;
    case NVPWindow::KEY_ESCAPE:
      close();
      break;
  }
}

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------
void AppWindowCameraInertia::onKeyboardChar(unsigned char key, int mods, int x, int y)
{
  m_renderCnt++;
  if(ImGuiH::key_char(key))
    return;
  // check registered toggles
  auto it = g_toggleMap.find(key);
  if(it != g_toggleMap.end())
  {
    it->second.p[0] = it->second.p[0] ? false : true;
  }
}

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------
int AppWindowCameraInertia::idle()
{
  //
  // Camera motion
  //
  m_bContinue = m_camera.update((float)m_realtime.getFrameDT());
  //
  // time sampling
  //
  m_realtime.update(m_bContinue, &m_timingGlitch);
  //
  // if requested: trigger again the next frame for rendering
  //
  if(m_bContinue || m_realtime.bNonStopRendering)
    m_renderCnt++;
  return m_renderCnt;
}

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------
void AppWindowCameraInertia::onWindowRefresh()
{
}

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------
void AppWindowCameraInertia::onWindowResize(int w, int h)
{
  NVPWindow::onWindowResize(w, h);
  auto& imgui_io       = ImGui::GetIO();
  imgui_io.DisplaySize = ImVec2(float(w), float(h));

  float r      = (float)w / (float)h;
  m_projection = perspective(m_fov, r, m_near, m_far);
  m_renderCnt++;
}
#endif  //WINDOWINERTIACAMERA_EXTERN
