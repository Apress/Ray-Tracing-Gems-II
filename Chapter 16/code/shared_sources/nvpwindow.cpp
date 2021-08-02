/*-----------------------------------------------------------------------
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include "nvpwindow.hpp"

#ifdef USESOCKETS
#include "socketSampleMessages.h"
#endif

#include <algorithm>
#include <stdio.h>
#include <string>

void NVPWindow::cb_windowrefreshfun(GLFWwindow* glfwwin)
{
  NVPWindow* win = (NVPWindow*)glfwGetWindowUserPointer(glfwwin);
  if (win->isClosing()) return;
  win->onWindowRefresh();
}

void NVPWindow::cb_windowsizefun(GLFWwindow* glfwwin, int w, int h)
{
  NVPWindow* win = (NVPWindow*)glfwGetWindowUserPointer(glfwwin);
  if (win->isClosing()) return;
  win->m_windowSize[0] = w;
  win->m_windowSize[1] = h;
  win->onWindowResize(w, h);
}
void NVPWindow::cb_windowclosefun(GLFWwindow* glfwwin)
{
  NVPWindow* win = (NVPWindow*)glfwGetWindowUserPointer(glfwwin);
  win->m_isClosing = true;
  win->onWindowClose();
}

void NVPWindow::cb_mousebuttonfun(GLFWwindow* glfwwin, int button, int action, int mods)
{
  double x,y;
  glfwGetCursorPos(glfwwin, &x, &y);

  NVPWindow* win = (NVPWindow*)glfwGetWindowUserPointer(glfwwin);
  if (win->isClosing()) return;
  win->m_keyModifiers = mods;
  win->m_mouseX = int(x);
  win->m_mouseY = int(y);
  win->onMouseButton((NVPWindow::MouseButton)button, (NVPWindow::ButtonAction)action, mods, win->m_mouseX, win->m_mouseY);
}
void NVPWindow::cb_cursorposfun(GLFWwindow* glfwwin,double x,double y)
{
  NVPWindow* win = (NVPWindow*)glfwGetWindowUserPointer(glfwwin);
  if (win->isClosing()) return;
  win->m_mouseX = int(x);
  win->m_mouseY = int(y);
  win->onMouseMotion(win->m_mouseX, win->m_mouseY);
}
void NVPWindow::cb_scrollfun(GLFWwindow* glfwwin, double x,double y)
{
  NVPWindow* win = (NVPWindow*)glfwGetWindowUserPointer(glfwwin);
  if (win->isClosing()) return;
  win->m_mouseWheel += int(y);
  win->onMouseWheel(int(y));
}
void NVPWindow::cb_keyfun(GLFWwindow* glfwwin, int key, int scancode, int action, int mods)
{
  NVPWindow* win = (NVPWindow*)glfwGetWindowUserPointer(glfwwin);
  if (win->isClosing()) return;
  win->m_keyModifiers = mods;
  win->onKeyboard((NVPWindow::KeyCode) key, (NVPWindow::ButtonAction)action, mods, win->m_mouseX, win->m_mouseY);
}
void NVPWindow::cb_charfun(GLFWwindow* glfwwin, unsigned int codepoint)
{
  NVPWindow* win = (NVPWindow*)glfwGetWindowUserPointer(glfwwin);
  if (win->isClosing()) return;
  win->onKeyboardChar(codepoint, win->m_keyModifiers, win->m_mouseX, win->m_mouseY);
}

void NVPWindow::cb_dropfun(GLFWwindow* glfwwin, int count, const char** paths)
{
  NVPWindow* win = (NVPWindow*)glfwGetWindowUserPointer(glfwwin);
  if (win->isClosing()) return;
  win->onDragDrop(count, paths);
}

bool NVPWindow::open(int posX, int posY, int width, int height, const char* title, bool requireGLContext)
{
  NV_ASSERT(NVPSystem::isInited() && "NVPSystem::Init not called");

  m_windowSize[0] = width;
  m_windowSize[1] = height;

  m_windowName = title ? title : "Sample";

#ifdef _WIN32
  (void)requireGLContext;
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
#else
  if(!requireGLContext)
  {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  }
  else
  {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  }
#endif

  m_internal = glfwCreateWindow(width, height, title, nullptr, nullptr);
  if(!m_internal)
  {
    return false;
  }
  
  if (posX != 0 || posY != 0){
    glfwSetWindowPos(m_internal, posX, posY);
  }
  glfwSetWindowUserPointer(m_internal, this);
  glfwSetWindowRefreshCallback(m_internal, cb_windowrefreshfun);
  glfwSetWindowCloseCallback(m_internal, cb_windowclosefun);
  glfwSetCursorPosCallback(m_internal, cb_cursorposfun);
  glfwSetMouseButtonCallback(m_internal, cb_mousebuttonfun);
  glfwSetKeyCallback(m_internal, cb_keyfun);
  glfwSetScrollCallback(m_internal, cb_scrollfun);
  glfwSetCharCallback(m_internal, cb_charfun);
  glfwSetWindowSizeCallback(m_internal, cb_windowsizefun);
  glfwSetDropCallback(m_internal, cb_dropfun);

  return true;
}

void NVPWindow::deinit()
{
  glfwDestroyWindow(m_internal);
  m_internal        = nullptr;
  m_windowSize[0] = 0;
  m_windowSize[1] = 0;
  m_windowName    = std::string();
}

void NVPWindow::close()
{
  glfwSetWindowShouldClose(m_internal, GLFW_TRUE);
}

void NVPWindow::setTitle(const char* title)
{
  glfwSetWindowTitle(m_internal, title);
}

void NVPWindow::maximize()
{
  glfwMaximizeWindow(m_internal);
}

void NVPWindow::restore()
{
  glfwRestoreWindow(m_internal);
}

void NVPWindow::minimize()
{
  glfwIconifyWindow(m_internal);
}

void NVPWindow::setWindowPos(int x, int y)
{
  glfwSetWindowPos(m_internal, x, y);
}

void NVPWindow::setWindowSize(int w, int h)
{
  glfwSetWindowSize(m_internal, w, h);
}

std::string NVPWindow::openFileDialog(const char* title, const char* exts)
{
  return NVPSystem::windowOpenFileDialog(m_internal, title, exts);
}
void NVPWindow::screenshot(const char* filename)
{
  NVPSystem::windowScreenshot(m_internal, filename);
}
void NVPWindow::clear(uint32_t r, uint32_t g, uint32_t b)
{
  NVPSystem::windowClear(m_internal, r, g, b);
}

void NVPWindow::setFullScreen(bool bYes)
{
  if (bYes == m_isFullScreen) return;

  GLFWmonitor* monitor = glfwGetWindowMonitor(m_internal);
  const GLFWvidmode* mode = glfwGetVideoMode(monitor);

  if (bYes){
    glfwGetWindowPos(m_internal, &m_preFullScreenPos[0], &m_preFullScreenPos[1]);
    glfwGetWindowSize(m_internal, &m_preFullScreenSize[0], &m_preFullScreenSize[1]);
    glfwSetWindowMonitor(m_internal, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
    glfwSetWindowAttrib(m_internal, GLFW_RESIZABLE, GLFW_FALSE);
    glfwSetWindowAttrib(m_internal, GLFW_DECORATED, GLFW_FALSE);
  }
  else {
    glfwSetWindowMonitor(m_internal, nullptr, m_preFullScreenPos[0], m_preFullScreenPos[1], m_preFullScreenSize[0], m_preFullScreenSize[1], 0);
    glfwSetWindowAttrib(m_internal, GLFW_RESIZABLE, GLFW_TRUE);
    glfwSetWindowAttrib(m_internal, GLFW_DECORATED, GLFW_TRUE);
  }

  m_isFullScreen = bYes;
}

