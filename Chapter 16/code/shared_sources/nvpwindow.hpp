/*-----------------------------------------------------------------------
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __NVPWINDOW_H__
#define __NVPWINDOW_H__

#include "nvpsystem.hpp"

class NVPWindow
{
public:
  // IMPORTANT
  //
  // Using and deriving of NVPWindow base-class is optional.
  // However one must always make use of the NVPSystem
  // That takes care of glfwInit/terminate as well.

  ////////////////////////////////////////////////////////////////////////
  // base class

  // these are taken from GLFW3 and must be kept in a matching state

  enum ButtonAction
  {
    BUTTON_RELEASE = GLFW_RELEASE,
    BUTTON_PRESS   = GLFW_PRESS,
    BUTTON_REPEAT  = GLFW_REPEAT,
  };

  enum MouseButton
  {
    MOUSE_BUTTON_LEFT   = GLFW_MOUSE_BUTTON_LEFT,
    MOUSE_BUTTON_RIGHT  = GLFW_MOUSE_BUTTON_RIGHT,
    MOUSE_BUTTON_MIDDLE = GLFW_MOUSE_BUTTON_MIDDLE,
    NUM_MOUSE_BUTTONIDX,
  };

  enum MouseButtonFlag
  {
    MOUSE_BUTTONFLAG_NONE   = 0,
    MOUSE_BUTTONFLAG_LEFT   = (1 << MOUSE_BUTTON_LEFT),
    MOUSE_BUTTONFLAG_RIGHT  = (1 << MOUSE_BUTTON_RIGHT),
    MOUSE_BUTTONFLAG_MIDDLE = (1 << MOUSE_BUTTON_MIDDLE)
  };

  enum KeyCode
  {
    KEY_UNKNOWN           = GLFW_KEY_UNKNOWN,
    KEY_SPACE             = GLFW_KEY_SPACE,
    KEY_APOSTROPHE        = GLFW_KEY_APOSTROPHE, /* ' */
    KEY_LEFT_PARENTHESIS  = 40,                  /* ( */
    KEY_RIGHT_PARENTHESIS = 41,                  /* ) */
    KEY_ASTERISK          = 42,                  /* * */
    KEY_PLUS              = 43,                  /* + */
    KEY_COMMA             = GLFW_KEY_COMMA,      /* , */
    KEY_MINUS             = GLFW_KEY_MINUS,      /* - */
    KEY_PERIOD            = GLFW_KEY_PERIOD,     /* . */
    KEY_SLASH             = GLFW_KEY_SLASH,      /* / */
    KEY_0                 = GLFW_KEY_0,
    KEY_1                 = GLFW_KEY_1,
    KEY_2                 = GLFW_KEY_2,
    KEY_3                 = GLFW_KEY_3,
    KEY_4                 = GLFW_KEY_4,
    KEY_5                 = GLFW_KEY_5,
    KEY_6                 = GLFW_KEY_6,
    KEY_7                 = GLFW_KEY_7,
    KEY_8                 = GLFW_KEY_8,
    KEY_9                 = GLFW_KEY_9,
    KEY_COLON             = 58,                 /* : */
    KEY_SEMICOLON         = GLFW_KEY_SEMICOLON, /* ; */
    KEY_LESS              = 60,                 /* < */
    KEY_EQUAL             = GLFW_KEY_EQUAL,     /* = */
    KEY_GREATER           = 62,                 /* > */
    KEY_A                 = GLFW_KEY_A,
    KEY_B                 = GLFW_KEY_B,
    KEY_C                 = GLFW_KEY_C,
    KEY_D                 = GLFW_KEY_D,
    KEY_E                 = GLFW_KEY_E,
    KEY_F                 = GLFW_KEY_F,
    KEY_G                 = GLFW_KEY_G,
    KEY_H                 = GLFW_KEY_H,
    KEY_I                 = GLFW_KEY_I,
    KEY_J                 = GLFW_KEY_J,
    KEY_K                 = GLFW_KEY_K,
    KEY_L                 = GLFW_KEY_L,
    KEY_M                 = GLFW_KEY_M,
    KEY_N                 = GLFW_KEY_N,
    KEY_O                 = GLFW_KEY_O,
    KEY_P                 = GLFW_KEY_P,
    KEY_Q                 = GLFW_KEY_Q,
    KEY_R                 = GLFW_KEY_R,
    KEY_S                 = GLFW_KEY_S,
    KEY_T                 = GLFW_KEY_T,
    KEY_U                 = GLFW_KEY_U,
    KEY_V                 = GLFW_KEY_V,
    KEY_W                 = GLFW_KEY_W,
    KEY_X                 = GLFW_KEY_X,
    KEY_Y                 = GLFW_KEY_Y,
    KEY_Z                 = GLFW_KEY_Z,
    KEY_LEFT_BRACKET      = GLFW_KEY_LEFT_BRACKET,  /* [ */
    KEY_BACKSLASH         = GLFW_KEY_BACKSLASH,     /* \ */
    KEY_RIGHT_BRACKET     = GLFW_KEY_RIGHT_BRACKET, /* ] */
    KEY_GRAVE_ACCENT      = GLFW_KEY_GRAVE_ACCENT,  /* ` */
    KEY_WORLD_1           = GLFW_KEY_WORLD_1,       /* non-US #1 */
    KEY_WORLD_2           = GLFW_KEY_WORLD_2,       /* non-US #2 */
    /* Function keys */
    KEY_ESCAPE        = GLFW_KEY_ESCAPE,
    KEY_ENTER         = GLFW_KEY_ENTER,
    KEY_TAB           = GLFW_KEY_TAB,
    KEY_BACKSPACE     = GLFW_KEY_BACKSPACE,
    KEY_INSERT        = GLFW_KEY_INSERT,
    KEY_DELETE        = GLFW_KEY_DELETE,
    KEY_RIGHT         = GLFW_KEY_RIGHT,
    KEY_LEFT          = GLFW_KEY_LEFT,
    KEY_DOWN          = GLFW_KEY_DOWN,
    KEY_UP            = GLFW_KEY_UP,
    KEY_PAGE_UP       = GLFW_KEY_PAGE_UP,
    KEY_PAGE_DOWN     = GLFW_KEY_PAGE_DOWN,
    KEY_HOME          = GLFW_KEY_HOME,
    KEY_END           = GLFW_KEY_END,
    KEY_CAPS_LOCK     = GLFW_KEY_CAPS_LOCK,
    KEY_SCROLL_LOCK   = GLFW_KEY_SCROLL_LOCK,
    KEY_NUM_LOCK      = GLFW_KEY_NUM_LOCK,
    KEY_PRINT_SCREEN  = GLFW_KEY_PRINT_SCREEN,
    KEY_PAUSE         = GLFW_KEY_PAUSE,
    KEY_F1            = GLFW_KEY_F1,
    KEY_F2            = GLFW_KEY_F2,
    KEY_F3            = GLFW_KEY_F3,
    KEY_F4            = GLFW_KEY_F4,
    KEY_F5            = GLFW_KEY_F5,
    KEY_F6            = GLFW_KEY_F6,
    KEY_F7            = GLFW_KEY_F7,
    KEY_F8            = GLFW_KEY_F8,
    KEY_F9            = GLFW_KEY_F9,
    KEY_F10           = GLFW_KEY_F10,
    KEY_F11           = GLFW_KEY_F11,
    KEY_F12           = GLFW_KEY_F12,
    KEY_F13           = GLFW_KEY_F13,
    KEY_F14           = GLFW_KEY_F14,
    KEY_F15           = GLFW_KEY_F15,
    KEY_F16           = GLFW_KEY_F16,
    KEY_F17           = GLFW_KEY_F17,
    KEY_F18           = GLFW_KEY_F18,
    KEY_F19           = GLFW_KEY_F19,
    KEY_F20           = GLFW_KEY_F20,
    KEY_F21           = GLFW_KEY_F21,
    KEY_F22           = GLFW_KEY_F22,
    KEY_F23           = GLFW_KEY_F23,
    KEY_F24           = GLFW_KEY_F24,
    KEY_F25           = GLFW_KEY_F25,
    KEY_KP_0          = GLFW_KEY_KP_0,
    KEY_KP_1          = GLFW_KEY_KP_1,
    KEY_KP_2          = GLFW_KEY_KP_2,
    KEY_KP_3          = GLFW_KEY_KP_3,
    KEY_KP_4          = GLFW_KEY_KP_4,
    KEY_KP_5          = GLFW_KEY_KP_5,
    KEY_KP_6          = GLFW_KEY_KP_6,
    KEY_KP_7          = GLFW_KEY_KP_7,
    KEY_KP_8          = GLFW_KEY_KP_8,
    KEY_KP_9          = GLFW_KEY_KP_9,
    KEY_KP_DECIMAL    = GLFW_KEY_KP_DECIMAL,
    KEY_KP_DIVIDE     = GLFW_KEY_KP_DIVIDE,
    KEY_KP_MULTIPLY   = GLFW_KEY_KP_MULTIPLY,
    KEY_KP_SUBTRACT   = GLFW_KEY_KP_SUBTRACT,
    KEY_KP_ADD        = GLFW_KEY_KP_ADD,
    KEY_KP_ENTER      = GLFW_KEY_KP_ENTER,
    KEY_KP_EQUAL      = GLFW_KEY_KP_EQUAL,
    KEY_LEFT_SHIFT    = GLFW_KEY_LEFT_SHIFT,
    KEY_LEFT_CONTROL  = GLFW_KEY_LEFT_CONTROL,
    KEY_LEFT_ALT      = GLFW_KEY_LEFT_ALT,
    KEY_LEFT_SUPER    = GLFW_KEY_LEFT_SUPER,
    KEY_RIGHT_SHIFT   = GLFW_KEY_RIGHT_SHIFT,
    KEY_RIGHT_CONTROL = GLFW_KEY_RIGHT_CONTROL,
    KEY_RIGHT_ALT     = GLFW_KEY_RIGHT_ALT,
    KEY_RIGHT_SUPER   = GLFW_KEY_RIGHT_SUPER,
    KEY_MENU          = GLFW_KEY_MENU,
    KEY_LAST          = GLFW_KEY_LAST,
  };

  enum KeyModifiers
  {
    KMOD_SHIFT   = GLFW_MOD_SHIFT,
    KMOD_CONTROL = GLFW_MOD_CONTROL,
    KMOD_ALT     = GLFW_MOD_ALT,
    KMOD_SUPER   = GLFW_MOD_SUPER,
  };

  //////////////////////////////////////////////////////////////////////////

  GLFWwindow* m_internal = nullptr;
  std::string m_windowName;


  inline bool pollEvents() // returns false on exit, can do while(pollEvents()){ ... }
  {
    NVPSystem::pollEvents();
    return !isClosing();
  }
  inline void        waitEvents() { NVPSystem::waitEvents(); }
  inline double      getTime() { return NVPSystem::getTime(); }
  inline std::string exePath() { return NVPSystem::exePath(); }

  // Accessors
  inline int getWidth() const { return m_windowSize[0]; }
  inline int getHeight() const { return m_windowSize[1]; }
  inline int getMouseWheel() const { return m_mouseWheel; }
  inline int getKeyModifiers() const { return m_keyModifiers; }
  inline int getMouseX() const { return m_mouseX; }
  inline int getMouseY() const { return m_mouseY; }

  void        setTitle(const char* title);
  void        setFullScreen(bool bYes);
  void        setWindowPos(int x, int y);
  void        setWindowSize(int w, int h);
  inline void setKeyModifiers(int m) { m_keyModifiers = m; }
  inline void setMouse(int x, int y)
  {
    m_mouseX = x;
    m_mouseY = y;
  }

  inline bool isFullScreen() const { return m_isFullScreen; }
  inline bool isClosing() const { return m_isClosing || glfwWindowShouldClose(m_internal); }
  inline bool isOpen() const
  {
    return glfwGetWindowAttrib(m_internal, GLFW_VISIBLE) == GLFW_TRUE
           && glfwGetWindowAttrib(m_internal, GLFW_ICONIFIED) == GLFW_FALSE && !isClosing();
  }

  virtual bool open(int posX, int posY, int width, int height, const char* title, bool requireGLContext);  // creates internal window and opens it
  void deinit();                                                            // destroys internal window

  void close();  //  triggers closing event, still needs deinit for final cleanup
  void maximize();
  void restore();
  void minimize();

  // uses operating system specific code for sake of debugging/automated testing
  void        screenshot(const char* filename);
  void        clear(uint32_t r, uint32_t g, uint32_t b);
  std::string openFileDialog(const char* title, const char* exts);

  // derived windows/apps should override to handle events
  virtual void onWindowClose() {}
  virtual void onWindowResize(int w, int h) {}
  virtual void onWindowRefresh() {}
  virtual void onMouseMotion(int x, int y) {}
  virtual void onMouseWheel(int delta) {}
  virtual void onMouseButton(MouseButton button, ButtonAction action, int mods, int x, int y) {}
  virtual void onKeyboard(KeyCode key, ButtonAction action, int mods, int x, int y) {}
  virtual void onKeyboardChar(unsigned char key, int mods, int x, int y) {}
  virtual void onDragDrop(int num, const char** paths) {}

  // derived windows/apps should override these. Essentially used for remote-control (via sockets)
  // the decoded remote paquets would invoke these methods. See shared_sources\nvsockets\socketSampleMessages.cpp
  virtual void requestTiming() {
  }  // the app can override it to return requested timing information over sockets : use sysPostTiming() below
  virtual void requestPaint() {}                       // the app needs to refresh once the window
  virtual void requestContinuousRefresh(bool bYes) {}  // the app might swith on/off the continuous rendering
  virtual void requestSetArg(char token, int arg0, int arg1, int arg2, int arg3) {
  }  // the app receives arbitrary params from remote, free of interpretation
  virtual void requestSetArg(char token, float arg0, float arg1, float arg2, float arg3) {
  }  // the app receives arbitrary params from remote, free of interpretation

private:
  int  m_mouseX;
  int  m_mouseY;
  int  m_mouseWheel;
  int  m_windowSize[2];
  int  m_keyModifiers;
  bool m_isFullScreen = false;
  bool m_isClosing    = false;
  int  m_preFullScreenPos[2];
  int  m_preFullScreenSize[2];

  static void cb_windowrefreshfun(GLFWwindow* glfwwin);
  static void cb_windowsizefun(GLFWwindow* glfwwin, int w, int h);
  static void cb_windowclosefun(GLFWwindow* glfwwin);
  static void cb_mousebuttonfun(GLFWwindow* glfwwin, int button, int action, int mods);
  static void cb_cursorposfun(GLFWwindow* glfwwin, double x, double y);
  static void cb_scrollfun(GLFWwindow* glfwwin, double x, double y);
  static void cb_keyfun(GLFWwindow* glfwwin, int key, int scancode, int action, int mods);
  static void cb_charfun(GLFWwindow* glfwwin, unsigned int codepoint);
  static void cb_dropfun(GLFWwindow* glfwwin, int count, const char** paths);
};


#endif
