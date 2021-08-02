/*-----------------------------------------------------------------------
Copyright (c) 2018, NVIDIA. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
  * Neither the name of its contributors may be used to endorse
    or promote products derived from this software without specific
    prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-----------------------------------------------------------------------*/

#ifndef NV_IMGUI_INCLUDED
#define NV_IMGUI_INCLUDED

#define IMGUI_DISABLE_OBSOLETE_FUNCTIONS 1
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "imgui_internal.h"

#include <nvpwindow.hpp>

#include <algorithm>
#include <climits>
#include <functional>
#include <string>
#include <vector>

namespace ImGuiH {
//////////////////////////////////////////////////////////////////////////

// NVPWindow callbacks

inline bool mouse_pos(int x, int y)
{
  auto& io    = ImGui::GetIO();
  io.MousePos = ImVec2(float(x), float(y));
  return io.WantCaptureMouse;
}

inline bool mouse_button(int button, int action)
{
  auto& io             = ImGui::GetIO();
  io.MouseDown[button] = action == NVPWindow::BUTTON_PRESS;
  return io.WantCaptureMouse;
}

inline bool mouse_wheel(int wheel)
{
  auto& io      = ImGui::GetIO();
  io.MouseWheel = static_cast<float>(wheel);
  return io.WantCaptureMouse;
}

inline bool key_char(int button)
{
  auto& io = ImGui::GetIO();
  io.AddInputCharacter(static_cast<unsigned int>(button));
  return io.WantCaptureKeyboard;
}

inline bool key_button(int button, int action, int mods)
{
  if(button == NVPWindow::KEY_KP_ENTER)
  {
    button = NVPWindow::KEY_ENTER;
  }

  auto& io            = ImGui::GetIO();
  io.KeyCtrl          = (mods & NVPWindow::KMOD_CONTROL) != 0;
  io.KeyShift         = (mods & NVPWindow::KMOD_SHIFT) != 0;
  io.KeyAlt           = (mods & NVPWindow::KMOD_ALT) != 0;
  io.KeySuper         = (mods & NVPWindow::KMOD_SUPER) != 0;
  io.KeysDown[button] = action == NVPWindow::BUTTON_PRESS;
  return io.WantCaptureKeyboard;
}

//////////////////////////////////////////////////////////////////////////

enum FontMode
{
  FONT_FIXED,
  FONT_PROPORTIONAL_SCALED,
  FONT_MONOSPACED_SCALED,
};

void Init(int width, int height, void* userData, FontMode fontmode = FONT_FIXED);

template <typename T>
bool Clamped(bool changed, T* value, T min, T max)
{
  *value = std::max(min, std::min(max, *value));
  return changed;
}

inline bool InputIntClamped(const char*         label,
                            unsigned int*       v,
                            int                 min       = INT_MIN,
                            int                 max       = INT_MAX,
                            int                 step      = 1,
                            int                 step_fast = 100,
                            ImGuiInputTextFlags flags     = 0)
{
  return Clamped(ImGui::InputInt(label, (int*)v, step, step_fast, flags), (int*)v, min, max);
}

inline bool InputIntClamped(const char* label, int* v, int min = INT_MIN, int max = INT_MAX, int step = 1, int step_fast = 100, ImGuiInputTextFlags flags = 0)
{
  return Clamped(ImGui::InputInt(label, v, step, step_fast, flags), v, min, max);
}

inline bool InputFloatClamped(const char*         label,
                              float*              v,
                              float               min       = 0.0,
                              float               max       = 1.0,
                              float               step      = 0.1f,
                              float               step_fast = 1.0f,
                              const char*         fmt       = "%.3f",
                              ImGuiInputTextFlags flags     = 0)
{
  return Clamped(ImGui::InputFloat(label, v, step, step_fast, fmt, flags), v, min, max);
}

enum ValueType
{
  TYPE_INT,
  TYPE_FLOAT,
  TYPE_BOOL
};


struct Enum
{
  union
  {
    int   ivalue;
    float fvalue;
    bool  bvalue;
  };
  std::string name;
};

void Combo(const char*     label,
           size_t          numEnums,
           const Enum*     enums,
           void*           value,
           ImGuiComboFlags flags        = 0,
           ValueType       valueType    = TYPE_INT,
           bool*           valueChanged = NULL);

class Registry
{
private:
  struct Entry
  {
    std::vector<Enum> enums;
    ValueType         valueType;
    bool              valueChanged;
  };
  std::vector<Entry> entries;

public:
  const std::vector<Enum>& getEnums(uint32_t type) const { return entries[type].enums; }

  void checkboxAdd(uint32_t type, bool value = false)
  {
    if(type >= entries.size())
    {
      entries.resize(type + 1);
    }
    entries[type].enums.push_back({value, std::string("Bool")});
    entries[type].valueChanged = false;
    entries[type].valueType    = TYPE_BOOL;
  }

  void inputIntAdd(uint32_t type, int value = 0)
  {
    if(type >= entries.size())
    {
      entries.resize(type + 1);
    }
    entries[type].enums.push_back({value, std::string("iI")});
    entries[type].valueChanged = false;
    entries[type].valueType    = TYPE_INT;
  }

  void inputFloatAdd(uint32_t type, float value = .0f)
  {
    if(type >= entries.size())
    {
      entries.resize(type + 1);
    }
    Enum e;
    e.fvalue = value;
    e.name   = std::string("iF");
    entries[type].enums.push_back(e);
    entries[type].valueChanged = false;
    entries[type].valueType    = TYPE_FLOAT;
  }

  void enumAdd(uint32_t type, int value, const char* name)
  {
    if(type >= entries.size())
    {
      entries.resize(type + 1);
    }
    entries[type].enums.push_back({value, name});
    entries[type].valueChanged = false;
    entries[type].valueType = TYPE_INT;  // the user must be consistent so that he adds only the same type for the same combo !
  }

  void enumAdd(uint32_t type, float value, const char* name)
  {
    if(type >= entries.size())
    {
      entries.resize(type + 1);
    }
    Enum e;
    e.fvalue = value;
    e.name   = name;
    entries[type].enums.push_back(e);
    entries[type].valueChanged = false;
    entries[type].valueType = TYPE_FLOAT;  // the user must be consistent so that he adds only the same type for the same combo !
  }

  void enumReset(uint32_t type)
  {
    if(type < entries.size())
    {
      entries[type].enums.clear();
      entries[type].valueChanged = false;
      entries[type].valueType    = TYPE_INT;
    }
  }

  void enumCombobox(uint32_t type, const char* label, void* value, ImGuiComboFlags flags = 0, bool* valueChanged = NULL)
  {
    Combo(label, entries[type].enums.size(), entries[type].enums.data(), value, flags, entries[type].valueType,
          &entries[type].valueChanged);
    if(valueChanged)
    {
      *valueChanged = entries[type].valueChanged;
    }
  }

  void checkbox(uint32_t type, const char* label, void* value, ImGuiComboFlags flags = 0, bool* valueChanged = NULL)
  {
    bool* pB        = &entries[type].enums[0].bvalue;
    bool  prevValue = *pB;
    ImGui::Checkbox(label, pB);
    if(prevValue != *pB)
    {
      entries[type].valueChanged = true;
    }
    if(valueChanged)
    {
      *valueChanged = entries[type].valueChanged;
    }
    if(value)
    {
      ((bool*)value)[0] = pB[0];
    }
  }

  void inputIntClamped(uint32_t type, const char* label, int* v, int min = INT_MIN, int max = INT_MAX, int step = 1, int step_fast = 100, ImGuiInputTextFlags flags = 0)
  {
    int* pI        = &entries[type].enums[0].ivalue;
    int  prevValue = *pI;
    bool bRes      = Clamped(ImGui::InputInt(label, pI, step, step_fast, flags), pI, min, max);
    if(prevValue != *pI)
    {
      entries[type].valueChanged = true;
    }
    if(v)
    {
      ((int*)v)[0] = pI[0];
    }
  }

  //void inputFloatClamped(uint32_t type, const char* label, float* v, float min = 0.0f, int max = 100.0f, int step = 0.1, int step_fast = 1.0, ImGuiInputTextFlags flags = 0) {
  //    float *pF = &entries[type].enums[0].fvalue;
  //    float prevValue = *pF;
  //    bool bRes = Clamped(ImGui::InputFloat(label, pF, step, step_fast, flags), pF, min, max);
  //    if (prevValue != *pF)
  //        entries[type].valueChanged = true;
  //    if (v)
  //        ((float*)v)[0] = pF[0];
  //}

  bool checkValueChange(uint32_t type, bool reset = true)
  {
    bool changed = entries[type].valueChanged;
    if(reset && changed)
    {
      entries[type].valueChanged = false;
    }
    return changed;
  }
};

//--------------------------------------------------------------------------------------------------
//
// If GLFW has been initialized, returns the DPI scale of the primary monitor. Otherwise, returns 1.
//
float getDPIScale();

//--------------------------------------------------------------------------------------------------
//
// Setting common style across samples
//
void setStyle();

//--------------------------------------------------------------------------------------------------
//
// Setting nicer default fonts
//
void setFonts(FontMode fontmode = FONT_PROPORTIONAL_SCALED);


//--------------------------------------------------------------------------------------------------
// Creating a window panel
// - Panel will be on the left or the right side of the window.
// - It fills the height of the window, and stays on the side it was created.
// - Style can be overwritten using ImGuiH::Panel::settings
class Panel /*static*/
{
public:
  // Side where the panel will be
  enum class Side
  {
    Left,
    Right,
  };

  // Default style for the panel. Override those values
  struct Style
  {

    ImGuiWindowFlags flags{ImGuiWindowFlags_NoTitleBar};  // No header, no collapse
    ImVec2           panel_offset{300, 0};                // 300 pixel wide
    float            bgAlpha{0.8f};                       // Slightly transparent
    float            panel_min_width{100.f};              // 100 pixel wide
    float            panel_max_perc{0.5};                 // Max width in percent of the window
  };
  static Style style;

  // Starting the panel, equivalent to ImGui::Begin for a window. Need ImGui::end()
  static void Begin(Side side = Side::Right)
  {
    auto& io = ImGui::GetIO();

    ImVec2&          pw    = style.panel_offset;
    ImGuiWindowFlags flags = style.flags;
    ImVec2           ds    = io.DisplaySize;
    float            pos_x = side == Side::Left ? 0 : ds.x - pw.x;


    // Window characteristics
    ImGui::SetNextWindowBgAlpha(style.bgAlpha);
    ImGui::SetNextWindowPos(ImVec2(pos_x, pw.y), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(pw.x, ds.y), ImGuiCond_Always);
    ImGui::SetNextWindowSizeConstraints(ImVec2(style.panel_min_width, -1.0f), ImVec2(ds.x * style.panel_max_perc, -1.0f));

    // Begin window
    ImGui::Begin("##_Panel", nullptr, flags);
    {
      style.panel_offset.x = ImGui::GetWindowSize().x;
    }
  }

  // Mirror begin but can use directly End()
  static void End() { ImGui::End(); }
};


//--------------------------------------------------------------------------------------------------
// GUI control elements which have a 'property page' look.
// - Name on the left, justified on the right
// - Value filling the right side
// - Support for 1..4 components, int, float, bool
// - Tooltip over all param name for the description
// - Default values  (reset to default)
// - Min / Max for all value widgets
//
// Controls:
//    Slider    - Slider with bounded values
//    Drag      - Infinite values, but min or max values can be set
//    Checkbox  - Simple [x]
//    Selection - Combobox with simple vector of string or fct callback
//    Text      - Text edit box to enter/modify text
//    Color     - Color picker
//    Button    - param + button, or just a button when param name is empty
//    Group     - Grouping a list of elements (collapsible)
//    Info      - Simple info text
class Control /*static*/
{
public:
  // Influences the rendering of elements.
  enum class Flags
  {
    Normal   = 0,       // Normal display
    Disabled = 1 << 0,  // Grayed out
  };

  struct Style
  {
    float ctrlPerc    = 0.7f;   // Percentage the value control takes in the pane
    float ctrlMaxSize = 500.f;  // Max pixel size control will take
    float ctrlMinSize = 50.f;   // Minimum size of the control
    float dragMinSize = 150.f;  // Minimum size to be on one line
  };

  static Style style;

public:
  // Slider - used for bounded values
  template <typename TValue, typename TTooltip>
  static bool Slider(const std::string& label,        // Name of the parameter
                     TTooltip           description,  // Tooltip info
                     TValue*            value,        // value
                     const void*        default_value = nullptr,
                     Flags              flags         = Flags::Normal,
                     TValue             min           = TValue(0),
                     TValue             max           = TValue(1),
                     const char*        format        = nullptr)
  {
    return show_numeric_control(label, description, value, default_value, flags,
                                [&] { return show_slider_control<TValue>(value, min, max, format); });
  }

  // Drag - used for unbound float values. Min and max (or only one) can be specified.
  template <typename TValue, typename TTooltip>
  static bool Drag(const std::string& label,
                   TTooltip           description,
                   TValue*            value,
                   const void*        default_value = nullptr,
                   Flags              flags         = Flags::Normal,
                   TValue             min           = std::numeric_limits<TValue>::lowest(),
                   TValue             max           = std::numeric_limits<TValue>::max(),
                   float              speed         = std::is_integral<TValue>::value ? 0.25f : 0.01f,
                   const char*        format        = nullptr)
  {
    return show_numeric_control(label, description, value, default_value, flags,
                                [&] { return show_drag_control<TValue>(value, speed, min, max, format); });
  }
  // Checkbox  -  to toggle options on and off.
  template <typename TTooltip>
  static bool Checkbox(const std::string& label, TTooltip description, bool* value, const bool* default_value = nullptr, Flags flags = Flags::Normal);

  // Combobox(1)  -  to select between options.
  template <typename Tinteger, typename TTooltip>
  static bool Selection(const std::string&                   label,
                        TTooltip                             description,
                        Tinteger*                            index,
                        const void*                          default_index,
                        Flags                                flags,
                        std::function<const char*(Tinteger)> get_value);

  // Combobox(2)  -  to select between options.
  template <typename Tinteger, typename TTooltip>
  static bool Selection(const std::string&       label,
                        TTooltip                 description,
                        Tinteger*                index,
                        const void*              default_index,
                        Flags                    flags,
                        std::vector<std::string> values)
  {
    return Selection<Tinteger>(label, description, index, default_index, flags,
                               [&](Tinteger i) { return i < values.size() ? values[i].c_str() : nullptr; });
  }

  // Text  -  to input text
  template <typename TTooltip>
  static bool Text(const std::string& label,
                   TTooltip           description,
                   char*              value_buffer,
                   size_t             value_buffer_size,
                   const char*        default_value,
                   Flags              flags = Flags::Normal);

  // Pick  -  to pick a color.
  template <typename TTooltip>
  static bool Color(const std::string& label,
                    TTooltip           description,
                    float*             value_float3,
                    const float*       default_value_float3 = nullptr,
                    Flags              flags                = Flags::Normal);

  // Button  -  to start an action or for navigation.
  template <typename TTooltip>
  static bool button(const std::string& label, const std::string& button_text, TTooltip description, Flags flags = Flags::Normal);

  // Group  -  Create a collapsible group that can nest other elements.
  template <typename TReturn>
  static TReturn Group(const std::string& name, bool expanded_by_default, std::function<TReturn(void)> show_content);

  // Info  -  Shows an text label only without interaction possibility.
  template <typename TTooltip>
  static void Info(const std::string& label, TTooltip description, const char* info_text, Flags flags = Flags::Normal);
  template <typename TTooltip>
  static void Info(const std::string& label, TTooltip description, const std::string& info_text, Flags flags = Flags::Normal)
  {
    Info(label, description, info_text.c_str(), flags);
  }


  template <typename TTooltip>
  static bool Custom(const std::string& label, TTooltip description, std::function<bool()> show_content, Flags flags = Flags::Normal);

private:
  template <typename TTooltip>
  static void show_tooltip(TTooltip description);

  template <typename TRet>
  static void show_tooltip(const std::function<TRet()>& action)
  {
    action();
  }

  template <typename TTooltip>
  static void show_property_label(const std::string& text, TTooltip description, Flags flags);

  template <typename TValue>
  static bool show_slider_control(TValue* value, TValue& min, TValue& max, const char* format);

  template <typename TValue>
  static bool show_drag_control(TValue* value, float speed, TValue& min, TValue& max, const char* format);

  template <typename TValue, typename TTooltip>
  static bool show_numeric_control(const std::string&        label,
                                   TTooltip                  description,
                                   TValue*                   value,
                                   const void*               default_value,
                                   Flags                     flags,
                                   std::function<bool(void)> show_numeric_control);

  template <typename TScalar, ImGuiDataType type, uint8_t dim>
  static bool show_drag_control_scalar(TScalar* value, float speed, TScalar* min, TScalar* max, const char* format);

  template <typename TScalar, ImGuiDataType type, uint8_t dim>
  static bool show_slider_control_scalar(TScalar* value, TScalar* min, TScalar* max, const char* format);
};


//////////////////////////////////////////////////////////////////////////
// Template IMPLEMENTATION
//


//
// Displaying the parameter name
// - When string ends with \n, the parameters will be on next line instead of on the right
//
template <typename TTooltip>
void ImGuiH::Control::show_property_label(const std::string& text, TTooltip description, Flags flags)
{
  if(flags == Flags::Disabled)
  {
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
  }

  if(text.back() == '\n')
  {
    ImGui::TextWrapped(text.c_str());
  }
  else
  {
    // Indenting the text to be right justified
    float              current_indent = ImGui::GetCursorPos().x;
    const ImGuiStyle&  imstyle        = ImGui::GetStyle();
    const ImGuiWindow* window         = ImGui::GetCurrentWindow();

    float control_width = std::min((ImGui::GetWindowWidth() - imstyle.IndentSpacing) * style.ctrlPerc, style.ctrlMaxSize);
    control_width -= window->ScrollbarSizes.x;
    control_width = std::max(control_width, style.ctrlMinSize);

    float  available_width     = ImGui::GetContentRegionAvail().x;
    float  avaiable_text_width = available_width - control_width - imstyle.ItemInnerSpacing.x;
    ImVec2 text_size = ImGui::CalcTextSize(text.c_str(), text.c_str() + text.size(), false, avaiable_text_width);
    float  indent    = current_indent + available_width - control_width - text_size.x - imstyle.ItemInnerSpacing.x;

    ImGui::AlignTextToFramePadding();
    ImGui::NewLine();
    ImGui::SameLine(indent);
    ImGui::PushTextWrapPos(indent + avaiable_text_width);
    ImGui::TextWrapped(text.c_str());
    ImGui::PopTextWrapPos();
    ImGui::SameLine();
  }

  if(flags == Flags::Disabled)
    ImGui::PopStyleColor();

  if(ImGui::IsItemHovered())
    show_tooltip(description);
}

//
// Add the value controls (sliders, drag)
// - handle default value
//
template <typename TValue, typename TTooltip>
bool ImGuiH::Control::show_numeric_control(const std::string&        label,
                                           TTooltip                  description,
                                           TValue*                   value,
                                           const void*               default_value,
                                           Flags                     flags,
                                           std::function<bool(void)> show_numeric_control)
{
  ImGui::PushID(value);
  show_property_label(label, description, flags);
  if(flags == Flags::Disabled)
  {
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
  }
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  bool changed = show_numeric_control();

  if(default_value && ImGui::BeginPopupContextItem("item context menu"))
  {
    if(ImGui::Selectable("set default"))
    {
      changed = *value != *(reinterpret_cast<const TValue*>(default_value));
      *value  = *(reinterpret_cast<const TValue*>(default_value));
    }
    ImGui::EndPopup();
  }

  if(flags == Flags::Disabled)
  {
    ImGui::PopItemFlag();
    ImGui::PopStyleVar();
  }

  ImGui::PopID();
  return changed;
}

//
//
template <typename TTooltip>
void ImGuiH::Control::Info(const std::string& label, TTooltip description, const char* info_text, Flags flags)
{
  ImGui::PushID(&info_text);
  if(label.empty() == false)
    show_property_label(label, description, flags);
  if(flags == Flags::Disabled)
  {
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
  }
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  ImGui::TextWrapped(info_text);

  if(flags == Flags::Disabled)
  {
    ImGui::PopItemFlag();
    ImGui::PopStyleVar();
  }

  if(ImGui::IsItemHovered() && label.empty())
    show_tooltip(description);

  ImGui::PopID();
}

template <typename TReturn>
TReturn ImGuiH::Control::Group(const std::string& name, bool expanded_by_default, std::function<TReturn(void)> show_content)
{
  ImDrawList* draw_list = ImGui::GetWindowDrawList();
  ImVec4      color     = ImGui::GetStyle().Colors[ImGuiCol_Text];
  color.x *= 0.6f;
  color.y *= 0.6f;
  color.z *= 0.6f;
  const ImU32 draw_color      = ImColor(color);
  const float draw_line_width = 1.0f;

  ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
  ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));

  TReturn      ret       = static_cast<TReturn>(0);
  const ImVec2 start_top = ImGui::GetCursorScreenPos();
  ImGui::PushID(&show_content);
  if(ImGui::TreeNodeEx(name.c_str(), expanded_by_default ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_None))
  {
    ImGui::PopStyleColor();
    ImGui::PopStyleColor();

    const ImVec2 start = ImGui::GetCursorScreenPos();
    ret                = show_content();
    const ImVec2 end   = ImGui::GetCursorScreenPos();

    draw_list->AddLine(ImVec2(start.x - draw_line_width * 0.5f, (start.y + start_top.y) * 0.5f),
                       ImVec2(end.x - draw_line_width * 0.5f, end.y), draw_color, draw_line_width);

    ImGui::TreePop();
  }
  else
  {
    ImGui::PopStyleColor();
    ImGui::PopStyleColor();
  }
  ImGui::PopID();
  ImGui::Spacing();
  return ret;
}

template <typename TTooltip>
bool ImGuiH::Control::button(const std::string& label, const std::string& button_text, TTooltip description, Flags flags /*= Flags::Normal*/)
{
  ImGui::PushID(&description);

  // show the left side label with tool tips
  if(!label.empty())
    show_property_label(label, description, flags);

  bool label_fits = (ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(button_text.c_str()).x
                     - ImGui::GetStyle().ItemInnerSpacing.x * 2.0f)
                    > 0.0f;

  if(flags == Flags::Disabled)
  {
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
  }
  bool pressed = ImGui::Button((label.empty() || label_fits ? button_text.c_str() : "..."),
                               ImVec2(ImGui::GetContentRegionAvail().x, 0.0f));

  if(flags == Flags::Disabled)
  {
    ImGui::PopItemFlag();
    ImGui::PopStyleVar();
  }

  // show the tool tip when hovering over the button in case there is no left side label
  if(label.empty() && ImGui::IsItemHovered())
    show_tooltip(description);

  ImGui::PopID();
  return pressed;
}

template <typename TTooltip>
bool ImGuiH::Control::Color(const std::string& label, TTooltip description, float* value_float3, const float* default_value_float3, Flags flags)
{
  ImGui::PushID(value_float3);
  show_property_label(label, description, flags);
  if(flags == Flags::Disabled)
  {
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
  }
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  bool changed = ImGui::ColorEdit3("##hidden", &value_float3[0],
                                   ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_Float | ImGuiColorEditFlags_NoAlpha);

  if(default_value_float3 && ImGui::BeginPopupContextItem("item context menu"))
  {
    if(ImGui::Selectable("set default"))
    {
      changed = value_float3[0] != default_value_float3[0] || value_float3[1] != default_value_float3[1]
                || value_float3[2] != default_value_float3[2];

      value_float3[0] = default_value_float3[0];
      value_float3[1] = default_value_float3[1];
      value_float3[2] = default_value_float3[2];
    }
    ImGui::EndPopup();
  }
  if(flags == Flags::Disabled)
  {
    ImGui::PopItemFlag();
    ImGui::PopStyleVar();
  }
  ImGui::PopID();
  return changed;
}


// Combo-box to select between options.
template <typename Tinteger, typename TTooltip>
bool ImGuiH::Control::Selection(const std::string&                   label,
                                TTooltip                             description,
                                Tinteger*                            index,
                                const void*                          default_index,
                                Flags                                flags,
                                std::function<const char*(Tinteger)> get_value)
{
  return show_numeric_control(label, description, index, default_index, flags, [&] {
    bool valid   = false;
    bool changed = false;
    if(ImGui::BeginCombo("##hidden", get_value(*index)))
    {
      Tinteger i = 0;
      while(true)
      {
        const char* option = get_value(i);
        if(!option)
          break;

        valid |= (i == *index);  // check if current selection is a valid option
        if(ImGui::Selectable(option, i == *index))
        {
          *index  = i;
          changed = true;
          valid   = true;
        }
        i++;
      }
      ImGui::EndCombo();

      if(!valid && default_index)
      {
        *index  = *reinterpret_cast<const Tinteger*>(default_index);
        changed = true;
      }
    }
    return changed;
  });
}

template <typename TTooltip>
bool ImGuiH::Control::Text(const std::string& label, TTooltip description, char* value_buffer, size_t value_buffer_size, const char* default_value, Flags flags)
{
  //assuming that this is unique, but constant from frame to frame
  int id = *static_cast<int*>((void*)&label);
  id ^= *static_cast<int*>((void*)&description);
  id ^= *static_cast<int*>((void*)&default_value);

  ImGui::PushID(id);
  show_property_label(label, description, flags);
  if(flags == Flags::Disabled)
  {
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
  }
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  bool changed = ImGui::InputText("##hidden", value_buffer, value_buffer_size, ImGuiInputTextFlags_EnterReturnsTrue);
  if(default_value && ImGui::BeginPopupContextItem("item context menu"))
  {
    if(ImGui::Selectable("set default"))
    {
      changed = strcmp(value_buffer, default_value) != 0;
      memset(value_buffer, '\0', value_buffer_size);
      strncpy(value_buffer, default_value, std::min(value_buffer_size, strlen(default_value)));
    }
    ImGui::EndPopup();
  }
  if(flags == Flags::Disabled)
  {
    ImGui::PopItemFlag();
    ImGui::PopStyleVar();
  }
  ImGui::PopID();
  return changed;
}

template <typename TTooltip>
bool ImGuiH::Control::Checkbox(const std::string& label, TTooltip description, bool* value, const bool* default_value /*= nullptr*/, Flags flags /*= Flags::Normal*/)
{
  ImGui::PushID(value);
  show_property_label(label, description, flags);
  if(flags == Flags::Disabled)
  {
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
  }
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  bool changed = ImGui::Checkbox("##hidden", value);
  if(default_value && ImGui::BeginPopupContextItem("item context menu"))
  {
    if(ImGui::Selectable("set default"))
    {
      changed = *value != *default_value;
      *value  = *default_value;
    }
    ImGui::EndPopup();
  }
  if(flags == Flags::Disabled)
  {
    ImGui::PopItemFlag();
    ImGui::PopStyleVar();
  }
  ImGui::PopID();
  return changed;
}


template <typename TScalar, ImGuiDataType type, uint8_t dim>
bool Control::show_drag_control_scalar(TScalar* value, float speed, TScalar* min, TScalar* max, const char* format)
{
  static const char* visible_labels[] = {"x:", "y:", "z:", "w:"};

  if(ImGui::GetContentRegionAvail().x > style.dragMinSize)
    return ImGui::DragScalarN("##hidden", type, &value[0], dim, speed, &min[0], &max[0], format);

  if(dim == 1)
    return ImGui::DragScalar("##hidden", type, &value[0], speed, &min[0], &max[0], format);

  float indent  = ImGui::GetCursorPos().x;
  bool  changed = false;

  for(uint8_t c = 0; c < dim; ++c)
  {
    ImGui::PushID(c);
    if(c > 0)
    {
      ImGui::NewLine();
      ImGui::SameLine(indent);
    }
    ImGui::Text("%s", visible_labels[c]);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    changed |= ImGui::DragScalar("##hidden", type, &value[c], speed, &min[c], &max[c], format);
    ImGui::PopID();
  }
  return changed;
}

template <typename TScalar, ImGuiDataType type, uint8_t dim>
bool Control::show_slider_control_scalar(TScalar* value, TScalar* min, TScalar* max, const char* format)
{
  static const char* visible_labels[] = {"x:", "y:", "z:", "w:"};

  if(dim == 1)
    return ImGui::SliderScalar("##hidden", type, &value[0], &min[0], &max[0], format);

  float indent  = ImGui::GetCursorPos().x;
  bool  changed = false;
  for(uint8_t c = 0; c < dim; ++c)
  {
    ImGui::PushID(c);
    if(c > 0)
    {
      ImGui::NewLine();
      ImGui::SameLine(indent);
    }
    ImGui::Text("%s", visible_labels[c]);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    changed |= ImGui::SliderScalar("##hidden", type, &value[c], &min[c], &max[c], format);
    ImGui::PopID();
  }
  return changed;
}

template <typename TTooltip>
bool ImGuiH::Control::Custom(const std::string& label, TTooltip description, std::function<bool()> show_content, Flags flags /*= Flags::Normal*/)
{
  //ImGui::PushID(value);
  show_property_label(label, description, flags);
  if(flags == Flags::Disabled)
  {
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
  }
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);

  bool changed = show_content();

  if(flags == Flags::Disabled)
  {
    ImGui::PopItemFlag();
    ImGui::PopStyleVar();
  }
  // ImGui::PopID();
  return changed;
}

//////////////////////////////////////////////////////////////////////////
// Property Editor like Helper
// - Entries are always between Begin/End
// - Entries are made of a 'property' and a 'value'
//
// There is one Entry with a string from both property and value,
// the other one is a string for the property and a std::function for
// the value. The last one can be implemented with a lambda, therefore
// any Imgui widget can be used on the right side.
//
// Ex:
// ImGuiH::PropertyEditor::Begin();
// ImGuiH::PropertyEditor::Entry("Total Render GPU [ms]", "0.401");
// ImGuiH::PropertyEditor::Entry("Vertex Size [MB]", "0");
// ImGuiH::PropertyEditor::Entry("Attrib Size [MB]", "0");
// ImGuiH::PropertyEditor::Entry("Custom widget",
//                               [&] { return ImGui::DragFloat3("##tt", &helloVk.m_pushConstant.lightPosition.x); });
// ImGuiH::PropertyEditor::End();
//
struct PropertyEditor
{
  // Beginning the Property Editor
  static void Begin()
  {
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
    ImGui::Columns(2);
    ImGui::Separator();
  }

  // Generic entry, the lambda function should return if the widget changed
  static bool Entry(const std::string& property_name, std::function<bool()> content_fct)
  {
    ImGui::AlignTextToFramePadding();
    ImGui::SetNextItemWidth(-1);
    ImGui::Text(property_name.c_str());
    ImGui::NextColumn();
    ImGui::AlignTextToFramePadding();
    ImGui::SetNextItemWidth(-1);
    bool result = content_fct();
    ImGui::NextColumn();
    return result;  // returning if the widget changed
  }

  // Text specialization
  static void Entry(const std::string& property_name, const std::string& value)
  {
    Entry(property_name, [&] {
      ImGui::Text(value.c_str());
      return false;  // dummy, no change
    });
  }

  // Ending the Editor
  static void End()
  {
    ImGui::Columns(1);
    ImGui::Separator();
    ImGui::PopStyleVar();
  }
};

}  // namespace ImGuiH


namespace ImGui {

struct ImPlotMulti
{
  ImGuiPlotType plot_type{ImGuiPlotType_Lines};
  const char*   name{nullptr};
  ImColor       color{0xFFFFFFFF};
  float         thickness{1.0};
  const float*  data{nullptr};
  int           values_count{0};
  int           values_offset{0};
  float         scale_min{FLT_MIN};
  float         scale_max{FLT_MAX};
};

void PlotMultiEx(const char* label, int num_datas, ImPlotMulti* datas, const char* overlay_text, ImVec2 frame_size);


}  // namespace ImGui

#endif
