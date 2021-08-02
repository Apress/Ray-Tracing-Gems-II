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
#define GLFW_INCLUDE_NONE
#include "imgui_helper.h"
#include "nvmath/nvmath.h"
#include "nvmath/nvmath_glsltypes.h"
#include <GLFW/glfw3.h>

#include <fstream>

namespace ImGuiH {

void Init(int width, int height, void* userData, FontMode fontmode)
{
  ImGui::CreateContext();
  setFonts(fontmode);
  auto& imgui_io       = ImGui::GetIO();
  imgui_io.IniFilename = nullptr;
  imgui_io.UserData    = userData;
  imgui_io.DisplaySize = ImVec2(float(width), float(height));
  imgui_io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable keyboard controls (tab, space, arrow keys)
  imgui_io.KeyMap[ImGuiKey_Tab]        = NVPWindow::KEY_TAB;
  imgui_io.KeyMap[ImGuiKey_LeftArrow]  = NVPWindow::KEY_LEFT;
  imgui_io.KeyMap[ImGuiKey_RightArrow] = NVPWindow::KEY_RIGHT;
  imgui_io.KeyMap[ImGuiKey_UpArrow]    = NVPWindow::KEY_UP;
  imgui_io.KeyMap[ImGuiKey_DownArrow]  = NVPWindow::KEY_DOWN;
  imgui_io.KeyMap[ImGuiKey_PageUp]     = NVPWindow::KEY_PAGE_UP;
  imgui_io.KeyMap[ImGuiKey_PageDown]   = NVPWindow::KEY_PAGE_DOWN;
  imgui_io.KeyMap[ImGuiKey_Home]       = NVPWindow::KEY_HOME;
  imgui_io.KeyMap[ImGuiKey_End]        = NVPWindow::KEY_END;
  imgui_io.KeyMap[ImGuiKey_Insert]     = NVPWindow::KEY_INSERT;
  imgui_io.KeyMap[ImGuiKey_Delete]     = NVPWindow::KEY_DELETE;
  imgui_io.KeyMap[ImGuiKey_Backspace]  = NVPWindow::KEY_BACKSPACE;
  imgui_io.KeyMap[ImGuiKey_Space]      = NVPWindow::KEY_SPACE;
  imgui_io.KeyMap[ImGuiKey_Enter]      = NVPWindow::KEY_ENTER;
  imgui_io.KeyMap[ImGuiKey_Escape]     = NVPWindow::KEY_ESCAPE;
  imgui_io.KeyMap[ImGuiKey_A]          = NVPWindow::KEY_A;
  imgui_io.KeyMap[ImGuiKey_C]          = NVPWindow::KEY_C;
  imgui_io.KeyMap[ImGuiKey_V]          = NVPWindow::KEY_V;
  imgui_io.KeyMap[ImGuiKey_X]          = NVPWindow::KEY_X;
  imgui_io.KeyMap[ImGuiKey_Y]          = NVPWindow::KEY_Y;
  imgui_io.KeyMap[ImGuiKey_Z]          = NVPWindow::KEY_Z;

  // Scale style sizes for high-DPI monitors
  ImGuiStyle& imgui_style = ImGui::GetStyle();
  imgui_style.ScaleAllSizes(fontmode == FONT_FIXED ? 1 : getDPIScale());
}

void Combo(const char* label, size_t numEnums, const Enum* enums, void* valuePtr, ImGuiComboFlags flags, ValueType valueType, bool* valueChanged)
{
  int*   ivalue = (int*)valuePtr;
  float* fvalue = (float*)valuePtr;

  size_t idx   = 0;
  bool   found = false;
  for(size_t i = 0; i < numEnums; i++)
  {
    switch(valueType)
    {
      case TYPE_INT:
        if(enums[i].ivalue == *ivalue)
        {
          idx   = i;
          found = true;
        }
        break;
      case TYPE_FLOAT:
        if(enums[i].fvalue == *fvalue)
        {
          idx   = i;
          found = true;
        }
        break;
    }
  }

  if(ImGui::BeginCombo(label, enums[idx].name.c_str(), flags))  // The second parameter is the label previewed before opening the combo.
  {
    for(size_t i = 0; i < numEnums; i++)
    {
      bool is_selected = i == idx;
      if(ImGui::Selectable(enums[i].name.c_str(), is_selected))
      {
        switch(valueType)
        {
          case TYPE_INT:
            *ivalue = enums[i].ivalue;
            break;
          case TYPE_FLOAT:
            *fvalue = enums[i].fvalue;
            break;
        }
        if(valueChanged)
          *valueChanged = true;
      }
      if(is_selected)
      {
        ImGui::SetItemDefaultFocus();  // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
      }
    }
    ImGui::EndCombo();
  }
}

//--------------------------------------------------------------------------------------------------
//
// If GLFW has been initialized, returns the DPI scale of the primary monitor. Otherwise, returns 1.
//
float getDPIScale()
{
  // Cached DPI scale, so that this doesn't change after the first time code calls getDPIScale.
  // A negative value indicates that the value hasn't been computed yet.
  static float cached_dpi_scale = -1.0f;

  if(cached_dpi_scale < 0.0f)
  {
    // Compute the product of the monitor DPI scale and any DPI scale
    // set in the NVPRO_DPI_SCALE variable.
    cached_dpi_scale = 1.0f;

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    if(monitor != nullptr)
    {
      float y_scale;
      glfwGetMonitorContentScale(monitor, &cached_dpi_scale, &y_scale);
    }
    // Otherwise, GLFW isn't initialized yet, but might be in the future.
    // (Note that this code assumes all samples use GLFW.)

    // Multiply by the value of the NVPRO_DPI_SCALE environment variable.
    const char* dpi_env = getenv("NVPRO_DPI_SCALE");
    if(dpi_env)
    {
      const float parsed_dpi_env = strtof(dpi_env, nullptr);
      if(parsed_dpi_env != 0.0f)
      {
        cached_dpi_scale *= parsed_dpi_env;
      }
    }

    cached_dpi_scale = (cached_dpi_scale > 0.0f ? cached_dpi_scale : 1.0f);
  }

  return cached_dpi_scale;
}

//--------------------------------------------------------------------------------------------------
// Setting a dark style for the GUI
//
void setStyle()
{
  ImGui::StyleColorsDark();

  ImGuiStyle& style                  = ImGui::GetStyle();
  style.WindowRounding               = 0.0f;
  style.WindowBorderSize             = 0.0f;
  style.ColorButtonPosition          = ImGuiDir_Left;
  style.FrameRounding                = 2.0f;
  style.FrameBorderSize              = 1.0f;
  style.GrabRounding                 = 4.0f;
  style.IndentSpacing                = 12.0f;
  style.Colors[ImGuiCol_WindowBg]    = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);
  style.Colors[ImGuiCol_MenuBarBg]   = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);
  style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);
  style.Colors[ImGuiCol_PopupBg]     = ImVec4(0.135f, 0.135f, 0.135f, 1.0f);
  style.Colors[ImGuiCol_Border]      = ImVec4(0.4f, 0.4f, 0.4f, 0.5f);
  style.Colors[ImGuiCol_FrameBg]     = ImVec4(0.05f, 0.05f, 0.05f, 0.5f);

  // Normal
  ImVec4                normal_color(0.465f, 0.465f, 0.525f, 1.0f);
  std::vector<ImGuiCol> to_change_nrm;
  to_change_nrm.push_back(ImGuiCol_Header);
  to_change_nrm.push_back(ImGuiCol_SliderGrab);
  to_change_nrm.push_back(ImGuiCol_Button);
  to_change_nrm.push_back(ImGuiCol_CheckMark);
  to_change_nrm.push_back(ImGuiCol_ResizeGrip);
  to_change_nrm.push_back(ImGuiCol_TextSelectedBg);
  to_change_nrm.push_back(ImGuiCol_Separator);
  to_change_nrm.push_back(ImGuiCol_FrameBgActive);
  for(auto c : to_change_nrm)
  {
    style.Colors[c] = normal_color;
  }

  // Active
  ImVec4                active_color(0.365f, 0.365f, 0.425f, 1.0f);
  std::vector<ImGuiCol> to_change_act;
  to_change_act.push_back(ImGuiCol_HeaderActive);
  to_change_act.push_back(ImGuiCol_SliderGrabActive);
  to_change_act.push_back(ImGuiCol_ButtonActive);
  to_change_act.push_back(ImGuiCol_ResizeGripActive);
  to_change_act.push_back(ImGuiCol_SeparatorActive);
  for(auto c : to_change_act)
  {
    style.Colors[c] = active_color;
  }

  // Hovered
  ImVec4                hovered_color(0.565f, 0.565f, 0.625f, 1.0f);
  std::vector<ImGuiCol> to_change_hover;
  to_change_hover.push_back(ImGuiCol_HeaderHovered);
  to_change_hover.push_back(ImGuiCol_ButtonHovered);
  to_change_hover.push_back(ImGuiCol_FrameBgHovered);
  to_change_hover.push_back(ImGuiCol_ResizeGripHovered);
  to_change_hover.push_back(ImGuiCol_SeparatorHovered);
  for(auto c : to_change_hover)
  {
    style.Colors[c] = hovered_color;
  }


  style.Colors[ImGuiCol_TitleBgActive]    = ImVec4(0.465f, 0.465f, 0.465f, 1.0f);
  style.Colors[ImGuiCol_TitleBg]          = ImVec4(0.125f, 0.125f, 0.125f, 1.0f);
  style.Colors[ImGuiCol_Tab]              = ImVec4(0.05f, 0.05f, 0.05f, 0.5f);
  style.Colors[ImGuiCol_TabHovered]       = ImVec4(0.465f, 0.495f, 0.525f, 1.0f);
  style.Colors[ImGuiCol_TabActive]        = ImVec4(0.282f, 0.290f, 0.302f, 1.0f);
  style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.465f, 0.465f, 0.465f, 0.350f);

  //Colors_ext[ImGuiColExt_Warning] = ImVec4 (1.0f, 0.43f, 0.35f, 1.0f);

  ImGui::SetColorEditOptions(ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel);
}

//
// Local, return true if the filename exist
//
static bool fileExists(const char* filename)
{
  std::ifstream stream;
  stream.open(filename);
  return stream.is_open();
}

//--------------------------------------------------------------------------------------------------
// Looking for TTF fonts, first on the VULKAN SDK, then Windows default fonts
//
void setFonts(FontMode fontmode)
{
  ImGuiIO&    io             = ImGui::GetIO();
  const float high_dpi_scale = getDPIScale();


  // Nicer fonts
  ImFont* font = nullptr;
  if(fontmode == FONT_MONOSPACED_SCALED)
  {
    if(font == nullptr)
    {
      const std::string p = R"(C:/Windows/Fonts/consola.ttf)";
      if(fileExists(p.c_str()))
        font = io.Fonts->AddFontFromFileTTF(p.c_str(), 12.0f * high_dpi_scale);
    }
  }
  else if(fontmode == FONT_PROPORTIONAL_SCALED)
  {
    const char* vk_path = getenv("VK_SDK_PATH");
    if(vk_path)
    {
      const std::string p = std::string(vk_path) + R"(/Samples/Layer-Samples/data/FreeSans.ttf)";
      if(fileExists(p.c_str()))
        font = io.Fonts->AddFontFromFileTTF(p.c_str(), 16.0f * high_dpi_scale);
    }
    if(font == nullptr)
    {
      const std::string p = R"(C:/Windows/Fonts/segoeui.ttf)";
      if(fileExists(p.c_str()))
        font = io.Fonts->AddFontFromFileTTF(p.c_str(), 16.0f * high_dpi_scale);
    }
  }

  if(font == nullptr)
  {
    ImFontConfig font_config = ImFontConfig();
    font_config.SizePixels   = 13.0f * ((fontmode == FONT_FIXED) ? 1 : high_dpi_scale);  // 13 is the default font size
    io.Fonts->AddFontDefault(&font_config);
  }
}

// ------------------------------------------------------------------------------------------------
template <>
void Control::show_tooltip(const char* description)
{
  if(!description || strlen(description) == 0)
    return;

  ImGui::BeginTooltip();
  ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
  ImGui::TextUnformatted(description);
  ImGui::PopTextWrapPos();
  ImGui::EndTooltip();
}

// ------------------------------------------------------------------------------------------------
template <>
void Control::show_tooltip(const std::string& description)
{
  if(description.empty())
    return;
  show_tooltip<const char*>(description.c_str());
}

// ------------------------------------------------------------------------------------------------

namespace {

template <typename TScalar, ImGuiDataType type, uint8_t dim>
bool show_slider_control_scalar(TScalar* value, TScalar* min, TScalar* max, const char* format)
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


}  // namespace

template <>
bool Control::show_slider_control<float>(float* value, float& min, float& max, const char* format)
{
  return show_slider_control_scalar<float, ImGuiDataType_Float, 1>(value, &min, &max, format ? format : "%.3f");
}

template <>
bool Control::show_slider_control<nvmath::vec2f>(nvmath::vec2f* value, nvmath::vec2f& min, nvmath::vec2f& max, const char* format)
{
  return show_slider_control_scalar<float, ImGuiDataType_Float, 2>(&value->x, &min.x, &max.x, format ? format : "%.3f");
}

template <>
bool Control::show_slider_control<nvmath::vec3f>(nvmath::vec3f* value, nvmath::vec3f& min, nvmath::vec3f& max, const char* format)
{
  return show_slider_control_scalar<float, ImGuiDataType_Float, 3>(&value->x, &min.x, &max.x, format ? format : "%.3f");
}

template <>
bool Control::show_slider_control<nvmath::vec4f>(nvmath::vec4f* value, nvmath::vec4f& min, nvmath::vec4f& max, const char* format)
{
  return show_slider_control_scalar<float, ImGuiDataType_Float, 4>(&value->x, &min.x, &max.x, format ? format : "%.3f");
}

template <>
bool Control::show_drag_control<float>(float* value, float speed, float& min, float& max, const char* format)
{
  return show_drag_control_scalar<float, ImGuiDataType_Float, 1>(value, speed, &min, &max, format ? format : "%.3f");
}

template <>
bool Control::show_drag_control<nvmath::vec2f>(nvmath::vec2f* value, float speed, nvmath::vec2f& min, nvmath::vec2f& max, const char* format)
{
  return show_drag_control_scalar<float, ImGuiDataType_Float, 2>(&value->x, speed, &min.x, &max.x, format ? format : "%.3f");
}

template <>
bool Control::show_drag_control<nvmath::vec3>(nvmath::vec3* value, float speed, nvmath::vec3& min, nvmath::vec3& max, const char* format)
{
  return show_drag_control_scalar<float, ImGuiDataType_Float, 3>(&value->x, speed, &min.x, &max.x, format ? format : "%.3f");
}

template <>
bool Control::show_drag_control<nvmath::vec4f>(nvmath::vec4f* value, float speed, nvmath::vec4f& min, nvmath::vec4f& max, const char* format)
{
  return show_drag_control_scalar<float, ImGuiDataType_Float, 4>(&value->x, speed, &min.x, &max.x, format ? format : "%.3f");
}


template <>
bool Control::show_slider_control<int>(int* value, int& min, int& max, const char* format)
{
  return show_slider_control_scalar<int, ImGuiDataType_S32, 1>(value, &min, &max, format ? format : "%d");
}

template <>
bool Control::show_slider_control<nvmath::vec2i>(nvmath::vec2i* value, nvmath::vec2i& min, nvmath::vec2i& max, const char* format)
{
  return show_slider_control_scalar<int, ImGuiDataType_S32, 2>(&value->x, &min.x, &max.x, format ? format : "%d");
}

template <>
bool Control::show_slider_control<nvmath::vec3i>(nvmath::vec3i* value, nvmath::vec3i& min, nvmath::vec3i& max, const char* format)
{
  return show_slider_control_scalar<int, ImGuiDataType_S32, 3>(&value->x, &min.x, &max.x, format ? format : "%d");
}

template <>
bool Control::show_slider_control<nvmath::vec4i>(nvmath::vec4i* value, nvmath::vec4i& min, nvmath::vec4i& max, const char* format)
{
  return show_slider_control_scalar<int, ImGuiDataType_S32, 4>(&value->x, &min.x, &max.x, format ? format : "%d");
}

template <>
bool Control::show_drag_control<int>(int* value, float speed, int& min, int& max, const char* format)
{
  return show_drag_control_scalar<int, ImGuiDataType_S32, 1>(value, speed, &min, &max, format ? format : "%d");
}

template <>
bool Control::show_drag_control<nvmath::vec2i>(nvmath::vec2i* value, float speed, nvmath::vec2i& min, nvmath::vec2i& max, const char* format)
{
  return show_drag_control_scalar<int, ImGuiDataType_S32, 2>(&value->x, speed, &min.x, &max.x, format ? format : "%d");
}

template <>
bool Control::show_drag_control<nvmath::vec3i>(nvmath::vec3i* value, float speed, nvmath::vec3i& min, nvmath::vec3i& max, const char* format)
{
  return show_drag_control_scalar<int, ImGuiDataType_S32, 3>(&value->x, speed, &min.x, &max.x, format ? format : "%d");
}

template <>
bool Control::show_drag_control<nvmath::vec4i>(nvmath::vec4i* value, float speed, nvmath::vec4i& min, nvmath::vec4i& max, const char* format)
{
  return show_drag_control_scalar<int, ImGuiDataType_S32, 4>(&value->x, speed, &min.x, &max.x, format ? format : "%d");
}


template <>
bool Control::show_slider_control<uint32_t>(uint32_t* value, uint32_t& min, uint32_t& max, const char* format)
{
  return show_slider_control_scalar<uint32_t, ImGuiDataType_U32, 1>(value, &min, &max, format ? format : "%d");
}
//
//template <>
//bool Control::show_slider_control<uint32_t_2>(uint32_t_2* value, uint32_t_2& min, uint32_t_2& max, const char* format)
//{
//  return show_slider_control_scalar<uint32_t, ImGuiDataType_U32, 2>(&value->x, &min.x, &max.x, format ? format : "%d");
//}
//
//template <>
//bool Control::show_slider_control<uint32_t_3>(uint32_t_3* value, uint32_t_3& min, uint32_t_3& max, const char* format)
//{
//  return show_slider_control_scalar<uint32_t, ImGuiDataType_U32, 3>(&value->x, &min.x, &max.x, format ? format : "%d");
//}
//
//template <>
//bool Control::show_slider_control<uint32_t_4>(uint32_t_4* value, uint32_t_4& min, uint32_t_4& max, const char* format)
//{
//  return show_slider_control_scalar<uint32_t, ImGuiDataType_U32, 4>(&value->x, &min.x, &max.x, format ? format : "%d");
//}
//
//template <>
//bool Control::show_drag_control<uint32_t>(uint32_t* value, float speed, uint32_t& min, uint32_t& max, const char* format)
//{
//  return show_drag_control_scalar<uint32_t, ImGuiDataType_U32, 1>(value, speed, &min, &max, format ? format : "%d");
//}
//
//template <>
//bool Control::show_drag_control<uint32_t_2>(uint32_t_2* value, float speed, uint32_t_2& min, uint32_t_2& max, const char* format)
//{
//  return show_drag_control_scalar<uint32_t, ImGuiDataType_U32, 2>(&value->x, speed, &min.x, &max.x, format ? format : "%d");
//}
//
//template <>
//bool Control::show_drag_control<uint32_t_3>(uint32_t_3* value, float speed, uint32_t_3& min, uint32_t_3& max, const char* format)
//{
//  return show_drag_control_scalar<uint32_t, ImGuiDataType_U32, 3>(&value->x, speed, &min.x, &max.x, format ? format : "%d");
//}
//
//template <>
//bool Control::show_drag_control<uint32_t_4>(uint32_t_4* value, float speed, uint32_t_4& min, uint32_t_4& max, const char* format)
//{
//  return show_drag_control_scalar<uint32_t, ImGuiDataType_U32, 4>(&value->x, speed, &min.x, &max.x, format ? format : "%d");
//}


template <>
bool Control::show_slider_control<size_t>(size_t* value, size_t& min, size_t& max, const char* format)
{
  return show_slider_control_scalar<size_t, ImGuiDataType_U64, 1>(value, &min, &max, format ? format : "%d");
}

template <>
bool Control::show_drag_control<size_t>(size_t* value, float speed, size_t& min, size_t& max, const char* format)
{
  return show_drag_control_scalar<size_t, ImGuiDataType_U64, 1>(value, speed, &min, &max, format ? format : "%d");
}

// Static member declaration
Panel::Style   Panel::style{};
Control::Style Control::style{};

}  // namespace ImGuiH

void ImGui::PlotMultiEx(const char* label, int num_datas, ImPlotMulti* datas, const char* overlay_text, ImVec2 frame_size)
{
  ImGuiWindow* window = GetCurrentWindow();
  if(window->SkipItems)
    return;

  ImGuiContext&     g     = *GImGui;
  const ImGuiStyle& style = g.Style;
  const ImGuiID     id    = window->GetID(label);

  const ImVec2 label_size = CalcTextSize(label, nullptr, true);
  if(frame_size.x == 0.0f)
    frame_size.x = CalcItemWidth();
  if(frame_size.y == 0.0f)
    frame_size.y = label_size.y + (style.FramePadding.y * 2);

  const ImRect frame_bb(window->DC.CursorPos, window->DC.CursorPos + frame_size);
  const ImRect inner_bb(frame_bb.Min + style.FramePadding, frame_bb.Max - style.FramePadding);
  const ImRect total_bb(frame_bb.Min,
                        frame_bb.Max + ImVec2(label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f, 0));
  ItemSize(total_bb, style.FramePadding.y);
  if(!ItemAdd(total_bb, 0, &frame_bb))
    return;
  const bool hovered = ItemHoverable(frame_bb, id);

  // Determine scale from values if not specified
  for(int data_idx = 0; data_idx < num_datas; data_idx++)
  {
    auto& cur_data = datas[data_idx];
    if(cur_data.scale_min == FLT_MAX || cur_data.scale_max == FLT_MAX)
    {
      float v_min = FLT_MAX;
      float v_max = -FLT_MAX;

      for(int i = 0; i < cur_data.values_count; i++)
      {
        const float v = cur_data.data[i];
        v_min         = ImMin(v_min, v);
        v_max         = ImMax(v_max, v);
      }

      if(cur_data.scale_min == FLT_MAX)
        cur_data.scale_min = v_min;
      if(cur_data.scale_max == FLT_MAX)
        cur_data.scale_max = v_max;
    }
  }

  RenderFrame(frame_bb.Min, frame_bb.Max, GetColorU32(ImGuiCol_FrameBg), true, style.FrameRounding);


  if(hovered && inner_bb.Contains(g.IO.MousePos))
  {
    ImGui::BeginTooltip();
    for(int data_idx = 0; data_idx < num_datas; data_idx++)
    {
      auto& cur_data = datas[data_idx];

      bool type_line = (cur_data.plot_type == ImGuiPlotType_Lines) || (cur_data.plot_type == ImGuiPlotType_Area);

      int res_w      = ImMin((int)frame_size.x, cur_data.values_count) + (type_line ? -1 : 0);
      int item_count = cur_data.values_count + (type_line ? -1 : 0);

      // Tooltip on hover
      const float t = ImClamp((g.IO.MousePos.x - inner_bb.Min.x) / (inner_bb.Max.x - inner_bb.Min.x), 0.0f, 0.9999f);
      const int   v_idx = (int)(t * item_count);
      IM_ASSERT(v_idx >= 0 && v_idx < cur_data.values_count);

      const float v0 = cur_data.data[(v_idx + cur_data.values_offset) % cur_data.values_count];
      TextColored(cur_data.color, "%8.4g | %s", v0, cur_data.name);
    }
    ImGui::EndTooltip();

    ImVec2 pos0 = ImVec2(g.IO.MousePos.x, inner_bb.Max.y);
    ImVec2 pos1 = ImVec2(g.IO.MousePos.x, inner_bb.Min.y);

    window->DrawList->AddLine(pos0, pos1, GetColorU32(ImGuiCol_PlotLinesHovered));
  }

  for(int data_idx = 0; data_idx < num_datas; data_idx++)
  {
    auto& cur_data  = datas[data_idx];
    bool  type_line = (cur_data.plot_type == ImGuiPlotType_Lines) || (cur_data.plot_type == ImGuiPlotType_Area);

    int res_w      = ImMin((int)frame_size.x, cur_data.values_count) + (type_line ? -1 : 0);
    int item_count = cur_data.values_count + (type_line ? -1 : 0);

    const float t_step = 1.0f / (float)res_w;
    const float inv_scale =
        (cur_data.scale_min == cur_data.scale_max) ? 0.0f : (1.0f / (cur_data.scale_max - cur_data.scale_min));

    float  v0                    = cur_data.data[(0 + cur_data.values_offset) % cur_data.values_count];
    float  t0                    = 0.0f;
    ImVec2 tp0                   = ImVec2(t0, 1.0f - ImSaturate((v0 - cur_data.scale_min) * inv_scale));  // Point in the normalized space of our target rectangle
    float  histogram_zero_line_t = (cur_data.scale_min * cur_data.scale_max < 0.0f) ?
                                      (-cur_data.scale_min * inv_scale) :
                                      (cur_data.scale_min < 0.0f ? 0.0f : 1.0f);  // Where does the zero line stands

    const ImU32 col_base   = ColorConvertFloat4ToU32(cur_data.color);
    const ImU32 col_base_a = ColorConvertFloat4ToU32(
        ImColor(cur_data.color.Value.x, cur_data.color.Value.y, cur_data.color.Value.z, cur_data.color.Value.w = 0.5));

    for(int n = 0; n < res_w; n++)
    {
      const float t1     = t0 + t_step;
      const int   v1_idx = (int)(t0 * item_count + 0.5f);
      IM_ASSERT(v1_idx >= 0 && v1_idx < cur_data.values_count);
      const float  v1  = cur_data.data[(v1_idx + cur_data.values_offset + 1) % cur_data.values_count];
      const ImVec2 tp1 = ImVec2(t1, 1.0f - ImSaturate((v1 - cur_data.scale_min) * inv_scale));

      // NB: Draw calls are merged together by the DrawList system. Still, we should render our batch are lower level to save a bit of CPU.
      ImVec2 pos0 = ImLerp(inner_bb.Min, inner_bb.Max, tp0);
      ImVec2 pos1 = ImLerp(inner_bb.Min, inner_bb.Max, type_line ? tp1 : ImVec2(tp1.x, histogram_zero_line_t));
      if(cur_data.plot_type == ImGuiPlotType_Lines)
      {
        window->DrawList->AddLine(pos0, pos1, col_base, cur_data.thickness);
      }
      else if(cur_data.plot_type == ImGuiPlotType_Area)
      {
        ImDrawListFlags backup_flags = window->DrawList->Flags;
        window->DrawList->Flags &= ~ImDrawListFlags_AntiAliasedFill;  // Disable AA on Quad to look nice when next to each other.
        window->DrawList->AddQuadFilled(pos0, pos1, ImVec2(pos1.x, inner_bb.Max.y), ImVec2(pos0.x, inner_bb.Max.y), col_base_a);
        window->DrawList->Flags = backup_flags;
        window->DrawList->AddLine(pos0, pos1, col_base, cur_data.thickness);
      }
      else if(cur_data.plot_type == ImGuiPlotType_Histogram)
      {
        if(pos1.x >= pos0.x + 2.0f)
          pos1.x -= 1.0f;
        window->DrawList->AddRectFilled(pos0, pos1, col_base, cur_data.thickness - 1, ImDrawCornerFlags_Top);
      }

      t0  = t1;
      tp0 = tp1;
    }
  }

  // Text overlay
  if(overlay_text)
    RenderTextClipped(ImVec2(frame_bb.Min.x, frame_bb.Min.y + style.FramePadding.y), frame_bb.Max, overlay_text, NULL,
                      NULL, ImVec2(0.5f, 0.0f));

  if(label_size.x > 0.0f)
    RenderText(ImVec2(frame_bb.Max.x + style.ItemInnerSpacing.x, inner_bb.Min.y), label);
}
