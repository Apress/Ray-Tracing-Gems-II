/*
* Copyright (c) 2014-2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "imgui_camera_widget.h"
#include "fileformats/json.hpp"
#include "imgui.h"
#include "imgui_helper.h"
#include "nvh/cameramanipulator.hpp"
#include <fstream>
#include <sstream>


namespace ImGuiH {

using nlohmann::json;
using Gui = ImGuiH::Control;

bool IsItemActiveLastFrame()
{
  ImGuiContext& g = *GImGui;
  if(g.ActiveIdPreviousFrame)
    return g.ActiveIdPreviousFrame == g.CurrentWindow->DC.LastItemId;
  return false;
}

bool IsItemJustReleased()
{
  return IsItemActiveLastFrame() && !ImGui::IsItemActive();
}


//--------------------------------------------------------------------------------------------------
// Holds all saved cameras in a vector of Cameras
// - The first camera in the list is the HOME camera, the one that was set before this is called.
// - The update function will check if something has changed and will save the JSON to disk, only
//  once in a while.
// - Adding a camera will be added only if it is different from all other saved cameras
// - load/save Setting will load next to the executable, the "jsonFilename" + ".json"
//
struct CameraManager
{
  // update setting, load or save
  void update(nvh::CameraManipulator& cameraM)
  {
    // Push the HOME camera and load default setting
    if(cameras.empty())
    {
      cameras.emplace_back(cameraM.getCamera());
    }
    if(doLoadSetting)
      loadSetting(cameraM);

    // Save settings (with a delay after the last modification, so we don't spam disk too much)
    auto& IO = ImGui::GetIO();
    if(settingsDirtyTimer > 0.0f)
    {
      settingsDirtyTimer -= IO.DeltaTime;
      if(settingsDirtyTimer <= 0.0f)
      {
        saveSetting(cameraM);
        settingsDirtyTimer = 0.0f;
      }
    }
  }

  // Clear all cameras except the HOME
  void removedSavedCameras()
  {
    if(cameras.size() > 1)
      cameras.erase(cameras.begin() + 1, cameras.end());
  }

  void setCameraJsonFile(const std::string& filename)
  {
    jsonFilename  = filename + ".json";
    doLoadSetting = true;
    removedSavedCameras();
  }


  void setHomeCamera(const nvh::CameraManipulator::Camera& camera)
  {
    if(cameras.empty())
      cameras.resize(1);
    cameras[0] = camera;
  }

  // Adding a camera only if it different from all the saved ones
  void addCamera(const nvh::CameraManipulator::Camera& camera)
  {
    bool unique = true;
    for(const auto& c : cameras)
    {
      if(c == camera)
      {
        unique = false;
        break;
      }
    }
    if(unique)
    {
      cameras.emplace_back(camera);
      markIniSettingsDirty();
    }
  }

  void markIniSettingsDirty()
  {
    auto& IO = ImGui::GetIO();
    if(settingsDirtyTimer <= 0.0f)
      settingsDirtyTimer = IO.IniSavingRate / 2.0f;
  }

  template <typename T>
  bool getJsonValue(const json& j, const std::string& name, T& value)
  {
    auto fieldIt = j.find(name);
    if(fieldIt != j.end())
    {
      value = (*fieldIt);
      return true;
    }
    LOGE("Could not find JSON field %s", name.c_str());
    return false;
  }

  template <typename T>
  bool getJsonArray(const json& j, const std::string& name, T& value)
  {
    auto fieldIt = j.find(name);
    if(fieldIt != j.end())
    {
      value = T((*fieldIt).begin(), (*fieldIt).end());
      return true;
    }
    LOGE("Could not find JSON field %s", name.c_str());
    return false;
  }


  void loadSetting(nvh::CameraManipulator& cameraM)
  {
    if(jsonFilename.empty() || cameras.empty())
      return;

    try
    {
      std::ifstream i(NVPSystem::exePath() + jsonFilename);
      if(!i.is_open())
        return;

      // Parsing the file
      json j;
      i >> j;

      // Clear all cameras except the HOME
      removedSavedCameras();

      // Temp
      int                iVal;
      float              fVal;
      std::vector<float> vfVal;

      // Settings
      if(getJsonValue(j, "mode", iVal))
        cameraM.setMode(static_cast<nvh::CameraManipulator::Modes>(iVal));
      if(getJsonValue(j, "speed", fVal))
        cameraM.setSpeed(fVal);
      if(getJsonValue(j, "anim_duration", fVal))
        cameraM.setAnimationDuration(fVal);

      // All cameras
      std::vector<json> cc;
      getJsonArray(j, "cameras", cc);
      for(auto& c : cc)
      {
        nvh::CameraManipulator::Camera camera;
        if(getJsonArray(c, "eye", vfVal))
          camera.eye = {vfVal[0], vfVal[1], vfVal[2]};
        if(getJsonArray(c, "ctr", vfVal))
          camera.ctr = {vfVal[0], vfVal[1], vfVal[2]};
        if(getJsonArray(c, "up", vfVal))
          camera.up = {vfVal[0], vfVal[1], vfVal[2]};
        if(getJsonValue(c, "fov", fVal))
          camera.fov = fVal;
        cameras.emplace_back(camera);
      }

      doLoadSetting = false;
    }
    catch(...)
    {
      return;
    }
  }

  void saveSetting(nvh::CameraManipulator& cameraM)
  {
    if(jsonFilename.empty())
      return;

    json j;
    j["mode"]          = cameraM.getMode();
    j["speed"]         = cameraM.getSpeed();
    j["anim_duration"] = cameraM.getAnimationDuration();

    // Save all extra cameras
    json cc = json::array();
    for(size_t n = 1; n < cameras.size(); n++)
    {
      auto& c   = cameras[n];
      json  jo  = json::object();
      jo["eye"] = std::vector<float>{c.eye.x, c.eye.y, c.eye.z};
      jo["up"]  = std::vector<float>{c.up.x, c.up.y, c.up.z};
      jo["ctr"] = std::vector<float>{c.ctr.x, c.ctr.y, c.ctr.z};
      jo["fov"] = c.fov;
      cc.push_back(jo);
    }
    j["cameras"] = cc;

    std::ofstream o(NVPSystem::exePath() + jsonFilename);
    o << j.dump(2) << std::endl;
    o.close();
  }

  // Holds all cameras. [0] == HOME
  std::vector<nvh::CameraManipulator::Camera> cameras;
  float                                       settingsDirtyTimer{0};
  std::string                                 jsonFilename;
  bool                                        doLoadSetting{true};

  ~CameraManager()
  { /*saveSetting();*/
  }
};
static CameraManager sCamMgr;

// Helper to display a tooltip when hovered.
static void HoverText(const std::string& desc)
{
  if(ImGui::IsItemHovered())
  {
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
    ImGui::TextUnformatted(desc.c_str());
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}

//--------------------------------------------------------------------------------------------------
// Display the values of the current camera: position, center, up and FOV
//
void CurrentCameraTab(nvh::CameraManipulator& cameraM, nvh::CameraManipulator::Camera& camera, bool& changed, bool& instantSet)
{

  bool           y_is_up = camera.up.y == 1;
  Control::Flags flag    = Control::Flags::Normal;

  // Using IsItemDeactivatedAfterEdit to avoid the value to change while typing
  Gui::Custom("Eye", "Position of the Camera", [&] { return ImGui::InputFloat3("##Eye", &camera.eye.x); });
  changed |= ImGui::IsItemDeactivatedAfterEdit();

  Gui::Custom("Center", "Center of camera interest", [&] { return ImGui::InputFloat3("##Ctr", &camera.ctr.x); });
  changed |= ImGui::IsItemDeactivatedAfterEdit();

  changed |= Gui::Checkbox("Y is UP", "Is Y pointing up or Z?", &y_is_up);
  if(Gui::Drag("FOV", "Field of view in degrees", &camera.fov, &sCamMgr.cameras[0].fov, flag, 1.0f, 179.0f, 0.1f,
               "%.2f deg"))
  {
    // Need to instantly set the camera, otherwise the transition to the new camera will have
    // a different value as the one currently set the next time it comes here and will be impossible
    // to set the value.
    instantSet = true;
    changed    = true;
  }
  camera.up = y_is_up ? nvmath::vec3f(0, 1, 0) : nvmath::vec3f(0, 0, 1);

  if(cameraM.isAnimated())
  {
    // Ignoring any changes while the camera is moving to the goal. The camera has to be in the
    // new position before setting a new value.
    changed = false;
  }

  ImGui::TextDisabled("(?)");
  HoverText(cameraM.getHelp());
  ImGui::SameLine();
  if(ImGui::SmallButton("Copy"))
  {
    char text[128];
    sprintf(text, "{%.3f, %.3f, %.3f}, {%.3f, %.3f, %.3f}, {%.3f, %.3f, %.3f}",  //
            camera.eye.x, camera.eye.y, camera.eye.z,                            //
            camera.ctr.x, camera.ctr.y, camera.ctr.z,                            //
            camera.up.x, camera.up.y, camera.up.z);
    ImGui::SetClipboardText(text);
  }
  HoverText("Copy to the clipboard the current camera: {eye}, {ctr}, {up}");
}

//--------------------------------------------------------------------------------------------------
// Display buttons for all saved cameras. Allow to create and delete saved cameras
//
void SavedCameraTab(nvh::CameraManipulator& cameraM, nvh::CameraManipulator::Camera& camera, bool& changed)
{
  // Dummy
  ImVec2      button_sz(50, 30);
  char        label[128];
  ImGuiStyle& style             = ImGui::GetStyle();
  int         buttons_count     = (int)sCamMgr.cameras.size();
  float       window_visible_x2 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;

  // The HOME camera button, different from the other ones
  if(ImGui::Button("Home", ImVec2(ImGui::GetWindowContentRegionMax().x, 50)))
  {
    camera  = sCamMgr.cameras[0];
    changed = true;
  }
  HoverText("Reset the camera to its origin");

  // Display all the saved camera in an array of buttons
  int delete_item = -1;
  for(int n = 1; n < buttons_count; n++)
  {
    ImGui::PushID(n);
    sprintf(label, "# %d", n);
    if(ImGui::Button(label, button_sz))
    {
      camera  = sCamMgr.cameras[n];
      changed = true;
    }

    // Middle click to delete a camera
    if(ImGui::IsItemHovered() && ImGui::GetIO().MouseClicked[GLFW_MOUSE_BUTTON_MIDDLE])
      delete_item = n;

    // Displaying the position of the camera when hovering the button
    sprintf(label, "Pos: %3.2f, %3.2f, %3.2f", sCamMgr.cameras[n].eye.x, sCamMgr.cameras[n].eye.y,
            sCamMgr.cameras[n].eye.z);
    HoverText(label);

    // Wrapping all buttons (see ImGUI Demo)
    float last_button_x2 = ImGui::GetItemRectMax().x;
    float next_button_x2 = last_button_x2 + style.ItemSpacing.x + button_sz.x;  // Expected position if next button was on same line
    if(n + 1 < buttons_count && next_button_x2 < window_visible_x2)
      ImGui::SameLine();

    ImGui::PopID();
  }

  // Adding a camera button
  if(ImGui::Button("+"))
  {
    sCamMgr.addCamera(cameraM.getCamera());
    sCamMgr.markIniSettingsDirty();
  }
  HoverText("Add a new saved camera");
  ImGui::SameLine();
  ImGui::TextDisabled("(?)");
  HoverText("Middle-click a camera to delete it");

  // Remove element
  if(delete_item > 0)
    sCamMgr.cameras.erase(sCamMgr.cameras.begin() + delete_item);
}

//--------------------------------------------------------------------------------------------------
// This holds all camera setting, like the speed, the movement mode, transition duration
//
void CameraExtraTab(nvh::CameraManipulator& cameraM, bool& changed)
{
  // Navigation Mode
  auto mode = cameraM.getMode();
  changed |= Gui::Custom("Navigation", "Camera Navigation Mode", [&] {
    int   rmode  = static_cast<int>(mode);
    float indent = ImGui::GetCursorPos().x;
    changed |= ImGui::RadioButton("Examine", &rmode, nvh::CameraManipulator::Examine);
    HoverText("The camera orbit around a point of interest");
    ImGui::NewLine();
    ImGui::SameLine(indent);
    changed |= ImGui::RadioButton("Fly", &rmode, nvh::CameraManipulator::Fly);
    HoverText("The camera is free and move toward the looking direction");
    ImGui::NewLine();
    ImGui::SameLine(indent);
    changed |= ImGui::RadioButton("Walk", &rmode, nvh::CameraManipulator::Walk);
    HoverText("The camera is free but stay on a plane");
    cameraM.setMode(static_cast<nvh::CameraManipulator::Modes>(rmode));
    return changed;
  });


  // Speed
  auto  speed = cameraM.getSpeed();
  float def_speed{3.0f};
  ImGuiH::Control::Slider("Speed", "Changing the default speed movement", &speed, &def_speed, Control::Flags::Normal, 0.01f, 10.f);
  cameraM.setSpeed(speed);

  // Animation
  float duration = (float)cameraM.getAnimationDuration();
  float def_duration{0.5f};
  ImGuiH::Control::Slider("Transition", "Nb seconds to move to new position", &duration, &def_duration,
                          Control::Flags::Normal, 0.0f, 2.f);
  cameraM.setAnimationDuration(duration);
}

//--------------------------------------------------------------------------------------------------
// Display the camera eye and center of interest position of the camera (nvh::CameraManipulator)
// Allow also to modify the field-of-view (FOV)
// And basic control information is displayed
bool CameraWidget(nvh::CameraManipulator& cameraM /*= nvh::CameraManipulator::Singleton()*/)
{

  bool changed{false};
  bool instantSet{false};
  auto camera = cameraM.getCamera();

  // Updating the camera manager
  sCamMgr.update(cameraM);

  // Starting UI
  if(ImGui::BeginTabBar("Hello"))
  {
    if(ImGui::BeginTabItem("Current"))
    {
      CurrentCameraTab(cameraM, camera, changed, instantSet);
      ImGui::EndTabItem();
    }

    if(ImGui::BeginTabItem("Cameras"))
    {
      SavedCameraTab(cameraM, camera, changed);
      ImGui::EndTabItem();
    }

    if(ImGui::BeginTabItem("Extra"))
    {
      CameraExtraTab(cameraM, changed);
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }

  // Apply the change back to the camera
  if(changed)
  {
    cameraM.setCamera(camera, instantSet);
    sCamMgr.markIniSettingsDirty();
  }

  // This makes the camera to transition smoothly to the new position
  cameraM.updateAnim();

  return changed;
}


void SetCameraJsonFile(const std::string& filename)
{
  sCamMgr.jsonFilename  = filename + ".json";
  sCamMgr.doLoadSetting = true;
}

void SetHomeCamera(const nvh::CameraManipulator::Camera& camera)
{
  sCamMgr.setHomeCamera(camera);
}

void AddCamera(const nvh::CameraManipulator::Camera& camera)
{
  sCamMgr.addCamera(camera);
}

}  // namespace ImGuiH
