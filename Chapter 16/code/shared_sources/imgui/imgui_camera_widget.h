/*
* Copyright (c) 2014-2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once
#include "nvh/cameramanipulator.hpp"
#include "nvmath/nvmath.h"

namespace ImGuiH {

bool CameraWidget(
    nvh::CameraManipulator& cameraM =
        nvh::CameraManipulator::Singleton());  // Camera widget for the the Camera Manipulator

void SetCameraJsonFile(
    const std::string&
        filename);  // set the name (without .json) of the setting file. It will load and replace all camera and settings
void SetHomeCamera(
    const nvh::CameraManipulator::Camera& camera);  // set the home camera - replace the one on load
void AddCamera(
    const nvh::CameraManipulator::Camera& camera);  // adding a camera to the list of cameras


}  // namespace ImGuiH
