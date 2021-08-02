#pragma once

struct ImDrawData;

namespace ImGui
{
  void InitGL();
  void ShutdownGL();

  void RenderDrawDataGL(const ImDrawData* drawData);
}
