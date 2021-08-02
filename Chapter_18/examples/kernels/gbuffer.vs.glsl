#version 300 es

// an attribute is an input (in) to a vertex shader.
// It will receive data from a buffer
layout(location = 0) in vec3 in_Position;
layout(location = 1) in vec3 in_Normal;
layout(location = 2) in vec2 in_UV;

uniform mat4 u_ProjectionMatrix;
uniform mat4 u_ViewMatrix;
uniform mat4 u_ModelMatrix;
uniform mat3 u_NormalMatrix;

out vec2 v_UV;
out vec3 v_Normal;
out vec3 v_PosWCS;

// all shaders have a main function
void main() {
  vec3 position = in_Position;

  gl_Position = u_ProjectionMatrix * u_ViewMatrix * vec4(position, 1);
  v_UV = in_UV;
  v_Normal = in_Normal;
  v_PosWCS = position;
}