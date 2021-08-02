#version 300 es
precision highp sampler2D;
precision highp float;
precision highp int;

layout(location = 0) out vec4 accumulation_OUT;

uniform float blend_factor;
uniform sampler2D accumulation_texture;

void main() {
  vec3 current_color = texelFetch(accumulation_texture, ivec2(gl_FragCoord.xy), 0).rgb;

  current_color = clamp(current_color, vec3(0.0),vec3(30.0));
  
  accumulation_OUT = vec4(current_color, blend_factor);
}
