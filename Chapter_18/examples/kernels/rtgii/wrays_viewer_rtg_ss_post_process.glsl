#version 300 es
precision highp float;
precision highp int;

layout(location = 0) out vec4 pixel_color_OUT;

uniform sampler2D accumulated_texture;

float A = 0.15;
float B = 0.5;
float C = 0.10;
float D = 0.20;
float E = 0.02;
float F = 0.30;
float W = 11.2;
vec3 Uncharted2Tonemap(vec3 x) 
{
	return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

vec3 filmic(vec3 color)
{
	float exposureBias = 2.0;
	vec3 curr = Uncharted2Tonemap(exposureBias * color);
	vec3 whiteScale = 1.0 / Uncharted2Tonemap(vec3(W));
	return pow(curr * whiteScale, vec3(1.0 / 2.2));
}

vec3 ACESFilm(vec3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

void main() {
  vec4 pixel_color = texelFetch(accumulated_texture, ivec2(gl_FragCoord.xy), 0);
  
  float bloom_thres = 1.0;
  vec3 leak = vec3(0.0);
  float sum_w = 0.0;
  
  for (int i=-5;i<5;i++)
  {
	for (int j=-5;j<5;j++)
	{
		vec3 leak_sample = texelFetch(accumulated_texture, ivec2(gl_FragCoord.xy)+ivec2(i,j), 0).rgb;
		float weight = 1.0/float(1+i*i+j*j);
		leak += max(vec3(0.0),leak_sample - vec3(bloom_thres)) * weight;
		sum_w += weight;
	}
  }
  pixel_color.rgb += leak / sum_w;
  
  pixel_color_OUT.rgb = pow(ACESFilm(pixel_color.rgb), vec3(1.0 / 1.7));
}
