#version 300 es

precision highp float;
precision highp sampler2D;

uniform sampler2D u_inputBuffer;
uniform uint frame_count;

// we need to declare an output for the fragment shader
layout(location = 0) out vec4 outColor; 

vec3 ACESFilm(vec3 x)
{
    // x *= 0.6f;
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), vec3(0), vec3(1));
}

vec3 uncharted2_tonemap_partial(vec3 x)
{
    float A = 0.15;
    float B = 0.50;
    float C = 0.10;
    float D = 0.20;
    float E = 0.02;
    float F = 0.30;
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

vec3 uncharted2_filmic(vec3 v)
{
    float exposure_bias = 2.0;
    vec3 curr = uncharted2_tonemap_partial(v * exposure_bias);

    vec3 W = vec3(11.2);
    vec3 white_scale = vec3(1.0) / uncharted2_tonemap_partial(W);
    return curr * white_scale;
}

void main() {
    vec3 color = texelFetch(u_inputBuffer, ivec2(gl_FragCoord.xy), 0).rgb;
    /*if(int(gl_FragCoord.x) < int(frame_count) % 512 )
        color = ACESFilm(color);
    else
        color = uncharted2_filmic(color);*/
    outColor = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);  
}