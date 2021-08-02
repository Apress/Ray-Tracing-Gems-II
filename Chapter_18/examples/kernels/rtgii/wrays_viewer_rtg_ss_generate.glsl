#version 300 es
precision highp float;
precision highp int;

layout(location = 0) out vec4 accumulation_OUT;
layout(location = 1) out vec4 origin_OUT;
layout(location = 2) out vec4 direction_OUT;
layout(location = 3) out vec4 payload_OUT;

uniform ivec4 tile;
uniform int tile_x;
uniform ivec2 frame;
uniform uvec2 seed;

/* Camera */
uniform vec3 camera_pos;
uniform vec3 camera_up;
uniform vec3 camera_front;
uniform vec3 camera_right;

vec4 rg_Random(uint index, uint seed0, uint seed1);

void main() {
  vec4 tilef = vec4(tile);
  vec2 framef = vec2(frame);
  vec2 pixel = tilef.xy * tilef.zw + gl_FragCoord.xy;
  
  // Generate random numbers   
  uint counter = uint(pixel.x) + uint(pixel.y) * uint(frame.x) + 1u;
  vec4 randoms = rg_Random(counter, seed.x, seed.y);

  float sx = (pixel.x / framef.x + randoms.x / framef.x);
  float sy = (pixel.y / framef.y + randoms.y / framef.y);
  
  float tanFOV2 = tan(radians(80.0) / 2.0);    
  vec3 cx = tanFOV2 * normalize(cross(camera_front, camera_up));
  vec3 cy = (tanFOV2 / (framef.x / framef.y)) * normalize(cross(cx, camera_front));
    
  vec3 ray_direction = normalize(2.0 * (sx - 0.5) * cx + 2.0 * (sy - 0.5) * cy + camera_front);
  
  vec4 accum = vec4(0, 0, 0, 0);
  direction_OUT = vec4(ray_direction, 10000.0);
  origin_OUT = vec4(camera_pos, 0.0);
  accumulation_OUT = accum;
  payload_OUT = vec4(1,1,1,0); // throughput
}

#define m4x32_0 0xD2511F53u
#define m4x32_1 0xCD9E8D57u
#define w32_0 0x9E3779B9u
#define w32_1 0xBB67AE85u

/* compute the upper 32 bits of the product of two unsigned 32-bit integers */
void umulExtended_(uint a, uint b, out uint hi, out uint lo) {
    const uint WHALF = 16u;
    const uint LOMASK = (1u<<WHALF)-1u;
    lo = a*b;               /* full low multiply */
    uint ahi = a>>WHALF;
    uint alo = a& LOMASK;
    uint bhi = b>>WHALF;
    uint blo = b& LOMASK;

    uint ahbl = ahi*blo;
    uint albh = alo*bhi;

    uint ahbl_albh = ((ahbl&LOMASK) + (albh&LOMASK));
    hi = ahi*bhi + (ahbl>>WHALF) +  (albh>>WHALF);
    hi += ahbl_albh >> WHALF; /* carry from the sum of lo(ahbl) + lo(albh) ) */
    /* carry from the sum with alo*blo */
    hi += ((lo >> WHALF) < (ahbl_albh&LOMASK)) ? 1u : 0u;
}

uvec2 philox4x32Bumpkey(uvec2 key) {
    uvec2 ret = key;
    ret.x += 0x9E3779B9u;
    ret.y += 0xBB67AE85u;
    return ret;
}

uvec4 philox4x32Round(uvec4 state, uvec2 key) {
    const uint M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
    uint hi0, lo0, hi1, lo1;
//    umulExtended(M0, state.x, hi0, lo0);
//    umulExtended(M1, state.z, hi1, lo1);
    umulExtended_(M0, state.x, hi0, lo0);
    umulExtended_(M1, state.z, hi1, lo1);

    return uvec4(
        hi1^state.y^key.x, lo1,
        hi0^state.w^key.y, lo0);
}

uvec4 philox4x32_7(uvec4 plain, uvec2 key) {
    uvec4 state = plain;
    uvec2 round_key = key;

    for(int i=0; i<7; ++i) {
        state = philox4x32Round(state, round_key);
        round_key = philox4x32Bumpkey(round_key);
    }

    return state;
}

float uintToFloat(uint src) {
    return uintBitsToFloat(0x3f800000u | (src & 0x7fffffu))-1.0;
}

vec4 uintToFloat(uvec4 src) {
    return vec4(uintToFloat(src.x), uintToFloat(src.y), uintToFloat(src.z), uintToFloat(src.w));
}

vec4
rg_Random(uint index, uint seed0, uint seed1) {
  //philox4x32_key_t k = {{tid, 0xdecafbad}};
  //philox4x32_ctr_t c = {{0, 0xf00dcafe, 0xdeadbeef, 0xbeeff00d}};

  uvec2 key = uvec2( index , seed0 );
  uvec4 ctr = uvec4( 0u , 0xf00dcafeu, 0xdeadbeefu, seed1 );
 

  uvec4 state = ctr;
  uvec2 round_key = key;

  for(int i=0; i<7; ++i) {
    state = philox4x32Round(state, round_key);
    round_key = philox4x32Bumpkey(round_key);
  }

  return uintToFloat(state);
  /*return vec4(
    float(ctr[0]) / float(0xFFFFFFFFu), float(ctr[1]) / float(0xFFFFFFFFu),
    float(ctr[2]) / float(0xFFFFFFFFu), float(ctr[3]) / float(0xFFFFFFFFu)
  );*/
}