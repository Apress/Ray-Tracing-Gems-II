# Errata for _Ray Tracing Gems II_

Corrections for the book [_Ray Tracing Gems II_](http://raytracinggems.com). The pages listed are for the released book, not for any preprints or other forms of the articles. A corrected copy of the free PDF version of the book can be found linked from [http://raytracinggems.com](http://raytracinggems.com) as errors are discovered and fixed.

## If you find an error, please let us know by emailing [raytracinggems@nvidia.com](mailto:raytracinggems@nvidia.com)

## Chapter 3: Essential Ray Generation Shaders

**Page 49: Listing 3-2**, the value computed for ```aspectScale``` should be divided by 2. This listing is also is missing ```pixel += vec2(0.5f) - (imageSize / 2.0f);``` after computing ```aspectScale```.

The correct GLSL shader code for Listing 3-2 is:
```GLSL
// Pixel is the integer position >= 0 and < imageSize.
Ray generatePinholeRay(vec2 pixel)
{
  float tanHalfAngle = tan(cameraFovAngle / 2.f);
  float aspectScale = ((cameraFovDirection == 0) ? imageSize.x : imageSize.y) / 2.f;
  
  pixel += vec2(0.5f) - (imageSize / 2.f);
  
  vec3 direction = normalize(vec3(vec2(pixel.x, -pixel.y) * tanHalfAngle / aspectScale, -1));
  
  return Ray(vec3(0.f), direction, 0.f, INFINITY);
}
```

**Page 54: Listing 3-6**, the variable ```hvPan``` is used but not declared or defined. The variable ```distance``` should be defined as ```normalize(x, -y, -z);```.

The correct GLSL shader code for Listing 3-6 is:
```GLSL
Ray paniniRay(vec2 pixel)
{
    vec2 pixelCenterCoords = vec2(pixel) + vec2(0.5f) - (imageSize / 2.f);

    float halfFOV = cameraFovAngle / 2.f;
    float halfPaniniFOV = atan(sin(halfFOV), cos(halfFOV) + paniniDistance);
    vec2 hvPan = vec2(tan(halfPaniniFOV)) * (1.f + paniniDistance); 
    
    hvPan *= pixelCenterCoords / (bool(cameraFovOrientation == 0) ? imageSize.x : imageSize.y);
    hvPan.x = atan(hvPan.x / (1.0 + paniniDistance));

    float M = sqrt(1 - square(sin(hvPan.x) * paniniDistance)) + paniniDistance * cos(hvPan.x);

    float x = sin(hvPan.x) * M;
    float z = (cos(hvPan.x) * M) - paniniDistance;

    float S = (paniniDistance + 1) / (paniniDistance + z);

    float y = lerp(hvPan.y / S, hvPan.y * z, verticalCompression);

    vec3 direction = normalize(x, -y, -z);

    return Ray(vec3(0.f), 0.f, direction, INFINITY);
}
```

**Page 55: Listing 3-7**, lines 6, 8, and 10 are incorrect.

The correct GLSL shader code for Listing 3-7 is:
```GLSL
Ray generateFisheyeRay(vec2 pixel)
{
    vec2 clampedHalfFOV = min(cameraFovAngle, pi) / 2.f;
    vec2 angle = (pixel - imageSize / 2.f) * clampedHalfFOV;

    if (cameraFovOrientation == 0) {
        angle /= imageSize.x / 2.f;
    } else if (cameraFovOrientation == 1) {
        angle /= imageSize.y / 2.f;
    } else {
        angle /= length(imageSize) / 2.f;
    }

    // Don't generate rays for pixels outside the fisheye
    // (circle and cropped circle only).
    if (length(angle) > 0.5.f * pi) {
        return Ray(vec3(0.f), 0.0f, vec3(0.f), -1);
    }
    vec3 dir = normalize(vec3(sin(angle.x), -sin(angle.y) * cos(angle.x), -cos(angle.x) * cos(angle.y)));
    return Ray(vec3(0.f), 0.f, dir, INFINITY);
}
```

## Chapter 5: Sampling Textures with Missing Derivatives

_Note: all scalars in this chapter can be coalesced into a single scalar, which multiplies two `float2`. After the below corrections, the code contains 15 scalar multiplications, but this can be reduced to 5. The corrected code removes the optimizations for clarity._

**Page 81: Section 5.2.3**, the ```BaryCentricWorldDerivatives()``` code listing near the bottom of the page has a few errors.

The correct code listing is:
```C++
void BarycentricWorldDerivatives(float3 A1, float3 A2, out float3 du_dx, out float3 dv_dx)
{
    float3 Nt = cross(A1, A2);
    du_dx = cross(A2, Nt) / dot(Nt, Nt);
    dv_dx = cross(Nt, A1) / dot(Nt, Nt);
}
```

**Page 82: Section 5.2.4**, the ```WorldScreenDerivatives()``` code listing in the middle of the page has an error related to the `wMx` variable.

The correct code listing is:
```C++
float3x3 WorldScreenDerivatives(float4x4 WorldToTargetMatrix, float4x4 TargetToWorldMatrix, float4 x)
{
    float wMx = dot(WorldToTargetMatrix[3], x);
    float3x3 dx_dxt = (float3x3)TargetToWorldMatrix;
    dx_dxt[0] = wMx * (dx_dxt[0] - x.x * TargetToWorldMatrix[3].xyz);
    dx_dxt[1] = wMx * (dx_dxt[1] - x.y * TargetToWorldMatrix[3].xyz);
    dx_dxt[2] = wMx * (dx_dxt[2] - x.z * TargetToWorldMatrix[3].xyz);
    return dx_dxt;
}
```

**Page 83: Section 5.2.6**, the ```BarycentricDerivatives()``` code listing is updated based on the `wMx` variable changes in `WorldScreenDerivatives()`.

The correct code listing is:
```C++
float2x2 BarycentricDerivatives(float4 x, float3 n, float3 x0, float3 x1, float3 x2,
        float4x4 WorldToTargetMatrix , float4x4 TargetToWorldMatrix)
{
    // Derivatives of barycentric coordinates with respect to
    // world-space coordinates (Section 5.2.3).
    float3 du_dx , dv_dx;
    BarycentricWorldDerivatives(x1 - x0, x2 - x0, du_dx , dv_dx);

    // Partial derivatives of world-space coordinates with respect
    // to screen-space coordinates (Section 5.2.4). (Only the
    // relevant 3x3 part is considered.)
    float3x3 dx_dxt = WorldScreenDerivatives(WorldToTargetMatrix, TargetToWorldMatrix, x);

    // Partial derivatives of barycentric coordinates with respect
    // to screen-space coordinates.
    float3 du_dxt = du_dx.x * dx_dxt[0] + du_dx.y * dx_dxt[1] + du_dx.z * dx_dxt[2];
    float3 dv_dxt = dv_dx.x * dx_dxt[0] + dv_dx.y * dx_dxt[1] + dv_dx.z * dx_dxt[2];

    // Derivatives of barycentric coordinates with respect to
    // screen-space x and y coordinates (Section 5.2.5).
    float2 ddepth_dXY = DepthGradient(x, n, TargetToWorldMatrix);
    float2 du_dXY = du_dxt.xy + du_dxt.z * ddepth_dXY;
    float2 dv_dXY = dv_dxt.xy + dv_dxt.z * ddepth_dXY;
    return float2x2(du_dXY , dv_dXY);
}
```

## Chapter 10: Texture Coordinate Gradients Estimation for Ray Cones

**Page 118**: Figure 10-4, the angle that is denoted α should instead be α/2

**Page 119**: Under equation 10-1, the 2nd sentence of the paragraph says we rotate by "+α" and "-α". It should read "+α/2" and "-α/2".

## Chapter 17: Using Bindless Resources with DirectX Raytracing

**Page 260**: line 17 of the code listing is incorrect. The ```HitGroupRecord``` structure's ```Padding1``` variable should be removed since it is not necessary and its inclusion causes the structure size to *not* be a multiple of 32 bytes as required.

The correct code for the ```HitGroupRecord``` structure is:
```C++
struct HitGroupRecord
{
  ShaderIdentifier ID;
  D3D12_GPU_DESCRIPTOR_HANDLE SRVTableA = { };
  D3D12_GPU_DESCRIPTOR_HANDLE SRVTableB = { };
  uint64_t CBV = 0;
  uint8_t Padding [8] = { }; // Needed to keep shader ID at 32-byte alignment
};
```

### Lesser Errors

None so far!

_Thanks to Zander Majercik, [@hatookov](https://twitter.com/hatookov), Jeremy Ong, Leon Brands, and Tomas Akenine-Möller for reporting these errors._

Page last updated **September 20, 2021**
