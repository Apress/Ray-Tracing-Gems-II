# Errata for _Ray Tracing Gems II_

Corrections for the book [_Ray Tracing Gems II_](http://raytracinggems.com). The pages listed are for the released book, not for any preprints or other forms of the articles. A corrected copy of the free PDF version of the book can be found linked from [http://raytracinggems.com](http://raytracinggems.com) as errors are discovered and fixed.

## If you find an error, please let us know by emailing [raytracinggems@nvidia.com](mailto:raytracinggems@nvidia.com)

## Chapter 3

**Page 49: Listing 3-2**, the value computed for ```aspectScale``` should be divided by 2. This listing is also is missing ```pixel += vec2(0.5f) - (imageSize / 2.0f);``` after computing ```aspectScale```.

The correct GLSL shader code for Listing 3-2 is:
```GLSL
// Pixel is the integer position >= 0 and < imageSize.
Ray generatePinholeRay(vec2 pixel) {
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
Ray paniniRay(vec2 pixel) {
    vec2 pixelCenterCoords = vec2(pixel) + vec2(0.5f) - (imageSize / 2.f);

    float halfFOV = cameraFovAngle / 2.f;
    float halfPaniniFOV = atan(sin(halfFOV), cos(halfFOV) + paniniDistance);
    vec2 hvPan = vec2(tan(halfPaniniFOV)) * (1.f + paniniDistance); 
    
    hvPan *= pixelCenterCoords / (bool(cameraFovOrientation == 0) ? imageSize.x : imageSize.y);
    hvPan.x = atan(hvPan.x / (1.0 + paniniDistance));

    float M = sqrt(1 - square(sin(hvPan.x) * paniniDistance)) + paniniDistance * cos(hvPan.x);

    float x = sin(hvPan.x) * M;
    float z = (cos(hvPan.x) * M) - paniniDistance;

    float S = (d + 1) / (d + z);

    float y = lerp(hvPan.y / S, hvPan.y * z, verticalCompression);

    vec3 direction = normalize(x, -y, -z);

    return Ray(vec3(0.f), 0.f, direction, INFINITY);
}
```

**Page 55: Listing 3-7**, lines 6, 8, and 10 are incorrect.

The correct GLSL shader code for Listing 3-7 is:
```GLSL
Ray generateFisheyeRay(vec2 pixel) {
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

## Chapter 17

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

_Thanks to Zander Majercik, [@hatookov](https://twitter.com/hatookov), and Jeremy Ong for reporting these errors._

Page last updated **August 31, 2021**
