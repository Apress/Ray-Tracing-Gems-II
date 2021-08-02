# The Reference Path Tracer Code sample

This is a code sample accompanying Chapter 14 "The Reference Path Tracer" in Ray Tracing Gems 2 book. Code is based on the [IntroToDXR](https://github.com/acmarrs/IntroToDXR/) and implements a path tracer described in the article. The sample application loads GLTF scenes, specified via command line. The core of the path tracer can be found in single HLSL file - `PathTracer.hlsl`.

## Command Line Arguments

* `-width [integer]` specifies the width (in pixels) of the rendering window
* `-height [integer]` specifies the height (in pixels) of the rendering window
* `-vsync [0|1]` specifies whether vsync is enabled or disabled
* `-scene [path]` specifies the file path to a GLTF model
* `-scenePath [path]` optionally specifies the root folder where scene data files can be found

## Licenses and Open Source Software

The code uses the following dependencies:
* [d3dx12.h](https://github.com/Microsoft/DirectX-Graphics-Samples/tree/master/Libraries/D3DX12), provided with an MIT license. 
* [stb_image.h](https://github.com/nothings/stb/blob/master/stb_image.h), provided with an MIT license.
* [DirectXTex](https://github.com/Microsoft/DirectXTex), provided with an MIT license.
* [DirectXTK](https://github.com/Microsoft/DirectXTK), provided with an MIT license.
* [DXC Compiler](https://github.com/microsoft/DirectXShaderCompiler), provided with an University of Illinois Open Source
* [ImGUI](https://github.com/ocornut/imgui), provided with an MIT license.
* [tiny glTF](https://github.com/syoyo/tinygltf), provided with an MIT license.
* [brdf.h](https://github.com/boksajak/brdf), provided with CC0 license.

The repository includes assets for use when testing the renderer:
[Bathroom](https://www.blendswap.com/blend/6369), by cenobi, licensed under a [Creative Commons Attribution 3.0 License](https://creativecommons.org/licenses/by/3.0/).


