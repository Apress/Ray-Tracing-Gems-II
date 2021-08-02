# WebRays: Ray Tracing on the Web 

## Use cases

The complete source code of the examples, described in this chapter, is listed in the following table. For more information about this technology, we refer readers to the official GitHub repository of [WebRays](https://github.com/phasmatic3d/webrays).

| Name | Task (depth) | Scene (#triangles) |
| ---            | ---               | ---      |
| Ray Tracing  ||
| [RTG_II_WebRays_AO_Fireplace](examples/RTG_II_WebRays_AO_Fireplace.html)          | Ambient Occlusion       | Fireplace     (143k) |
| [RTG_II_WebRays_PT_Fireplace](examples/RTG_II_WebRays_PT_Fireplace.html)          | Path Tracing (4)        | Fireplace     (143k) |
| [RTG_II_WebRays_PT_ToyCar](examples/RTG_II_WebRays_PT_ToyCar.html)                | Path Tracing (4)        | ToyCar        (109k) |
| [RTG_II_WebRays_PT_SpaceStation](examples/RTG_II_WebRays_PT_SpaceStation.html)    | Path Tracing (4)        | Space Station (35k)  |
| Hybrid Rendering  ||
| [RTG_II_WebRays_SH_Tree](examples/RTG_II_WebRays_SH_Tree.html)                    | Soft Shadows            | Tree          (51k)  |
| [RTG_II_WebRays_SH_Mosaic](examples/RTG_II_WebRays_SH_Mosaic.html)                | Soft Shadows            | Mosaic        (4.6k) |
| [RTG_II_WebRays_PT_KitchenTable](examples/RTG_II_WebRays_PT_KitchenTable.html)    | Path Tracing (6)        | Kitchen Table (123k) |

## How to use

In order to open each example in a browser, we need to setup a basic HTTP server. 
Current CORS rules in modern browsers will not allow loading of WebAssembly binaries directly from the file system. 

The simplest way to do this is with Python. It comes preinstalled on most operating systems and has modules for setting up a simple HTTP server. To find which version we have, we execute this command in the terminal.

`python --version`

Depending on our version, we change directory to the `examples` folder and execute either: 
 
`python2 -m SimpleHTTPServer`

or
 
`python3 -m http.server`

This will start a local HTTP server that by default listens on the port 8000.

We then have to simply point our browser to `http://localhost:8000/` and click on one demo, for example on the `examples/RTG_II_WebRays_PT_SpaceStation.html` file to navigate into the Space Station scene.

## Resources

These examples would not have been possible without these amazing open-source technologies and free multimedia content:

### 3D Models

The _ToyCar_ and _WaterBottle_ models was downloaded from [Khronos Group repository](https://github.com/KhronosGroup/glTF-Sample-Models/). 
The _San Miguel_ (whose _Tree_ model was used) and _Fireplace_ scenes were downloaded from McGuire's  [Computer Graphics Archive](https://casual-effects.com/data). The remaining scenes (Kitchen Table and Space Station) were created by the authors.

### Textures

The _park_2k.jpg_ HDRI image was downloaded from [HDRIHaven](https://hdrihaven.com/) texture repository.
The _HDR_RGBA_0.png_ image was part of a database of [blue noise textures](http://momentsingraphics.de/Media/BlueNoise/FreeBlueNoiseTextures.zip).

### Open-source Libraries

- [gl-matrix-min.js](https://github.com/toji/gl-matrix)
- [gltf-loader.js](https://github.com/shrekshao/minimal-gltf-loader)
- [UPNG.js](https://github.com/photopea/UPNG.js)
- [webgl-obj-loader.js](https://github.com/frenchtoast747/webgl-obj-loader)
- <a href="https://webassembly.org/"> WebAssembly </a> through <a href="https://emscripten.org/"> emscripten </a>