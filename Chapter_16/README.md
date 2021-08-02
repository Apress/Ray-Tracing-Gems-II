# Ray Tracing Gems 2: Introduction to Vulkan Ray Tracing: Sample Code

Hybrid rasterization/ray tracing ambient occlusion sample for the Ray Tracing Gems 2 introduction to Vulkan Ray Tracing. It uses only ray tracing to render a linear-space image of a scene's ambient occlusion.

This is built to be a runnable version of the sample code from the article, with relatively minimal reliance on helpers, and written in a medium-length offline (non-interactive) renderer fashion. This is useful if you're looking for the core of ray tracing at a low level, for instance. Depending on what you're looking for in a ray tracing sample, other resources may be useful. For instance,

- nvpro-samples' [gl_vk_raytrace_interop](https://github.com/nvpro-samples/gl_vk_raytrace_interop) is a longer sample with more use of helpers showing how to render hybrid ray-traced ambient occlusion using ray tracing pipelines in an interactive context, using OpenGL for rasterization and including blurring. It uses the Vulkan-Hpp C++ bindings.
- nvpro-samples' [vk_mini_path_tracer](https://github.com/nvpro-samples/vk_mini_path_tracer) is a path tracing introduction to Vulkan's C API for beginners, with an emphasis on minimizing code length, and uses more helpers. It also starts with ray queries instead of ray tracing pipelines, before moving to ray tracing pipelines later on.
- nvpro-samples' [vk_raytracing_tutorial_KHR](https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR) is a comprehensive introduction to Vulkan ray tracing pipelines, showing how to modify a rasterizer to perform ray tracing in an interactive context. It also uses the Vulkan-Hpp C++ bindings.
- [Falcor](https://developer.nvidia.com/falcor) is a real-time rendering framework designed for rapid graphics research prototyping using DXR. It's useful if one wants to build something on top of ray tracing, for instance.

## Cloning and Building

This sample supports both Windows and Linux.

To clone this sample along with its submodules, run

```
git clone --recurse-submodules [Git URL]
```

You'll also need to install at least version 1.2.162.0 of the Vulkan SDK. You can download the latest version of the SDK for Windows, Mac, and Linux at [https://vulkan.lunarg.com/sdk/home](https://vulkan.lunarg.com/sdk/home); if your distro is compatible with Ubuntu packages, it may be easier to download the Vulkan SDK package following the instructions on the LunarG VulkanSDK Packages website at [https://packages.lunarg.com/](https://packages.lunarg.com/).

Then configure and generate the project using CMake on `CMakeLists.txt`. Make sure CMake's build directory is not the same folder as the source directory, and note that this project requires C++14 support. For instance, to generate the project for Visual Studio 2019, run

```
mkdir cmake_build
cd cmake_build
cmake .. -G "Visual Studio 16 2019" -A x64
```

or use the CMake GUI.

Finally, compile and run the `vk_ray_tracing_gems_2_ao` target!

## License

This project uses the Apache 2.0 license. Please see the copyright notice in the [LICENSE](LICENSE) file.

This project also uses the NVIDIA nvpro-samples framework. Please see the license for nvpro-samples' shared_sources [here](https://github.com/nvpro-samples/shared_sources/blob/master/LICENSE.md), and the third-party packages it uses in shared_external [here](https://github.com/nvpro-samples/shared_external/blob/master/README.md).