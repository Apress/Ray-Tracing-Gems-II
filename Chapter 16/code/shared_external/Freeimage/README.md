[FreeImage](https://freeimage.sourceforge.io/intro.html) loads very quickly images (see [all](https://freeimage.sourceforge.io/features.html) supported format) and can apply transformation, like flip, adding channels, histogram, … With this, scenes with textures are loading 10x faster in debug and almost 2x in release.

To use FreeImage, add the package to CMakeLists.txt and use C++17, because I’m using filesystem for handling file paths.

```cmake
_add_package_FreeImage()
set(CMAKE_CXX_STANDARD 17)
```

 

If you are using glTF, here is the example to differ [^1] the loading of the image and using FreeImage to load them.

 
```C++
#include "fileformats/tiny_gltf_freeimage.h"
…
// Differ image loading
tGltf.SetImageLoader(nullptr, nullptr);
tGltf.LoadASCIIFromFile(&gltfModel, &error, &warn, filename);
// Load external images
tinygltf::loadExternalImages(&gltfModel, filename);
```

>**Note**: the images will have **BGR order**, make sure to use the right VkFormat.

>**Note2**: The C++ wrapper is also present, which simplifies dramatically loading and handling images.