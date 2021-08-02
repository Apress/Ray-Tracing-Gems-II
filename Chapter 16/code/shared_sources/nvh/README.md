# Generic Helpers

Non-exhaustive list of utilities provided in the `nvh` directory

Table of Contents:
- [appwindowprofiler.hpp:](#appwindowprofilerhpp)
  - class [nvh::AppWindowProfiler](#class-nvhappwindowprofiler)
- [bitarray.hpp:](#bitarrayhpp)
  - class [nvh::BitArray](#class-nvhbitarray)
- [cameracontrol.hpp:](#cameracontrolhpp)
  - class [nvh::CameraControl](#class-nvhcameracontrol)
- [cameramanipulator.hpp:](#cameramanipulatorhpp)
  - class [nvh::CameraManipulator](#class-nvhcameramanipulator)
- [fileoperations.hpp:](#fileoperationshpp)
- [geometry.hpp:](#geometryhpp)
- [gltfscene.hpp:](#gltfscenehpp)
- [inputparser.h:](#inputparserh)
  - class [nvh::InputParser](#class-nvhinputparser)
- [misc.hpp:](#mischpp)
- [nvprint.hpp:](#nvprinthpp)
- [parametertools.hpp:](#parametertoolshpp)
  - class [nvh::ParameterList](#class-nvhparameterlist)
  - class [nvh::ParameterSequence](#class-nvhparametersequence)
- [profiler.hpp:](#profilerhpp)
  - class [nvh::Profiler](#class-nvhprofiler)
- [radixsort.hpp:](#radixsorthpp)
- [shaderfilemanager.hpp:](#shaderfilemanagerhpp)
  - class [nvh::ShaderFileManager](#class-nvhshaderfilemanager)
- [trangeallocator.hpp:](#trangeallocatorhpp)
  - class [nvh::TRangeAllocator](#class-nvhtrangeallocator)

_____

## appwindowprofiler.hpp

### class **nvh::AppWindowProfiler**
**AppWindowProfiler** provides an alternative utility wrapper class around NVPWindow.
It is useful to derive single-window applications from and is used by some
but not all nvpro-samples.

Further functionality is provided :
- built-in profiler/timer reporting to console
- command-line argument parsing as well as config file parsing using the ParameterTools
  see AppWindowProfiler::setupParameters() for built-in commands
- benchmark/automation mode using ParameterTools
- screenshot creation
- logfile based on devicename (depends on context)
- optional context/swapchain interface
  the derived classes nvvk/appwindowprofiler_vk and nvgl/appwindowprofiler_gl make use of this

## bitarray.hpp

### class **nvh::BitArray**

The **BitArray** class implements a tightly packed boolean array using single bits stored in uint64_t values.
Whenever you want large boolean arrays this representation is preferred for cache-efficiency.
The Visitor and **OffsetVisitor** traversal mechanisms make use of cpu intrinsics to speed up iteration over bits.
  
Example:
``` c++
BitArray modifiedObjects(1024);
  
// set some bits
modifiedObjects.setBit(24,true);
modifiedObjects.setBit(37,true);
  
// iterate over all set bits using the built-in traversal mechanism
  
struct MyVisitor {
void operator()( size_t index ){
    // called with the index of a set bit
    myObjects[index].update();
  }
};
  
MyVisitor visitor;
modifiedObjects.traverseBits(visitor);
```

## cameracontrol.hpp

### class **nvh::CameraControl**

**CameraControl** is a utility class to create a viewmatrix
based on mouse inputs.
  
It can operate in perspective or orthographic mode (`m_sceneOrtho==true`).
  
perspective:
- LMB: rotate
- RMB or WHEEL: zoom via dolly movement
- MMB: pan/move within camera plane
  
ortho:
- LMB: pan/move within camera plane
- RMB or WHEEL: zoom via dolly movement, application needs to use `m_sceneOrthoZoom` for projection matrix adjustment
- MMB: rotate
  
The camera can be orbiting (`m_useOrbit==true`) around `m_sceneOrbit` or
otherwise provide "first person/fly through"-like controls.
  
Speed of movement/rotation etc. is influenced by `m_sceneDimension` as well as the 
sensitivity values.

## cameramanipulator.hpp

### class **nvh::CameraManipulator**

This is a camera manipulator help class
It allow to simply do
- Orbit        (LMB)
- Pan          (LMB + CTRL  | MMB)
- Dolly        (LMB + SHIFT | RMB)
- Look Around  (LMB + ALT   | LMB + CTRL + SHIFT)
- Trackball

In a various ways:
- examiner(orbit around object)
- walk (look up or down but stays on a plane)
- fly ( go toward the interest point)

Do use the camera manipulator, you need to do the following
- Call setWindowSize() at creation of the application and when the window size change
- Call setLookat() at creation to initialize the camera look position
- Call setMousePosition() on application mouse down
- Call mouseMove() on application mouse move

Retrieve the camera matrix by calling getMatrix()

See: appbase_vkpp.hpp

Note: There is a singleton `CameraManip` which can be use across the entire application

~~~ C++
// Retrieve/set camera information
CameraManip.getLookat(eye, center, up);
CameraManip.setLookat(eye, center, nvmath::vec3f(m_upVector == 0, m_upVector == 1, m_upVector == 2));
CameraManip.getFov();
CameraManip.setSpeed(navSpeed);
CameraManip.setMode(navMode == 0 ? nvh::CameraManipulator::Examine : nvh::CameraManipulator::Fly);
// On mouse down, keep mouse coordinates
CameraManip.setMousePosition(x, y);
// On mouse move and mouse button down
if(m_inputs.lmb || m_inputs.rmb || m_inputs.mmb)
{
CameraManip.mouseMove(x, y, m_inputs);
}
// Wheel changes the FOV
CameraManip.wheel(delta > 0 ? 1 : -1, m_inputs);
// Retrieve the matrix to push to the shader
m_ubo.view = CameraManip.getMatrix();	
~~~

## fileoperations.hpp

### functions in nvh

- **fileExists** : check if file exists
- **findFile** : finds filename in provided search directories
- **loadFile** : (multiple overloads) loads file as std::string, binary or text, can also search in provided directories
- **getFileName** : splits filename from filename with path
- **getFilePath** : splits filepath from filename with path

## geometry.hpp

### namespace nvh::geometry
The geometry namespace provides a few procedural mesh primitives
that are subdivided.

**nvh::geometry::Mesh** template uses the provided TVertex which must have a 
constructor from **nvh::geometry::Vertex**. You can also use **nvh::geometry::Vertex**
directly.

It provides triangle indices, as well as outline line indices. The outline indices
are typical feature lines (rectangle for plane, some circles for sphere/torus).

All basic primitives are within -1,1 ranges along the axis they use

- **nvh::geometry::Plane** (x,y subdivision)
- **nvh::geometry::Box** (x,y,z subdivision, made of 6 planes)
- **nvh::geometry::Sphere** (lat,long subdivision)
- **nvh::geometry::Torus** (inner, outer circle subdivision)
- **nvh::geometry::RandomMengerSponge** (subdivision, tree depth, probability)

Example:

``` c++
// single primitive
nvh::geometry::Box<nvh::geometry::Vertex> box(4,4,4);

// construct from primitives

```

## gltfscene.hpp

### namespace nvh::gltf

These utilities are for loading glTF models in a
canonical scene representation. From this representation
you would create the appropriate 3D API resources (buffers
and textures).
 
~~~ C++
// Typical Usage
// Load the GLTF Scene using TinyGLTF
 
tinygltf::Model    gltfModel;
tinygltf::TinyGLTF gltfContext;
fileLoaded = gltfContext.LoadASCIIFromFile(&gltfModel, &error, &warn, m_filename);
 
// Fill the data in the gltfScene
gltfScene.getMaterials(tmodel);
gltfScene.getDrawableNodes(tmodel, GltfAttributes::Normal | GltfAttributes::Texcoord_0);

// Todo in App:
//   create buffers for vertices and indices, from gltfScene.m_position, gltfScene.m_index
//   create textures from images: using tinygltf directly
//   create descriptorSet for material using directly gltfScene.m_materials
~~~

## inputparser.h

### class **nvh::InputParser**
Simple command line parser

Example of usage for: test.exe -f name.txt -size 200 100

Parsing the command line: mandatory '-f' for the filename of the scene

``` c++
nvh::InputParser parser(argc, argv);
std::string filename = parser.getString("-f");
if(filename.empty())  filename = "default.txt";
if(parser.exist("-size") {
      auto values = parser.getInt2("-size");
```

## misc.hpp

### functions in nvh

- **mipMapLevels** : compute number of mip maps
- **stringFormat** : sprintf for std::string
- **frand** : random float using rand()
- **permutation** : fills uint vector with random permutation of values [0... vec.size-1]

## nvprint.hpp

### global nvprintf functions

Multiple functions and macros that should be used for logging purposes,
rather than printf

- **nvprintf** : print at default loglevel
- **nvprintfLevel** : print at a certain loglevel
- **nvprintSetLevel** : sets default loglevel
- **nvprintGetLevel** : gets default loglevel
- **nvprintSetLogFileName** : sets log filename
- **nvprintSetLogging** : sets file logging state
- **nvprintSetCallback** : sets custom callback
- **LOGI** : macro that does nvprintfLevel(LOGLEVEL_INFO)
- **LOGW** : macro that does nvprintfLevel(LOGLEVEL_WARNING)
- **LOGE** : macro that does nvprintfLevel(LOGLEVEL_ERROR)
- **LOGE_FILELINE** : macro that does nvprintfLevel(LOGLEVEL_ERROR) combined with filename/line
- **LOGD** : macro that does nvprintfLevel(LOGLEVEL_DEBUG) (only in debug builds)
- **LOGOK** : macro that does nvprintfLevel(LOGLEVEL_OK)
- **LOGSTATS** : macro that does nvprintfLevel(LOGLEVEL_STATS)

## parametertools.hpp

### class **nvh::ParameterList**

The **ParameterList** helps parsing commandline arguments
or commandline arguments stored within ascii config files.

**Parameters** always update the values they point to, and optionally
can trigger a callback that can be provided per-parameter.

``` c++
ParameterList list;
std::string   modelFilename;
float         modelScale;

list.addFilename(".gltf|model filename", &modelFilename);
list.add("scale|model scale", &modelScale);

list.applyTokens(3, {"blah.gltf","-scale","4"}, "-", "/assets/");
``` 

Use in combination with the **ParameterSequence** class to iterate
sequences of parameter changes for benchmarking/automation.


### class **nvh::ParameterSequence**

The **ParameterSequence** processes provided tokens in sequences.
The sequences are terminated by a special "separator" token.
All tokens between the last iteration and the separator are applied
to the provided **ParameterList**.
Useful to process commands in sequences (automation, benchmarking etc.).
  
Example:
  
``` c++
ParameterSequence sequence;
ParameterList     list;
int               mode;
list.add("mode", &mode);
  
std::vector<const char*> tokens;
ParameterList::tokenizeString("benchmark simple -mode 10 benchmark complex -mode 20", tokens);
sequence.init(&list, tokens);
  
   // 1 means our separator is followed by one argument (simple/complex)
   // "-" as parameters in the string are prefixed with -
  
while(!sequence.advanceIteration("benchmark", 1, "-")) {
  printf("%d %s mode %d\n", sequence.getIteration(), sequence.getSeparatorArg(0), mode);
}

// would print:
//   0 simple mode 10
//   1 complex mode 20
```

## profiler.hpp

### class **nvh::Profiler**

The **Profiler** class is designed to measure timed sections.
Each section has a cpu and gpu time. Gpu times are typically provided
by derived classes for each individual api (e.g. OpenGL, Vulkan etc.).

There is functionality to pretty print the sections with their nesting level.
Multiple profilers can reference the same database, so one profiler
can serve as master that they others contribute to. Typically the
base class measuring only CPU time could be the master, and the api
derived classes reference it to share the same database.

**Profiler::Clock** can be used standalone for time measuring.

## radixsort.hpp

### function nvh::radixsort

The radixsort function sorts the provided keys based on
BYTES many bytes stored inside TKey starting at BYTEOFFSET.
The sorting result is returned as indices into the keys array.

For example:

``` c++
struct MyData {
  uint32_t objectIdentifier;
  uint16_t objectSortKey;
};


// 4-byte offset of objectSortKey within MyData
// 2-byte size of sorting key

result = radixsort<4,2>(keys, indicesIn, indicesTemp);

// after sorting the following is true

keys[result[i]].objectSortKey < keys[result[i + 1]].objectSortKey

// result can point either to indicesIn or indicesTemp (we swap the arrays
// after each byte iteration)
```

## shaderfilemanager.hpp

### class **nvh::ShaderFileManager**

The **ShaderFileManager** class is meant to be derived from to create the actual api-specific 
shader/program managers.

The **ShaderFileManager** provides a system to find/load shader files.
It also allows resolving #include instructions in HLSL/GLSL source files.
Such includes can be registered before pointing to strings in memory.

Furthermore it handles injecting prepended strings (typically used for #defines) 
after the #version statement of GLSL files.

## trangeallocator.hpp

### class **nvh::TRangeAllocator**

The **TRangeAllocator**<GRANULARITY> template allows to sub-allocate ranges from a fixed
maximum size. **Ranges** are allocated at GRANULARITY and are merged back on freeing.
Its primary use is within allocators that sub-allocate from fixed-size blocks.

The implementation is based on [MakeID by Emil Persson](http://www.humus.name/3D/MakeID.h).

Example :

~~~ C++
TRangeAllocator<256> range;

// initialize to a certain range
range.init(range.alignedSize(128 * 1024 * 1024));

...

// allocate a sub range
// example
uint32_t size = vertexBufferSize;
uint32_t alignment = vertexAlignment;

uint32_t allocOffset;
uint32_t allocSize;
uint32_t alignedOffset;

if (range.subAllocate(size, alignment, allocOffset, alignedOffset, allocSize)) {
  ... use the allocation space
  // [alignedOffset + size] is guaranteed to be within [allocOffset + allocSize]
}

// give back the memory range for re-use
range.subFree(allocOffset, allocSize);

...

// at the end cleanup
range.deinit();
~~~




_____
auto-generated by `docgen.lua`
