# OWL: A Node Graph "Wrapper" Library for OptiX 7

<!--- ------------------------------------------------------- -->
## What is OWL?

OWL is a convenience/productivity-oriented library on top of OptiX
7.x, and aims at making it easier to write OptiX programs by taking
some of the more arcane arts (like knowing what a Shader Binding Table
is, and how to actually build it), and doing that for the user. For
example, assuming the node graph (ie, the programs, geometries, and
acceleration structures) have already been built, the shader binding
table (SBT) can be built and properly populated by a single call
`owlBuildSBT(context)`.

In addition, OWL also allows for somewhat higher-level abstractions
than native OptiX+CUDA for operations such as creating device buffers,
uploading data, building shader programs and pipelines, building
acceleration structures, etc.

## Who is OWL designed/intended for?

OWL is particularly targetted at two groups of users: First, those
that do want to use GPU Ray Tracing and RTX hardware acceleration, and
that are comfortable with typical GPU concepts such as GPU memory vs
device memory, ray tracing pipeline, shader programs, and some CUDA
programming - but that are *not* "Ninja" OptiX/Vulkan/DirectX users,
and might not be 100% sure about the most nitty-bitty grits of details
on SBT data layout and order, or on just how exactly to do the BVH
compaction, how exactly to deal with async launches or refitting, etc.

Second, it targets those that *do* know all these concepts, but would
rather spent their time on the actual shader programs and
functionality of the program, rather than on doing and all the
low-level steps themselves; ie, those that are willing to trade a bit
of low-level control (and *maybe* some tiny amount of performance) for
higher developing productivity.

## Simple Example

As an example of how easy it is to use OWL to build OptiX data
strucutres, the following example code snippet takes a host-based
triangle mesh and:

- uploads the index and vertex buffer to the active GPU(s)
- creates a triangle mesh geometry with a sample 'color' SBT entry
- puts this mesh into a triangle bottom-level accel structure (BLAS)
- builds that acceleration structure, including BVH compaction
- creates an instance with an instance transform, and finally
- builds and returns an instance acceleration structure over that.

Note how this little example will do these step: including data
upload, set-up of build inputs, BVH construction, BVH compaction, and
everything else that's required for this. Though still a relatively
benign example, doing the same in low-level CUDA and OptiX code would
result in significantly more code that the user would have to write,
debug, and maintain.

    /* simple sample of setting up a full geometry, BLAS and
	   IAS for a simple single-triangle mesh model */
    OWLGroup buildBlasAndIas(max3x4f             &instXfm,
	                         std::vector<float3> &vtx,
	                         std::vector<int3>   &idx,
							 float3               color)
	{
	   /* upload the buffers */
	   OWLBuffer vtxBuffer
	      = owlDeviceBufferCreate(ctx,OWL_FLOAT3,
		                          vtx.size(),vtx.data());
	   OWLBuffer idxBuffer
	      = owlDeviceBufferCreate(ctx,OWL_INT3,
		                          idx.size(),idx.data());

	   /* create triangle mesh geometry */
	   OWLGeom mesh = owlGeomCreate(ctx,myMeshGT);
	   owlTrianglesSetVertices(mesh,vtxBuffer,vtx.size(),
	                          /*stride+ofs*/sizeof(vtx[0],0);
	   owlTrianglesSetIndices(mesh,vtxBuffer,vtx.size(),
	                          /*stride+ofs*/sizeof(idx[0]),0);

	   /* create and build triangle BLAS */
	   OWLGroup blas = owlTrianglesGroupCreate(ctx,1,&mesh);
	   owlGroupBuildAccel(blas);

	   /* create and build instance accel struct (IAS) */
	   OWLGroup ias = owlInstanceGroupCreate(ctx,1,
	       /* instantiated BLASes */&blas,
	       /* instance IDs:       */nullptr,
		   /* instance transforms */&instXfm);
	   owlGroupBuildAccel(ias);
	   return blas; // that's it!
	}

Of course, even with OWL there's still much more that needs to be done
for a full renderer: For example, in this code we assumed that a
context (`ctx`) and a geometry type for this mesh (`myMeshGT`) have
already been created; the user also still has to set up the programs,
create frame buffer and launch data, build the programs
(`owlBuildPrograms()`), the pipline (`owlBuildPipeline()`), and the
SBT (`owlBuildSBT(ctx)`), etc.

## What about Advanced Users?

As stated above, OWL explicitly aims for helping entry-level or casual
RTX users get started, and get working productively with OptiX and RTX
without having to first become an OptiX "Ninja".

However, that is not to mean that it is *only* useful for beginners.
In fact, OWL currently supports lots of rather advanced features as
well, including, for example:
- multi-level instancing
- accel structure refitting (compaction is always on)
- multiple raygen programs and multiple ray types
- motion blur, including instance motion blur
- multi-GPU support, including proper handling of entities that might
be different per GPU (such as buffers, traversables, and textures)
- async launches
- different buffer types including pinned and managed memory, and
including buffers of buffers, buffers of traversables, and buffers of
textures
- textures, with different formats and filter modes
- triangle mesh and user-defined geometry types (curves to be supported soon)
- etc

In particular for advanced users, OWL is *explicitly* intended to
allow advanced users to mix OWL code and data structures with other,
manually written CUDA code if and whenever so desired. For example,
OWL offers functions to easily query the CUDA device-addresses of
buffers, OptixTraversableHandle's from groups, CUDA streams from
launches, etc. As such, it is absolutely possible to mix OWL and CUDA
code by, for example, having a multi-pass renderer in which CUDA does
all the shading code and set-up of ray streams, and OWL doing the
acceleration structure build and (RTX hardware-accelerated) tracing of
these ray streams, even in multi-threaded and multi-GPU settings, with
proper CUDA streams, etc (in fact, I do that in several of my own OWL
applications).

<!--- ------------------------------------------------------- -->
# Current State of Development

OWL was first publicly released early 2019, and has been used in
several research/paper projects (see below). OWL initially targetted a
much smaller scope of work - initially it was supposed to be only a
"wrapper" around things like building acceleration structures (hence
the name "OptiX *Wrapper* Library"), but the need for a higher
abstraction level soon became evident, primarily due to the need to
help users build and populate the SBT - which needs more "global"
information than a single acceleration structure.

Despite these significant changes after the initial release, the
current abstraction level and API have remained stable over roughly a
year now, with only relatively minor additions such as buffers of
buffers, refitting, textures, or motion blur. Some features will still
need adding (e.g., curves, which got added to OptiX 7.1 but are not yet
exposed in OWL); however, we consider the current release to be
sufficiently stable to finally have given it the long-awaited "version
1.x".

<!--- ------------------------------------------------------- -->
# Sample Use Cases

Some sample use projects/papers that recently used OWL:

- Moana on OWL/OptiX (Oct 2020)

  (https://ingowald.blog/2020/10/26/moana-on-rtx-first-light/)

  ![Sample "Moana on OWL/OptiX" images](doc/jpg/collage-moana.jpg)

- "VisII - A Python-Scriptable Virtual Scene Imaging Interface (2020)

  (https://github.com/owl-project/ViSII)

  ![Sample "VisII" images](doc/jpg/collage-visii.jpg)

- "Ray Tracing Structured AMR Data Using ExaBricks". I Wald, S
  Zellmann, W Usher, N Morrical, U Lang, and V Pascucci. IEEE
  TVCG(Proceedings of IEEE Vis
  2020).

  (https://www.willusher.io/publications/exabrick)

  ![Sample "ExaBricks" images (Image Credits: See Authors)](doc/jpg/collage-exabricks.jpg)

- "Accelerating Force-Directed Graph Drawing with RT Cores".  S
  Zellmann, M Weier, I Wald, IEEE Vis Short
  Papers 2020.

  (https://arxiv.org/pdf/2008.11235.pdf)


 - "A Virtual Frame Buffer Abstraction for Parallel Rendering of Large
  Tiled Display Walls". M Han, I Wald, W Usher, N Morrical, A Knoll, V
  Pascucci, C R Johnson. IEEE Vis Short Papers 2020.

  (http://www.sci.utah.edu/~wald/Publications/2020/dw2/dw2.pdf)

- "Spatial Partitioning Strategies for Memory-Efficient Ray Tracing of
  Particles".  P Gralka, I Wald, S Geringer, G Reina, Th Ertl. IEEE
  Symposium on Large Data Analysis and Viusalization (LDAV) 2020.

- "Finding Efficient Spatial Distributions for Massively Instanced 3-d
  Models".  S Zellmann, N Morrical, I Wald, V Pascucci.  Eurographics
  Symposium on Parallel Graphics and Visualization (EGPGV 2020).

  (https://vis.uni-koeln.de/forschung/publikationen/finding-efficient-spatial-distributions-for-massively-instanced-3-d-models)

  ![Sample "Data Parallel Ray Tracing w/ OWL" images (Image Credits: See Authors)](doc/jpg/collage-instances.jpg)

- "High-Quality Rendering of Glyphs Using Hardware-Accelerated Ray
  Tracing".  S Zellmann, M Aumüller, N Marshak, I Wald.  Eurographics
  Symposium on Parallel Graphics and Visualization (EGPGV 2020).

  (https://vis.uni-koeln.de/forschung/publikationen/high-quality-rendering-of-glyphs-using-hardware-accelerated-ray-tracing)

  ![Sample "ExaBricks" images (Image Credits: See Authors)](doc/jpg/collage-tubes.jpg)

- "RTX Beyond Ray Tracing: Exploring the Use of Hardware Ray Tracing
  Cores for Tet-Mesh Point Location". I Wald, W Usher, N Morrical, L
  Lediaev, and V Pascucci.  In High Performance Graphics Short Papers,
  2019

  (https://www.willusher.io/publications/rtx-points)

- "Using Hardware Ray Transforms to Accelerate Ray/Primitive
  Intersections for Long, Thin Primitive Types". I Wald, N Morrical, S
  Zellmann, L Ma, W Usher, T Huang, V Pascucci.  Proceedings of the
  ACM on Computer Graphics and Interactive Techniques (Proceedings of
  High Performance Graphics), 2020

  (https://www.willusher.io/publications/owltubes)

- "Efficient Space Skipping and Adaptive Sampling of Unstructured
  Volumes Using Hardware Accelerated Ray Tracing. N Morrical, W
  Usher, I Wald, V Pascucci. In IEEE VIS Short Papers, 2019

  (https://www.willusher.io/publications/rtx-space-skipping)


<!--- ------------------------------------------------------- -->
# Building OWL / Supported Platforms

General Requirements:
- OptiX 7 SDK (version 7.0, 7.1, 7.2, 7.3, or 7.4; should work with either)
- CUDA version 10 or 11
- a C++11 capable compiler (regular gcc on CentOS, Ubuntu, or any other Linux should do; as should VS on Windows)
- OpenGL

Per-OS Instructions:

- Ubuntu 18, 19, and 20 (automatically tested on 18, mostly developed on 20)
    - Dependencies
		- cmake for building (`sudo apt install cmake-curses-gui`)
		- if you want to build the graphical examples: glfw (`sudo apt-get install libglfw3-dev`), or all the libraries to build it from included source code (`sudo apt-get install x11-xserver-utils libxrandr-dev libxinerama-dev libxcb-xkb-dev libxcursor-dev libxcb-xinput-dev libxi-dev`)
	- Build:
	```bash
	mkdir build
	cd build
	cmake ..
	make
	```
- CentOS 7:
    - Requires: `sudo yum install cmake3`
	- Build:
	```bash
	mkdir build
	cd build
	cmake3 ..
	make
	```
	(mind to use `cmake3`, not `cmake`, using the wrong one will mess up the build directory)
- Windows
    - Requires: Visual Studio (both 2017 and 2019 work), OptiX 7.0, cmake
	- Build: Use CMake-GUI to build Visual Studio project, then use VS to build
		- Specifics: source code path is ```...Gitlab/owl```, binaries ```...Gitlab/owl/build```, and after pushing the Configure button choose ```x64``` for the optional platform.
		- You may need to Configure twice.
		- If you get "OptiX headers (optix.h and friends) not found." then define OptiX_INCLUDE manually in CMake-gui by setting it to ```C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.0.0/include```

<!--- ------------------------------------------------------- -->
# Using OWL through CMake

Though you can of course use OWL without CMake, it is highly encouraged
to use OWL as a git submodule, using CMake to configure and build this
submodule. In particular, the suggested procedure is to first
do a `add_subdirectory` with the owl submodules as such:

```cmake
set(owl_dir ${PROJECT_SOURCE_DIR}/whereeverYourOWLSubmoduleIs)
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)
```

(the `EXCLUDE_FROM_ALL` makes sure that your main project won't automatically
build any owl samples or test cases unless you explicitly request so).

Once your project has called `add_subdirectory` on owl, it only has to link the
`owl::owl` target in order to bring in all includes, linked libraries, etc. to
fully use it. This might look like:

```cmake
target_link_libraries(myOwlApp PRIVATE owl::owl)
```

OptiX will need to be in a place that can be found by CMake. Point CMake at your
OptiX directory by adding it to `CMAKE_PREFIX_PATH` (where it works on all
platforms similar to how `LD_LIBRARY_PATH` resolves runtime linking on Linux).
Note that `CMAKE_PREFIX_PATH` can be specified as an environment variable or as
a CMake variable when you run CMake on your project.

<!--- ------------------------------------------------------- -->
# Latest Progress/Revision History

Latest additions, not yet in any release
----------------------------------------------------------------------

- build fix to automatically build glfw on linux if no system-glfw is
  installed (kudos srogatch)

- now handling empty user-geometries gracefully (fixes #147)

v1.1 - Switched to "modern cmake" technology (kudos lpisha, and jda)
----------------------------------------------------------------------

*v.1.1.6*: 

- bugfix: supporting optix 7.4 now.

- renamed all `CUDA_` macros to `OWL_CUDA_` to avoid naming conflicts with other projects
<none yet>


*v.1.1.5*: bugfix: various buffer types didn't properly release all memory.

*v.1.1.4*:

- added new (optional) `EMBEDDED_SYMBOL_NAMES` argument  to `embed_ptx()` to
  permit overriding the C symbol names used (default is `${CU_FILE_NAME}_ptx`)
    - see `tests/t01-many-spheres` as an example of using this feature
- added new (optional) `PTX_TARGET` argument to `embed_ptx()` to allow
  specifying the target name used to compile `.cu` files to PTX
    - see `tests/t02-group-rebuilds` as an example of using this feature

*v.1.1.3*:  bugfix: fixed TBB includes for windows, when sued as submodule

*v.1.1.2*:

- added `owlBufferClear()`, at least for copyable data

- added `owlLaunch3D()`, as requested in feature request issue #123

- added new cmdline sample s10-launch3D that demonstrates both 3D
  launches, and how to do 'interop' between CUDA and OWL (using owl
  for rendering, and CUDA to convert framebuffer from float3 to RGBA8.

*1.1.1*: various fixes for the 'modern cmake' version, with a pretty
   big re-vamp of the entire build system. projects using owl as a
   submodule need to update how they use cmake accordingly; pls
   consult the samples for how to cleanly do that.

*1.1.0*: first release with louis pisha's 'modern cmake' version of the build system.
	Also includes several new samples, including the voxel renderer
	from the "ray tracing gems2" article.

v1.0.x - First "considered to be complete" version
----------------------------------------------------------------------

*1.0.4*: fixed issue #68 - now compiles with optix 7.2, and newewst intel tbb

*1.0.3*: bugfix: no longer fatally failing when memadvise optimization didn't work

*1.0.2*: (finally) fixed long-standing bug in owlViewer that caused
   samples to crash when forcing OWL to run on a GPU that's different
   from the GPU that held the OpenGL graphics context for the viewer. Fixed.

*1.0.1: bugfix for missing owlSet4{}() functions

v0.9.x - Elimination of LL layer, and support for motion blur
----------------------------------------------------------------------

*v0.9.1*: added support for more texture formats, access to the raw texture objects

*v0.9.0*: initial motion blur, and inital elimination of ll layer

- Major code re-org: eliminated most of ll layer, includign most of
  ll::Device and virutally all of ll::DeviceGroup; in new design
  Device will only contain device context, and all object-specific
  stuff will live in the respective api::Group, api::Geom etc
  classes. Device-specific data for a given object is handled by this
  object itself.

- initial support for motion blur on triangle meshes, by specifying
  two vertex arrays

- initial support of motion blur on instances, by specifyign two sets
  of transforms.

- new api fct owlMotionBlurEnable() to enable support for motion blur.

- groups and geoms now have methods to compute their world-space
  bounding boxes, as required for instance motion blur. These will get
  called/evaluated/used only hwne motion blur is enabled.

- moved some files from .cpp to .cu; to allow calling device kernels
  for computing bboxes.

- initial support for optix 7.1; code will automatically detect
  version and compile to proper version where they differ.

- owlLaunch2D now synchronous, async version explicitly owlLaunhc2DAsync

- added owlMissProgSet(context,rayType,missProg) to set miss program
  for a given ray type

v0.8.x - Revamped build system, owl viewer, interative samples, and textures
----------------------------------------------------------------------

*v0.8.3*: fixes, github issues, and naming

- added OWL_CHAR and OWL_UCHAR types

- renamed: owlParamsLaunch2D -> owlLaunch2D (added to
  owl_deprecated.h, and also axed lloLaunch function for cleanups)

- renamed owlLaunchParams<XYZ> -> owlParams<XYZ> (create, set, vetvariable etc)

- added OWL_INVALID_TYPE

- added owlXyzSetPointer()

- variables can now have type OWL_BUFFER (not just BUFPTR), and will
  write a owl::device::Buffer type (with size, type, and data members)

- device buffers can now be created over OWL_BUFFER and OWL_TEXTURE types

- added int12-buffer-of-objects sample that shows/tests buffers of
  buffers, and buffers of textures (by creating a buffer of buffers of
  textures)

*v0.8.2*: double types, interactive sample

- added OWL_DOUBLE type for variables, and al owl3d, setVariable, etc

- added int11-rotationCubes sample that has NxMxK roating textured cubes

*v0.8.1*: first light of textures

- added basic infrastructure for textures: OWLTexture type,
  OWLTextureFormat and OWLTextureFilterMode enums, OWL_TEXTURE
  variable types, owlVariableSetTexture, owl<Type>SetTexture(),
  etc. Textures currently only working for OWL_TEXEL_FORMAT_RGBA8,
  OWL_TEXEL_FORMAT_RGBA32F, OWL_TEXEL_FORMAT_R8, OWL_TEXEL_FORMAT_R32F
  and with OWL_TEXTURE_FILTER_LINEAR.

- added int10-texturedTriangles that opens a window with a
  checkerboard-textured box.

*v0.8.0*: build system, glfw, and owl viewer

- cmake build system now cleaner, and can use entire owl dir as
  add_subdirectory in other projects; main owl variables
  (OWL_INCLUDES, OWL_VIEWER_LIBRARIES, etc) now get exported to
  whoever includes, thus allowing includee to use same build flags,
  proper set of libraries an dincludes, etc.

- existing (glut-based) viewerWidget got replaced with glfw based
  OWLViewer class. New class has cleaner setup code, and no longer
  requires installing glut binaries for windows build

- build system picks up glfw where available, and otherwise builds
  glfw from source (full glfw source in samples/3rdParty)

- owlViewer now handles frame buffer resize and display internally (no
  longer app's job to do that), and does so with cuda/gl resource
  sharing using managed mem frame buffer. App still writes render() method,
  but simply writes final pixels to viewer-handles frame buffer.

- added first two interactive samples, using owlViewer

- changed samples/advaned/optixCourse to use owlViewer - glut now
  completely eliminated from owl, and all samples use same viewer.


v0.7.x - Unifiction of ng and ll APIs into one single owl API
----------------------------------------------------------------------

*v0.7.4*: major cleanups of "low-level" and "api" layer abstractions

- 'll' and 'ng' layers mostly merged, at least from the API layer; led
  to significant reduction in duplicate code.

- merged in PRs to enable TBB on windows, and to add cuda grphics resource buffer

*v0.7.3*: performance "guiding"

- OWL no longer allows for tracing directly into BLASes ... this is
  highly discouraged in the driver, so better to just disallow it.

*v0.7.3*: bug hotfix

- hotfix for bug introduced when auto-freeing of device memory, which
  accidentally freed instance BVH whiel still in use.

- various windows fixes; in particular removing tbb by default (windows only)

*v0.7.2*: various feature extensions and bug fixes

- lots of additional use in owl prime, m-owl-ana, distributed renderer, etc.

- fixed memory leak in instance builder

- added multi-device sample (`s07-rtow-multiGPU`), including
  `samples/s07-rtow-multiGPU/README.md` with brief notes on how to do
  multi-GPU in owl

- added a lot more documentation to api functions (though much is still missing)

- added ManagedMemory buffer type

- added several sanity checks for group sizes, traversable graph
  depth, etc (checking w/ optix limits before trying to build)

- added AnyHit shader support

*v0.7.1*: bugfix release.

- added variable plumbing for missing unsigned and 64-bit types

- fixed race condition in creating/using api handles that caused
  random crashes when setting launch params from different threads

*v0.7.0*: merged ng and ll APIs into one single API

- now have a single owl library, a single header file, etc

- eliminated all old ll/ samples (they only confused users)

- sierpinski, rtow, and rtow-mixedGeom samples now in owl API

- eliminated compaction in user geom and instance groups (doesn't
  help, anyway, and now have lower peak memory)

- camke now defines (and all samples etc use) cmake variables for
  `OWL_INCLUDES` and `OWL_LIBRARIES`

- fixes for TBB; TBB now gets detected more automatically, and used
  if found, with fallback to serial implementation if not

- added array3D, various cleanups and exntensions to owl/common

- various bugfixes and sanity/range checks throughout


v0.6.x - Buffer updates, launch params, first interactive example, ...
----------------------------------------------------------------------

*v0.6.1*: cleanup/flesh-out of instance transform API

- added creation of instance groups to ng/owl API

- added ability to use different matrix memory layouts (row major, column major)

- on ll layer: axed 'transform' parameter in `InstanceGroupSetChild`;
  matrices should now get passed to dedicated
  `InstanceGroupSetTransform` function.

- fixed various bugs related to instancing

*v0.6.0*: Buffer updates, launch params, first interactive example, ...

- Added OWL-port of github.com/ingowald/optix7course siggraph course
  notes sample as a first 'advanced' sample that allows for
  interactive fly-throughs trhough a "real" model. See original course
  notes for instructions on how to use.

Process of adding this sample also required, among others, the following
feature changes/additions

- added support for resizing (`owlBufferResize`), destroying
  (`owlBufferDestroy`), and uploading data to (`owlBufferUpload`)
  buffers.

- added concept of LaunchParams (with variables, similar to geoms),
  both for creating (`owlLaunchParamsCreate`) and launching with
  launch params (`owlParamLaunch2D`)

- added support for *asynchronous* launches, where multiple launches
  can be in flight in parallel, using different streams. Wrote
  experimental "owl prime" project to debug, debug, and test this;
  project allow, for example, highly threaded cpu-side shading with
  owl-based, async GPU offload of the ray tracing (this project is not
  yet included).

- worked on better interoperability between owl and CUDA-based host
  app; app can query buffer device pointers, add can add raw data (eg,
  cuda textures) to objects, app can query streams used for async
  launches (eg to add async cudamemcpys to that same stream, sync
  itself w/ owl, etc) and fixed includes to allow mixing cuda and
  optix code

- added ability to create user-type variables (to pass, e.g.,
  CUDA texture objects as parameters to meshes)

- rewritten interface for getting and setting variables using macros,
  all object and variable types now supported

- added support for more than one ray type (`owlContextSetRayTypeCount`)

- clamped down on verbosity of the logging - most logs now visible
  only in debug mode

- various minor bugfixes throughout the code


v0.5.x - First Public Release Cleanups
--------------------------------------

*v0.5.4*: First external windows-app

- various changes to cmake scripts, library names, and in partciualr
  owl/common/viewerWidget to remove roadblocks for windows apps using
  that infrastructure

- first external windows sandbox app (particle viewer) using owl/ng
  and owl/viewerWidget

*v0.5.3*: First *serious* node graph sample

- ported `ll05-rtow` sample to node graph api

- added bound program, user geom, user geom group, setprimcount and
  other missing functionality to node graph api

- `ng05-rtow` ported, working, and passing tests

*v0.5.2*: First (partial) node graph sample

- first working version of subset of node graph library (all that is
  required for 'firstTriangleMesh' example)

- `ng01-firstTriangleMesh` working

- significant renames and cleanups of owl/common (in particular, all
  'gdt::' and 'gdt/' merged into owl::common and owl/common)

- cleaned up owl/common/viewerWidget. Not used in owl itself (to avoid
  dependencies to glut etc), but now working successfully in first
  external test project

*v0.5.1*: First "c-api" version

- added public c-linkage api (in `include/owl/ll.h`)

- changed to build both static and dynamic/shared lib (tested working
  both linux and windows)

- ported all samples to this new api


*v0.5.0*: First public release

- first publicly accessible project on
  http://github.com/owl-project/owl

- major cleanups: "inlined" al the gdt submodule sources into
  owl/common to make owl external-dependency-fee. Feplaced gdt::
  namespace with owl::common:: to match.

v0.4.x - Instances
------------------

*v0.4.5*: `ll08-sierpinski` now uses path tracing

*v0.4.4*: multi-level instancing

- added new `DeviceGroup::setMaxInstancingDepth` that allows to set max
  instance depth and stack depth on pipeline.

- added `ll08-sierpinski` example that allows for testing user-supplied number
  of instance levels with a sierpinski pyramid (Thx Nate!)

*v0.4.3*: new api fcts to set transforms and children for instance groups

- added `instanceGroupSetChild` and `instanceGroupSetTransform`
- extended `ll07-groupOfGroups` by two test cases that set transforms

*v0.4.2*: bugfix - all samples working in multi-device again

*v0.4.1*: example `ll06-rtow-mixedGeometries.png`
 working w/ manual sucessive traced into two different accels

*v0.4.0*: new way of building SBT now based on groups

- api change: allocated geom groups now have their program size
  set in geomTypeCreate(), miss and raygen programs have it set in
  type rather than in sbt{raygen/miss}build (ie, program size now
  for all types set exactly once in type, then max size computed during
  sbt built)

- can handle more than one group; for non-0 group has to query
  geomGroupGetSbtOffset() and pass that value to trace

- new sbt structure no longer uses 'one entry per geom' (that unfortunately
  doesnt' work), but now builds sbt by iterating over all groups, and
  putting each groups' geom children in one block before putting
  next group. groups store the allcoated SBT offset for later use
  by instances

v0.3.x - User Geometries
------------------------

*v0.3.4*: bugfix: adding bounds prog broke bounds buffer variant. fixed.

*v0.3.4*: first 'serious' example: RTOW-finalChapter on OWL

- added `s05-rtow` example that runs Pete's "final chapter" example
  (iterative version) on top of OWL, with multi-device, different material, etc.

*v0.3.3*: major bugfix in bounds program for geoms w/ more than 128 prims.

*v0.3.2*: added two explicit examples for uesr geom - one with
  host-generation of bounds passed thrugh buffer, and one with bounds
  program

*v0.3.1*: First draft of *device-side* user prim bounds generation

- added `groupBuildPrimitiveBounds` function that builds, for a
  user geom group, all the the primbounds required for the respective
  user geoms and prims in that group. The input for the user geoms'
  bounding bxo functions is generated using same callback mechanism
  as sbt writing.

*v0.3.0*: First example of user geometry working

- can create user geometries through `createUserGeom`, and set
  type's isec program through `setGeomTypeIntersect`
- supports passing of new `userGeomSetBoundsBuffer` fct to pass user
  geoms through a buffer
- first example (8 sphere geometries, each with one sphere per geom)
  available as `s03-userGeometry`

v0.2.x
------

*v0.2.1*: multiple triangle meshes working
- multiple triangle meshes in same group debugged and working
- added `ll02-multipleTriangleGroups` sample that generates 8 boxes

*v0.2.0*: first triangle mesh with trace and SBT data working
- finalized `llTest` sample that ray traced image of one (tessellated) box

v0.1.x
------

- first version that does "some" sort of launch with mostly functional SBT

Contributors
============

- Ingo Wald
- Nate Morrical
- Eric Haines
