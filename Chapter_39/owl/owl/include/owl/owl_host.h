// ======================================================================== //
// Copyright 2019-2021 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include <cuda.h>
#include <driver_types.h>
#include <optix.h>

#include <sys/types.h>
#include <stdint.h>

#ifdef __cplusplus
# include <cstddef> 
#endif


#if defined(_MSC_VER)
#  define OWL_DLL_EXPORT __declspec(dllexport)
#  define OWL_DLL_IMPORT __declspec(dllimport)
#elif defined(__clang__) || defined(__GNUC__)
#  define OWL_DLL_EXPORT __attribute__((visibility("default")))
#  define OWL_DLL_IMPORT __attribute__((visibility("default")))
#else
#  define OWL_DLL_EXPORT
#  define OWL_DLL_IMPORT
#endif

#ifdef __cplusplus
# define OWL_IF_CPP(a) a
#else
# define OWL_IF_CPP(a) /* drop it */
#endif

//#if defined(OWL_DLL_INTERFACE)
//#  ifdef owl_EXPORTS
//#    define OWL_API OWL_DLL_EXPORT
//#  else
//#    define OWL_API OWL_DLL_IMPORT
//#  endif
//#else
#  ifdef __cplusplus
#    define OWL_API extern "C" OWL_DLL_EXPORT
#  else
#    define OWL_API /* bla */
#  endif
//#  define OWL_API /*static lib*/
//#endif
//#ifdef __cplusplus
//# define OWL_API extern "C" OWL_DLL_EXPORT
//#else
//# define OWL_API /* bla */
//#endif



#define OWL_OFFSETOF(type,member)                       \
   (uint32_t)((char *)(&((struct type *)0)-> member )   \
   -                                                    \
   (char *)(((struct type *)0)))
  
  
/*! enum that specifies the different possible memory layouts for
  passing transformation matrices */
typedef enum
  {
   /*! 4x3-float column-major matrix format, where a matrix is
     specified through four vec3fs, the first three being the basis
     vectors of the linear transform, and the fourth one the
     translation part. This is exactly the same layout as used in
     owl::common::affine3f (owl/common/math/AffineSpae.h) */
   OWL_MATRIX_FORMAT_COLUMN_MAJOR=0,
   
   /*! just another name for OWL_MATRIX_FORMAT_4X3_COLUMN_MAJOR that
     is easier to type - the "_OWL" indicates that this is the default
     format in the owl::common namespace */
   OWL_MATRIX_FORMAT_OWL=OWL_MATRIX_FORMAT_COLUMN_MAJOR,
   
   /*! 3x4-float *row-major* layout as preferred by optix; in this
     case it doesn't matter if it's a 4x3 or 4x4 matrix, since the
     last row in a 4x4 row major matrix can simply be ignored */
   OWL_MATRIX_FORMAT_ROW_MAJOR
  } OWLMatrixFormat;

typedef enum
  {
   OWL_SBT_HITGROUPS = 0x1,
   OWL_SBT_GEOMS     = OWL_SBT_HITGROUPS,
   OWL_SBT_RAYGENS   = 0x2,
   OWL_SBT_MISSPROGS = 0x4,
   OWL_SBT_ALL   = 0x7
  } OWLBuildSBTFlags;
  
typedef enum
  {
   OWL_INVALID_TYPE = 0,

   OWL_BUFFER=10,
   /*! a 64-bit int representing the number of elemnets in a buffer */
   OWL_BUFFER_SIZE,
   OWL_BUFFER_ID,
   OWL_BUFFER_POINTER,
   OWL_BUFPTR=OWL_BUFFER_POINTER,

   OWL_GROUP=20,

   /*! implicit variable of type integer that specifies the *index*
     of the given device. this variable type is implicit in the
     sense that it only gets _declared_ on the host, and gets set
     automatically during SBT creation */
   OWL_DEVICE=30,

   /*! texture(s) */
   OWL_TEXTURE=40,
   OWL_TEXTURE_2D=OWL_TEXTURE,


   /* all types that are naively copyable should be below this value,
      all that aren't should be above */
   _OWL_BEGIN_COPYABLE_TYPES = 1000,
   
   
   OWL_FLOAT=1000,
   OWL_FLOAT2,
   OWL_FLOAT3,
   OWL_FLOAT4,

   OWL_INT=1010,
   OWL_INT2,
   OWL_INT3,
   OWL_INT4,
   
   OWL_UINT=1020,
   OWL_UINT2,
   OWL_UINT3,
   OWL_UINT4,
   
   OWL_LONG=1030,
   OWL_LONG2,
   OWL_LONG3,
   OWL_LONG4,

   OWL_ULONG=1040,
   OWL_ULONG2,
   OWL_ULONG3,
   OWL_ULONG4,

   OWL_DOUBLE=1050,
   OWL_DOUBLE2,
   OWL_DOUBLE3,
   OWL_DOUBLE4,
    
   OWL_CHAR=1060,
   OWL_CHAR2,
   OWL_CHAR3,
   OWL_CHAR4,

   /*! unsigend 8-bit integer */
   OWL_UCHAR=1070,
   OWL_UCHAR2,
   OWL_UCHAR3,
   OWL_UCHAR4,

   OWL_SHORT=1080,
   OWL_SHORT2,
   OWL_SHORT3,
   OWL_SHORT4,

   /*! unsigend 8-bit integer */
   OWL_USHORT=1090,
   OWL_USHORT2,
   OWL_USHORT3,
   OWL_USHORT4,

   OWL_BOOL,
   OWL_BOOL2,
   OWL_BOOL3,
   OWL_BOOL4,
   
   /*! just another name for a 64-bit data type - unlike
     OWL_BUFFER_POINTER's (which gets translated from OWLBuffer's
     to actual device-side poiners) these OWL_RAW_POINTER types get
     copied binary without any translation. This is useful for
     owl-cuda interaction (where the user already has device
     pointers), but should not be used for logical buffers */
   OWL_RAW_POINTER=OWL_ULONG,
   OWL_BYTE = OWL_UCHAR,
   // OWL_BOOL = OWL_UCHAR,
   // OWL_BOOL2 = OWL_UCHAR2,
   // OWL_BOOL3 = OWL_UCHAR3,
   // OWL_BOOL4 = OWL_UCHAR4,


   /* matrix formats */
   OWL_AFFINE3F=1300,

   /*! at least for now, use that for buffers with user-defined types:
     type then is "OWL_USER_TYPE_BEGIN+sizeof(elementtype). Note
     that since we always _add_ the user type's size to this value
     this MUST be the last entry in the enum */
   OWL_USER_TYPE_BEGIN=10000
  }
  OWLDataType;

#define OWL_USER_TYPE(userType) ((OWLDataType)(OWL_USER_TYPE_BEGIN+sizeof(userType)))

typedef enum
  {
   // soon to be deprecated old naming
   OWL_GEOMETRY_USER,
   // new naming, to be consistent with type OLWGeom (not OWLGeometry):
   OWL_GEOM_USER=OWL_GEOMETRY_USER,
   // soon to be deprecated old naming
   OWL_GEOMETRY_TRIANGLES,
   // new naming, to be consistent with type OLWGeom (not OWLGeometry):
   OWL_GEOM_TRIANGLES=OWL_GEOMETRY_TRIANGLES,
   OWL_TRIANGLES=OWL_GEOMETRY_TRIANGLES,
   OWL_GEOMETRY_HAIR
  }
  OWLGeomKind;

#define OWL_ALL_RAY_TYPES -1


typedef float    OWL_float;
typedef double   OWL_double;
typedef int32_t  OWL_int;
typedef uint32_t OWL_uint;
typedef int64_t  OWL_long;
typedef uint64_t OWL_ulong;

typedef struct _OWL_int2    { int32_t  x,y; } owl2i;
typedef struct _OWL_uint2   { int32_t  x,y; } owl2ui;
typedef struct _OWL_long2   { int64_t  x,y; } owl2l;
typedef struct _OWL_ulong2  { uint64_t x,y; } owl2ul;
typedef struct _OWL_float2  { float    x,y; } owl2f;
typedef struct _OWL_double2 { double   x,y; } owl2d;

typedef struct _OWL_int3    { int32_t  x,y,z; } owl3i;
typedef struct _OWL_uint3   { uint32_t x,y,z; } owl3ui;
typedef struct _OWL_long3   { int64_t  x,y,z; } owl3l;
typedef struct _OWL_ulong3  { uint64_t x,y,z; } owl3ul;
typedef struct _OWL_float3  { float    x,y,z; } owl3f;
typedef struct _OWL_double3 { double   x,y,z; } owl3d;

typedef struct _OWL_int4    { int32_t  x,y,z,w; } owl4i;
typedef struct _OWL_uint4   { uint32_t x,y,z,w; } owl4ui;
typedef struct _OWL_long4   { int64_t  x,y,z,w; } owl4l;
typedef struct _OWL_ulong4  { uint64_t x,y,z,w; } owl4ul;
typedef struct _OWL_float4  { float    x,y,z,w; } owl4f;
typedef struct _OWL_double4 { double   x,y,z,w; } owl4d;

typedef struct _OWL_affine3f { owl3f vx,vy,vz,t; } owl4x3f;

typedef struct _OWLVarDecl {
  const char *name;
  OWLDataType type;
  uint32_t    offset;
} OWLVarDecl;

typedef struct _OWLBoundValueDecl {
  OWLVarDecl var;
  void *boundValuePtr;
} OWLBoundValueDecl;


/*! supported formats for texels in textures */
typedef enum {
  OWL_TEXEL_FORMAT_RGBA8,
  OWL_TEXEL_FORMAT_RGBA32F,
  OWL_TEXEL_FORMAT_R8,
  OWL_TEXEL_FORMAT_R32F
}
OWLTexelFormat;

/*! currently supported texture filter modes */
typedef enum {
  OWL_TEXTURE_NEAREST,
  OWL_TEXTURE_LINEAR
}
OWLTextureFilterMode;

/*! currently supported texture filter modes */
typedef enum {
  OWL_TEXTURE_WRAP,
  OWL_TEXTURE_CLAMP,
  OWL_TEXTURE_BORDER,
  OWL_TEXTURE_MIRROR
}
OWLTextureAddressMode;

/*! Indicates if a texture is linear or SRGB */
typedef enum {
  OWL_COLOR_SPACE_LINEAR,
  OWL_COLOR_SPACE_SRGB
}
OWLTextureColorSpace;

// ------------------------------------------------------------------
// device-objects - size of those _HAS_ to match the device-side
// definition of these types
// ------------------------------------------------------------------
typedef OptixTraversableHandle OWLDeviceTraversable;
typedef struct _OWLDeviceBuffer2D { void *d_pointer; owl2i dims; } OWLDeviceBuffer2D;


typedef struct _OWLContext       *OWLContext;
typedef struct _OWLBuffer        *OWLBuffer;
typedef struct _OWLTexture       *OWLTexture;
typedef struct _OWLGeom          *OWLGeom;
typedef struct _OWLGeomType      *OWLGeomType;
typedef struct _OWLVariable      *OWLVariable;
typedef struct _OWLModule        *OWLModule;
typedef struct _OWLGroup         *OWLGroup;
typedef struct _OWLRayGen        *OWLRayGen;
typedef struct _OWLMissProg      *OWLMissProg;
/*! launch params (or "globals") are variables that can be put into
  device constant memory, accessible through a CUDA "__constant__
  <Type> optixLaunchParams;" variable on the device side. Launch
  params capture the layout of this struct, and the value of its
  members, on the host side, then properly fill it in before executing
  a launch. OptiX calls those "launch parameters" because they are
  similar to how parameters to a CUDA kernel are internally treated;
  we also call them "globals" because they are globally accessible to
  all programs within a given launch */
typedef struct _OWLLaunchParams  *OWLLaunchParams, *OWLParams, *OWLGlobals;

OWL_API void owlBuildPrograms(OWLContext context);
OWL_API void owlBuildPipeline(OWLContext context);
OWL_API void owlBuildSBT(OWLContext context,
                         OWLBuildSBTFlags flags OWL_IF_CPP(=OWL_SBT_ALL));

/*! returns number of devices available in the given context */
OWL_API int32_t
owlGetDeviceCount(OWLContext context);

/*! creates a new device context with the gives list of devices. 

  If requested device IDs list if null it implicitly refers to the
  list "0,1,2,...."; if numDevices <= 0 it automatically refers to
  "all devices you can find". Examples:

  - owlContextCreate(nullptr,1) creates one device on the first GPU

  - owlContextCreate(nullptr,0) creates a context across all GPUs in
  the system

  - int gpu=2;owlContextCreate(&gpu,1) will create a context on GPU #2
  (where 2 refers to the CUDA device ordinal; from that point on, from
  owl's standpoint (eg, during owlBufferGetPointer() this GPU will
  from that point on be known as device #0 */
OWL_API OWLContext
owlContextCreate(int32_t *requestedDeviceIDs OWL_IF_CPP(=nullptr),
                 int numDevices OWL_IF_CPP(=0));
                 
OWL_API void
owlContextDestroy(OWLContext context);

/*! enable motion blur for this context. this _has_ to be called
    before creating any geometries, groups, etc, and before the
    pipeline gets compiled. Ie, it shold be called _right_ after
    context creation */
OWL_API void
owlEnableMotionBlur(OWLContext _context);

/*! set number of ray types to be used in this context; this should be
  done before any programs, pipelines, geometries, etc get
  created */
OWL_API void
owlContextSetRayTypeCount(OWLContext context,
                          size_t numRayTypes);

/* Set number of attributes for passing data from custom Intersection programs
   to ClosestHit programs.  Default 2.  Has no effect once programs are built.*/
OWL_API void
owlContextSetNumAttributeValues(OWLContext context,
                                size_t numAttributeValues);

/*! tells OptiX to specialize the values of certain launch parameters
  when compiling modules, and ignore their values at launch.
  See section 6.3.1 of the OptiX 7.2 programming guide.
  This call is a no-op for OptiX versions < 7.2, and programs should not 
  rely on it for correct behavior.
  OWL stores a copy of the array internally. */
OWL_API void
owlContextSetBoundLaunchParamValues(OWLContext context,
                                    const OWLBoundValueDecl *boundValues,
                                    int numBoundValues);

/*! sets maximum instancing depth for the given context:

  '0' means 'no instancing allowed, only bottom-level accels; Note
  this mode isn't actually allowed in OWL right now, as the most
  convenient way of realizing it is actually *slower* than simply
  putting a single "dummy" instance (with just this one child, and a
  identify transform) over each blas.

  '1' means 'at most one layer of instances' (ie, a two-level scene),
  where the 'root' world rays are traced against can be an instance
  group, but every child in that inscne group is a geometry group.

  'N>1" means "up to N layers of instances are allowed.

  The default instancing depth is 1 (ie, a two-level scene), since
  this allows for most use cases of instancing and is still
  hardware-accelerated. Using a node graph with instancing deeper than
  the configured value will result in wrong results; but be aware that
  using any value > 1 here will come with a cost. It is recommended
  to, if at all possible, leave this value to one and convert the
  input scene to a two-level scene layout (ie, with only one level of
  instances) */
OWL_API void
owlSetMaxInstancingDepth(OWLContext context,
                         int32_t maxInstanceDepth);

/* return the cuda stream associated with the given device. */
OWL_API CUstream
owlContextGetStream(OWLContext context, int deviceID);

/* return the optix context associated with the given device. */
OWL_API OptixDeviceContext
owlContextGetOptixContext(OWLContext context, int deviceID);

OWL_API OWLModule
owlModuleCreate(OWLContext  context,
                const char *ptxCode);
                
OWL_API void
owlModuleRelease(OWLModule module);

OWL_API OWLGeom
owlGeomCreate(OWLContext  context,
              OWLGeomType type);
              
OWL_API void 
owlGeomRelease(OWLGeom geometry);

OWL_API OWLParams
owlParamsCreate(OWLContext  context,
                size_t      sizeOfVarStruct,
                OWLVarDecl *vars,
                int         numVars);

OWL_API OWLRayGen
owlRayGenCreate(OWLContext  context,
                OWLModule   module,
                const char *programName,
                size_t      sizeOfVarStruct,
                OWLVarDecl *vars,
                int         numVars);
                
OWL_API void 
owlRayGenRelease(OWLRayGen rayGen);

/*! creates a miss program with given function name (in given module)
    and given variables. Note due to backwards compatibility this will
    also automatically *set*, by default, the first such created
    program as miss program for ray type number 0, the second one for
    ray type number 1, etc. If another order is desired, you can use
    \see owlMissProgSet to explicitly assign miss programs to specific
    ray types */
OWL_API OWLMissProg
owlMissProgCreate(OWLContext  context,
                  OWLModule   module,
                  const char *programName,
                  size_t      sizeOfVarStruct,
                  OWLVarDecl *vars,
                  int         numVars);

/*! sets the given miss program for the given ray type */
OWL_API void
owlMissProgSet(OWLContext  context,
               int rayType,
               OWLMissProg missProgToUse);

// ------------------------------------------------------------------
/*! create a new group (which handles the acceleration strucure) for
  triangle geometries.

  \param numGeometries Number of geometries in this group, must be
  non-zero.

  \param arrayOfChildGeoms A array of 'numGeomteries' child
  geometries. Every geom in this array must be a valid owl geometry
  created with owlGeomCreate, and must be of a OWL_GEOM_USER
  type.

  \param buildFlags A combination of OptixBuildFlags.  The default
  of 0 means to use OWL default build flags.
*/
OWL_API OWLGroup
owlUserGeomGroupCreate(OWLContext context,
                       size_t     numGeometries,
                       OWLGeom   *arrayOfChildGeoms,
                       unsigned int buildFlags OWL_IF_CPP(=0));


// ------------------------------------------------------------------
/*! create a new group (which handles the acceleration strucure) for
  triangle geometries.

  \param numGeometries Number of geometries in this group, must be
  non-zero.

  \param arrayOfChildGeoms A array of 'numGeometries' child
  geometries. Every geom in this array must be a valid owl geometry
  created with owlGeomCreate, and must be of a OWL_GEOM_TRIANGLES
  type.

  \param buildFlags A combination of OptixBuildFlags.  The default
  of 0 means to use OWL default build flags.
*/
OWL_API OWLGroup
owlTrianglesGeomGroupCreate(OWLContext context,
                            size_t     numGeometries,
                            OWLGeom   *initValues,
                            unsigned int buildFlags OWL_IF_CPP(=0));

// ------------------------------------------------------------------
/*! create a new instance group with given number of instances. The
  child groups and their instance IDs and/or transforms can either
  be specified "in bulk" as part of this call, or can be set later on
  with individual calls to \see owlInstanceGroupSetChild and \see
  owlInstanceGroupSetTransform. Note however, that in the case of
  having millions of instances in a group it will be *much* more
  efficient to set them in bulk open creation, than in millions of
  inidiviual API calls.

  Either or all of initGroups, initTranforms, or initInstanceIDs may
  be null, in which case the values used for the 'th child will be an
  uninitialized (invalid) group, a unit transform, and 'i', respectively.
  If initGroups was null, make sure to set all of the child groups
  with \see owlInstanceGroupSetChild before using this group, or
  it will crash.
*/
OWL_API OWLGroup
owlInstanceGroupCreate(OWLContext context,
                       
                       /*! number of instances in this group */
                       size_t     numInstances,
                       
                       /*! the initial list of owl groups to use by
                         the instances in this group; must be either
                         null, or an array of the size
                         'numInstances', the i'th instance in this
                         group will be an instance of the i'th
                         element in this list. If null, you must
                         set all the children individually before
                         using this group. */
                       const OWLGroup *initGroups      OWL_IF_CPP(= nullptr),

                       /*! instance IDs to use for the instance in
                         this group; must be eithe rnull, or an
                         array of size numInstnaces. If null, the
                         i'th child of this instance group will use
                         instanceID=i, otherwise, it will use the
                         user-provided instnace ID from this
                         list. Specifying an instanceID will affect
                         what value 'optixGetInstanceID' will return
                         in a CH program that refers to the given
                         instance */
                       const uint32_t *initInstanceIDs OWL_IF_CPP(= nullptr),
                       
                       /*! initial list of transforms that this
                         instance group will use; must be either
                         null, or an array of size numInstnaces, of
                         the format specified */
                       const float    *initTransforms  OWL_IF_CPP(= nullptr),
                       OWLMatrixFormat matrixFormat    OWL_IF_CPP(=OWL_MATRIX_FORMAT_OWL),

                       /*! A combination of OptixBuildFlags.  The default
                         of 0 means to use OWL default build flags.*/
                       unsigned int buildFlags OWL_IF_CPP(=0)
                       );

                       
OWL_API void
owlGroupRelease(OWLGroup group);

OWL_API void owlGroupBuildAccel(OWLGroup group);
OWL_API void owlGroupRefitAccel(OWLGroup group);

/*! returns the (device) memory used for this group's acceleration
    structure (but _excluding_ the memory for the geometries
    itself). "memFinal" is how much memory is used for the _final_
    version of the BVH (after it is done building), "memPeak" is peak
    memory used during construction. passing a NULL pointer to any
    value is valid; these values will get ignored. */
OWL_API void
owlGroupGetAccelSize(OWLGroup group,
                     size_t *p_memFinal,
                     size_t *p_memPeak);
                                  
OWL_API OWLGeomType
owlGeomTypeCreate(OWLContext context,
                  OWLGeomKind kind,
                  size_t sizeOfVarStruct,
                  OWLVarDecl *vars,
                  int         numVars);


/*! create new texture of given format and dimensions - for now, we
  only do "wrap" textures, and eithe rbilinear or nearest filter;
  once we allow for doing things like texture borders we'll have to
  change this api */
OWL_API OWLTexture
owlTexture2DCreate(OWLContext context,
                   OWLTexelFormat texelFormat,
                   /*! number of texels in x dimension */
                   uint32_t size_x,
                   /*! number of texels in y dimension */
                   uint32_t size_y,
                   const void *texels,
                   OWLTextureFilterMode filterMode OWL_IF_CPP(=OWL_TEXTURE_LINEAR),
                   OWLTextureAddressMode addressMode OWL_IF_CPP(=OWL_TEXTURE_CLAMP),
                   OWLTextureColorSpace colorSpace OWL_IF_CPP(=OWL_COLOR_SPACE_LINEAR),
                   /*! number of bytes between one line of texels and
                     the next; '0' means 'size_x * sizeof(texel)' */
                   uint32_t linePitchInBytes       OWL_IF_CPP(=0)
                   );
                   
/*! destroy the given texture; after this call any accesses to the 
   given texture are invalid */
OWL_API void
owlTexture2DDestroy(OWLTexture texture);

/*! returns the device handle of the given texture for the given
    device ID. Useful for custom texture object arrays. */
OWL_API CUtexObject
owlTextureGetObject(OWLTexture texture, int deviceID);

/*! creates a device buffer where every device has its own local copy
  of the given buffer */
OWL_API OWLBuffer
owlDeviceBufferCreate(OWLContext  context,
                      OWLDataType type,
                      size_t      count,
                      const void *init);

/*! creates a buffer that uses CUDA host pinned memory; that memory is
  pinned on the host and accessive to all devices in the deviec
  group */
OWL_API OWLBuffer
owlHostPinnedBufferCreate(OWLContext context,
                          OWLDataType type,
                          size_t      count);

/*! creates a buffer that uses CUDA managed memory; that memory is
  managed by CUDA (see CUDAs documentatoin on managed memory) and
  accessive to all devices in the deviec group */
OWL_API OWLBuffer
owlManagedMemoryBufferCreate(OWLContext context,
                             OWLDataType type,
                             size_t      count,
                             const void *init);

/*! creates a buffer wrapping a CUDA graphics resource;
  the resource must be created and registered by the user */
OWL_API OWLBuffer
owlGraphicsBufferCreate(OWLContext             context,
                        OWLDataType            type,
                        size_t                 count,
                        cudaGraphicsResource_t resource);

/*! OWL objects are reference-counted. This will release the
  reference to the buffer, and free it if it was the last
  reference. */
OWL_API void
owlBufferRelease(OWLBuffer buffer);

/*! destroy the given buffer; this will both release the app's
  refcount on the given buffer handle, *and* the buffer itself; ie,
  even if some objects still hold variables that refer to the old
  handle the buffer itself will be freed */
OWL_API void 
owlBufferDestroy(OWLBuffer buffer);

OWL_API void
owlGraphicsBufferMap(OWLBuffer buffer);

OWL_API void
owlGraphicsBufferUnmap(OWLBuffer buffer);

/*! returns the device pointer of the given pointer for the given
  device ID. For host-pinned or managed memory buffers (where the
  buffer is shared across all devices) this pointer should be the
  same across all devices (and even be accessible on the host); for
  device buffers each device *may* see this buffer under a different
  address, and that address is not valid on the host. Note this
  function is paricuarly useful for CUDA-interop; allowing to
  cudaMemcpy to/from an owl buffer directly from CUDA code */
OWL_API const void *
owlBufferGetPointer(OWLBuffer buffer, int deviceID);

OWL_API OptixTraversableHandle 
owlGroupGetTraversable(OWLGroup group, int deviceID);

OWL_API void 
owlBufferResize(OWLBuffer buffer, size_t newItemCount);

OWL_API size_t
owlBufferSizeInBytes(OWLBuffer buffer);

/*! uploads data from given host poiner to given device. offset refers
    to the offset (in bytes) on the device. \param numbytes is the
    number of bytes to upload; -1 meaning "full buffer" */
OWL_API void 
owlBufferUpload(OWLBuffer buffer,
                const void *hostPtr,
                size_t offset OWL_IF_CPP(=0),
                size_t numBytes OWL_IF_CPP(=size_t(-1)));

/*! clears a buffer in the sense that it sets the entire memory region
    to zeroes. Note this is currently implemneted only for buffers of
    copyable data (ie, not buffers of objects). */
OWL_API void 
owlBufferClear(OWLBuffer buffer);

/*! executes an optix launch of given size, with given launch
  program. Note this is asynchronous, and may _not_ be
  completed by the time this function returns. */
OWL_API void
owlRayGenLaunch2D(OWLRayGen rayGen, int dims_x, int dims_y);

/*! 3D-launch variant of \see owlRayGenLaunch2D */
OWL_API void
owlRayGenLaunch3D(OWLRayGen rayGen, int dims_x, int dims_y, int dims_z);

/*! perform a raygen launch with launch parameters, in a *synchronous*
    way; it, by the time this function returns the launch is
    completed. Both rayGen and params must be valid handles; it is
    valid to have a empty params, but it may not be null */
OWL_API void
owlLaunch2D(OWLRayGen rayGen, int dims_x, int dims_y,
            OWLParams params);

/*! 3D launch variant of owlLaunch2D */
OWL_API void
owlLaunch3D(OWLRayGen rayGen, int dims_x, int dims_y, int dims_z,
            OWLParams params);

/*! perform a raygen launch with launch parameters, in a *A*synchronous
    way; it, this will only launch, but *NOT* wait for completion (see
    owlLaunchSync). Both rayGen and params must be valid handles; it is
    valid to have a empty params, but it may not be null */
OWL_API void
owlAsyncLaunch2D(OWLRayGen rayGen, int dims_x, int dims_y,
                 OWLParams params);

/*! 3D-launch equivalent of \see owlAsyncLaunch2D */
OWL_API void
owlAsyncLaunch3D(OWLRayGen rayGen, int dims_x, int dims_y, int dims_z,
                 OWLParams params);

/*! perform a raygen launch with launch parameters, but only for a given device, 
    and in an asynchronous way. This function is useful for dynamic load balancing. */
OWL_API void
owlAsyncLaunch2DOnDevice(OWLRayGen rayGen, int dims_x, int dims_y, 
                        int deviceID, OWLParams params);


OWL_API CUstream
owlParamsGetCudaStream(OWLParams params, int deviceID);

/*! wait for the async launch to finish */
OWL_API void
owlLaunchSync(OWLParams params);

// ==================================================================
// "Triangles" functions
// ==================================================================
OWL_API void owlTrianglesSetVertices(OWLGeom triangles,
                                     OWLBuffer vertices,
                                     size_t count,
                                     size_t stride,
                                     size_t offset);
OWL_API void owlTrianglesSetMotionVertices(OWLGeom triangles,
                                           /*! number of vertex arrays
                                               passed here, the first
                                               of those is for t=0,
                                               thelast for t=1,
                                               everything is linearly
                                               interpolated
                                               in-between */
                                           size_t    numKeys,
                                           OWLBuffer *vertexArrays,
                                           size_t count,
                                           size_t stride,
                                           size_t offset);
OWL_API void owlTrianglesSetIndices(OWLGeom triangles,
                                    OWLBuffer indices,
                                    size_t count,
                                    size_t stride,
                                    size_t offset);

// -------------------------------------------------------
// group/hierarchy creation and setting
// -------------------------------------------------------
OWL_API void
owlInstanceGroupSetChild(OWLGroup group,
                         int whichChild,
                         OWLGroup child);

/*! sets the transformatoin matrix to be applied to the childID'th
  child of the given instance group */
OWL_API void
owlInstanceGroupSetTransform(OWLGroup group,
                             int whichChild,
                             const float *floats,
                             OWLMatrixFormat matrixFormat
                             OWL_IF_CPP(=OWL_MATRIX_FORMAT_OWL));

/*! this function allows to set up to N different arrays of trnsforms
    for motion blur; the first such array is used as transforms for
    t=0, the last one for t=1.  */
OWL_API void
owlInstanceGroupSetTransforms(OWLGroup group,
                              /*! whether to set for t=0 or t=1 -
                                  currently supporting only 0 or 1*/
                              uint32_t timeStep,
                              const float *floatsForThisStimeStep,
                              OWLMatrixFormat matrixFormat
                              OWL_IF_CPP(=OWL_MATRIX_FORMAT_OWL));

/*! sets the list of IDs to use for the child instnaces. By default
    the instance ID of child #i is simply i, but optix allows to
    specify a user-defined instnace ID for each instance, which with
    owl can be done through this array. Array size must match number
    of instances in the specified group */
OWL_API void
owlInstanceGroupSetInstanceIDs(OWLGroup group,
                               const uint32_t *instanceIDs);

OWL_API void
owlInstanceGroupSetVisibilityMasks(OWLGroup group,
                               const uint8_t *visibilityMasks);


OWL_API void
owlGeomTypeSetClosestHit(OWLGeomType type,
                         int rayType,
                         OWLModule module,
                         const char *progName);

OWL_API void
owlGeomTypeSetAnyHit(OWLGeomType type,
                     int rayType,
                     OWLModule module,
                     const char *progName);

OWL_API void
owlGeomTypeSetIntersectProg(OWLGeomType type,
                            int rayType,
                            OWLModule module,
                            const char *progName);

OWL_API void
owlGeomTypeSetBoundsProg(OWLGeomType type,
                         OWLModule module,
                         const char *progName);

/*! set the primitive count for the given uesr geometry. this _has_ to
  be set before the group(s) that this geom is used in get built */
OWL_API void
owlGeomSetPrimCount(OWLGeom geom,
                    size_t  primCount);

// -------------------------------------------------------
// VariableGet for the various types
// -------------------------------------------------------
OWL_API OWLVariable
owlGeomGetVariable(OWLGeom geom,
                   const char *varName);

OWL_API OWLVariable
owlRayGenGetVariable(OWLRayGen geom,
                     const char *varName);

OWL_API OWLVariable
owlMissProgGetVariable(OWLMissProg geom,
                       const char *varName);

OWL_API OWLVariable
owlParamsGetVariable(OWLParams object,
                     const char *varName);
                     
OWL_API void
owlVariableRelease(OWLVariable variable);

// -------------------------------------------------------
// VariableSet for different variable types
// -------------------------------------------------------

// setters for bools (c++ only)
#ifdef __cplusplus
OWL_API void owlVariableSet1b(OWLVariable var, bool val);
OWL_API void owlVariableSet2b(OWLVariable var, bool x, bool y);
OWL_API void owlVariableSet3b(OWLVariable var, bool x, bool y, bool z);
OWL_API void owlVariableSet4b(OWLVariable var, bool x, bool y, bool z, bool w);
OWL_API void owlVariableSet2bv(OWLVariable var, const bool *val);
OWL_API void owlVariableSet3bv(OWLVariable var, const bool *val);
OWL_API void owlVariableSet4bv(OWLVariable var, const bool *val);
#endif

// setters for 8-bit signed ints
OWL_API void owlVariableSet1c(OWLVariable var, int8_t val);
OWL_API void owlVariableSet2c(OWLVariable var, int8_t x, int8_t y);
OWL_API void owlVariableSet3c(OWLVariable var, int8_t x, int8_t y, int8_t z);
OWL_API void owlVariableSet4c(OWLVariable var, int8_t x, int8_t y, int8_t z, int8_t w);
OWL_API void owlVariableSet2cv(OWLVariable var, const int8_t *val);
OWL_API void owlVariableSet3cv(OWLVariable var, const int8_t *val);
OWL_API void owlVariableSet4cv(OWLVariable var, const int8_t *val);

// setters for 8-bit unsigned ints
OWL_API void owlVariableSet1uc(OWLVariable var, uint8_t val);
OWL_API void owlVariableSet2uc(OWLVariable var, uint8_t x, uint8_t y);
OWL_API void owlVariableSet3uc(OWLVariable var, uint8_t x, uint8_t y, uint8_t z);
OWL_API void owlVariableSet4uc(OWLVariable var, uint8_t x, uint8_t y, uint8_t z, uint8_t w);
OWL_API void owlVariableSet2ucv(OWLVariable var, const uint8_t *val);
OWL_API void owlVariableSet3ucv(OWLVariable var, const uint8_t *val);
OWL_API void owlVariableSet4ucv(OWLVariable var, const uint8_t *val);

// setters for 16-bit signed ints
OWL_API void owlVariableSet1s(OWLVariable var, int16_t val);
OWL_API void owlVariableSet2s(OWLVariable var, int16_t x, int16_t y);
OWL_API void owlVariableSet3s(OWLVariable var, int16_t x, int16_t y, int16_t z);
OWL_API void owlVariableSet4s(OWLVariable var, int16_t x, int16_t y, int16_t z, int16_t w);
OWL_API void owlVariableSet2sv(OWLVariable var, const int16_t *val);
OWL_API void owlVariableSet3sv(OWLVariable var, const int16_t *val);
OWL_API void owlVariableSet4sv(OWLVariable var, const int16_t *val);

// setters for 16-bit unsigned ints
OWL_API void owlVariableSet1us(OWLVariable var, uint16_t val);
OWL_API void owlVariableSet2us(OWLVariable var, uint16_t x, uint16_t y);
OWL_API void owlVariableSet3us(OWLVariable var, uint16_t x, uint16_t y, uint16_t z);
OWL_API void owlVariableSet4us(OWLVariable var, uint16_t x, uint16_t y, uint16_t z, uint16_t w);
OWL_API void owlVariableSet2usv(OWLVariable var, const uint16_t *val);
OWL_API void owlVariableSet3usv(OWLVariable var, const uint16_t *val);
OWL_API void owlVariableSet4usv(OWLVariable var, const uint16_t *val);

// setters for 32-bit floats
OWL_API void owlVariableSet1f(OWLVariable var, float val);
OWL_API void owlVariableSet2f(OWLVariable var, float x, float y);
OWL_API void owlVariableSet3f(OWLVariable var, float x, float y, float z);
OWL_API void owlVariableSet4f(OWLVariable var, float x, float y, float z, float w);
OWL_API void owlVariableSet2fv(OWLVariable var, const float *val);
OWL_API void owlVariableSet3fv(OWLVariable var, const float *val);
OWL_API void owlVariableSet4fv(OWLVariable var, const float *val);

// setters for 32-bit signed ints
OWL_API void owlVariableSet1i(OWLVariable var, int32_t val);
OWL_API void owlVariableSet2i(OWLVariable var, int32_t x, int32_t y);
OWL_API void owlVariableSet3i(OWLVariable var, int32_t x, int32_t y, int32_t z);
OWL_API void owlVariableSet4i(OWLVariable var, int32_t x, int32_t y, int32_t z, int32_t w);
OWL_API void owlVariableSet2iv(OWLVariable var, const int32_t *val);
OWL_API void owlVariableSet3iv(OWLVariable var, const int32_t *val);
OWL_API void owlVariableSet4iv(OWLVariable var, const int32_t *val);

// setters for 32-bit unsigned ints
OWL_API void owlVariableSet1ui(OWLVariable var, uint32_t val);
OWL_API void owlVariableSet2ui(OWLVariable var, uint32_t x, uint32_t y);
OWL_API void owlVariableSet3ui(OWLVariable var, uint32_t x, uint32_t y, uint32_t z);
OWL_API void owlVariableSet4ui(OWLVariable var, uint32_t x, uint32_t y, uint32_t z, uint32_t w);
OWL_API void owlVariableSet2uiv(OWLVariable var, const uint32_t *val);
OWL_API void owlVariableSet3uiv(OWLVariable var, const uint32_t *val);
OWL_API void owlVariableSet4uiv(OWLVariable var, const uint32_t *val);

// setters for 64-bit doubles
OWL_API void owlVariableSet1d(OWLVariable var, double val);
OWL_API void owlVariableSet2d(OWLVariable var, double x, double y);
OWL_API void owlVariableSet3d(OWLVariable var, double x, double y, double z);
OWL_API void owlVariableSet4d(OWLVariable var, double x, double y, double z, double w);
OWL_API void owlVariableSet2dv(OWLVariable var, const double *val);
OWL_API void owlVariableSet3dv(OWLVariable var, const double *val);
OWL_API void owlVariableSet4dv(OWLVariable var, const double *val);

// setters for 64-bit signed ints
OWL_API void owlVariableSet1l(OWLVariable var, int64_t val);
OWL_API void owlVariableSet2l(OWLVariable var, int64_t x, int64_t y);
OWL_API void owlVariableSet3l(OWLVariable var, int64_t x, int64_t y, int64_t z);
OWL_API void owlVariableSet4l(OWLVariable var, int64_t x, int64_t y, int64_t z, int64_t w);
OWL_API void owlVariableSet2lv(OWLVariable var, const int64_t *val);
OWL_API void owlVariableSet3lv(OWLVariable var, const int64_t *val);
OWL_API void owlVariableSet4lv(OWLVariable var, const int64_t *val);

// setters for 64-bit unsigned ints
OWL_API void owlVariableSet1ul(OWLVariable var, uint64_t val);
OWL_API void owlVariableSet2ul(OWLVariable var, uint64_t x, uint64_t y);
OWL_API void owlVariableSet3ul(OWLVariable var, uint64_t x, uint64_t y, uint64_t z);
OWL_API void owlVariableSet4ul(OWLVariable var, uint64_t x, uint64_t y, uint64_t z, uint64_t w);
OWL_API void owlVariableSet2ulv(OWLVariable var, const uint64_t *val);
OWL_API void owlVariableSet3ulv(OWLVariable var, const uint64_t *val);
OWL_API void owlVariableSet4ulv(OWLVariable var, const uint64_t *val);

// setters for 'meta' variables
OWL_API void owlVariableSetGroup  (OWLVariable variable, OWLGroup value);
OWL_API void owlVariableSetTexture(OWLVariable variable, OWLTexture value);
OWL_API void owlVariableSetBuffer (OWLVariable variable, OWLBuffer value);
OWL_API void owlVariableSetRaw    (OWLVariable variable, const void *valuePtr);
OWL_API void owlVariableSetPointer(OWLVariable variable, const void *valuePtr);

#ifdef __cplusplus
// ------------------------------------------------------------------
// setters for variables of type "bool" (bools only on c++)
// ------------------------------------------------------------------

// setters for variables on "RayGen"s
OWL_API void owlRayGenSet1b(OWLRayGen var, const char *name, bool val);
OWL_API void owlRayGenSet2b(OWLRayGen var, const char *name, bool x, bool y);
OWL_API void owlRayGenSet3b(OWLRayGen var, const char *name, bool x, bool y, bool z);
OWL_API void owlRayGenSet4b(OWLRayGen var, const char *name, bool x, bool y, bool z, bool w);
OWL_API void owlRayGenSet2bv(OWLRayGen var, const char *name, const bool *val);
OWL_API void owlRayGenSet3bv(OWLRayGen var, const char *name, const bool *val);
OWL_API void owlRayGenSet4bv(OWLRayGen var, const char *name, const bool *val);

// setters for variables on "MissProg"s
OWL_API void owlMissProgSet1b(OWLMissProg var, const char *name, bool val);
OWL_API void owlMissProgSet2b(OWLMissProg var, const char *name, bool x, bool y);
OWL_API void owlMissProgSet3b(OWLMissProg var, const char *name, bool x, bool y, bool z);
OWL_API void owlMissProgSet4b(OWLMissProg var, const char *name, bool x, bool y, bool z, bool w);
OWL_API void owlMissProgSet2bv(OWLMissProg var, const char *name, const bool *val);
OWL_API void owlMissProgSet3bv(OWLMissProg var, const char *name, const bool *val);
OWL_API void owlMissProgSet4bv(OWLMissProg var, const char *name, const bool *val);

// setters for variables on "Geom"s
OWL_API void owlGeomSet1b(OWLGeom var, const char *name, bool val);
OWL_API void owlGeomSet2b(OWLGeom var, const char *name, bool x, bool y);
OWL_API void owlGeomSet3b(OWLGeom var, const char *name, bool x, bool y, bool z);
OWL_API void owlGeomSet4b(OWLGeom var, const char *name, bool x, bool y, bool z, bool w);
OWL_API void owlGeomSet2bv(OWLGeom var, const char *name, const bool *val);
OWL_API void owlGeomSet3bv(OWLGeom var, const char *name, const bool *val);
OWL_API void owlGeomSet4bv(OWLGeom var, const char *name, const bool *val);

// setters for variables on "Params"s
OWL_API void owlParamsSet1b(OWLParams var, const char *name, bool val);
OWL_API void owlParamsSet2b(OWLParams var, const char *name, bool x, bool y);
OWL_API void owlParamsSet3b(OWLParams var, const char *name, bool x, bool y, bool z);
OWL_API void owlParamsSet4b(OWLParams var, const char *name, bool x, bool y, bool z, bool w);
OWL_API void owlParamsSet2bv(OWLParams var, const char *name, const bool *val);
OWL_API void owlParamsSet3bv(OWLParams var, const char *name, const bool *val);
OWL_API void owlParamsSet4bv(OWLParams var, const char *name, const bool *val);
#endif

// ------------------------------------------------------------------
// setters for variables of type "char"
// ------------------------------------------------------------------

// setters for variables on "RayGen"s
OWL_API void owlRayGenSet1c(OWLRayGen obj, const char *name, char val);
OWL_API void owlRayGenSet2c(OWLRayGen obj, const char *name, char x, char y);
OWL_API void owlRayGenSet3c(OWLRayGen obj, const char *name, char x, char y, char z);
OWL_API void owlRayGenSet4c(OWLRayGen obj, const char *name, char x, char y, char z, char w);
OWL_API void owlRayGenSet2cv(OWLRayGen obj, const char *name, const char *val);
OWL_API void owlRayGenSet3cv(OWLRayGen obj, const char *name, const char *val);
OWL_API void owlRayGenSet4cv(OWLRayGen obj, const char *name, const char *val);

// setters for variables on "MissProg"s
OWL_API void owlMissProgSet1c(OWLMissProg obj, const char *name, char val);
OWL_API void owlMissProgSet2c(OWLMissProg obj, const char *name, char x, char y);
OWL_API void owlMissProgSet3c(OWLMissProg obj, const char *name, char x, char y, char z);
OWL_API void owlMissProgSet4c(OWLMissProg obj, const char *name, char x, char y, char z, char w);
OWL_API void owlMissProgSet2cv(OWLMissProg obj, const char *name, const char *val);
OWL_API void owlMissProgSet3cv(OWLMissProg obj, const char *name, const char *val);
OWL_API void owlMissProgSet4cv(OWLMissProg obj, const char *name, const char *val);

// setters for variables on "Geom"s
OWL_API void owlGeomSet1c(OWLGeom obj, const char *name, char val);
OWL_API void owlGeomSet2c(OWLGeom obj, const char *name, char x, char y);
OWL_API void owlGeomSet3c(OWLGeom obj, const char *name, char x, char y, char z);
OWL_API void owlGeomSet4c(OWLGeom obj, const char *name, char x, char y, char z, char w);
OWL_API void owlGeomSet2cv(OWLGeom obj, const char *name, const char *val);
OWL_API void owlGeomSet3cv(OWLGeom obj, const char *name, const char *val);
OWL_API void owlGeomSet4cv(OWLGeom obj, const char *name, const char *val);

// setters for variables on "Params"s
OWL_API void owlParamsSet1c(OWLParams obj, const char *name, char val);
OWL_API void owlParamsSet2c(OWLParams obj, const char *name, char x, char y);
OWL_API void owlParamsSet3c(OWLParams obj, const char *name, char x, char y, char z);
OWL_API void owlParamsSet4c(OWLParams obj, const char *name, char x, char y, char z, char w);
OWL_API void owlParamsSet2cv(OWLParams obj, const char *name, const char *val);
OWL_API void owlParamsSet3cv(OWLParams obj, const char *name, const char *val);
OWL_API void owlParamsSet4cv(OWLParams obj, const char *name, const char *val);

// ------------------------------------------------------------------
// setters for variables of type "uint8_t"
// ------------------------------------------------------------------

// setters for variables on "RayGen"s
OWL_API void owlRayGenSet1uc(OWLRayGen obj, const char *name, uint8_t val);
OWL_API void owlRayGenSet2uc(OWLRayGen obj, const char *name, uint8_t x, uint8_t y);
OWL_API void owlRayGenSet3uc(OWLRayGen obj, const char *name, uint8_t x, uint8_t y, uint8_t z);
OWL_API void owlRayGenSet4uc(OWLRayGen obj, const char *name, uint8_t x, uint8_t y, uint8_t z, uint8_t w);
OWL_API void owlRayGenSet2ucv(OWLRayGen obj, const char *name, const uint8_t *val);
OWL_API void owlRayGenSet3ucv(OWLRayGen obj, const char *name, const uint8_t *val);
OWL_API void owlRayGenSet4ucv(OWLRayGen obj, const char *name, const uint8_t *val);

// setters for variables on "MissProg"s
OWL_API void owlMissProgSet1uc(OWLMissProg obj, const char *name, uint8_t val);
OWL_API void owlMissProgSet2uc(OWLMissProg obj, const char *name, uint8_t x, uint8_t y);
OWL_API void owlMissProgSet3uc(OWLMissProg obj, const char *name, uint8_t x, uint8_t y, uint8_t z);
OWL_API void owlMissProgSet4uc(OWLMissProg obj, const char *name, uint8_t x, uint8_t y, uint8_t z, uint8_t w);
OWL_API void owlMissProgSet2ucv(OWLMissProg obj, const char *name, const uint8_t *val);
OWL_API void owlMissProgSet3ucv(OWLMissProg obj, const char *name, const uint8_t *val);
OWL_API void owlMissProgSet4ucv(OWLMissProg obj, const char *name, const uint8_t *val);

// setters for variables on "Geom"s
OWL_API void owlGeomSet1uc(OWLGeom obj, const char *name, uint8_t val);
OWL_API void owlGeomSet2uc(OWLGeom obj, const char *name, uint8_t x, uint8_t y);
OWL_API void owlGeomSet3uc(OWLGeom obj, const char *name, uint8_t x, uint8_t y, uint8_t z);
OWL_API void owlGeomSet4uc(OWLGeom obj, const char *name, uint8_t x, uint8_t y, uint8_t z, uint8_t w);
OWL_API void owlGeomSet2ucv(OWLGeom obj, const char *name, const uint8_t *val);
OWL_API void owlGeomSet3ucv(OWLGeom obj, const char *name, const uint8_t *val);
OWL_API void owlGeomSet4ucv(OWLGeom obj, const char *name, const uint8_t *val);

// setters for variables on "Params"s
OWL_API void owlParamsSet1uc(OWLParams obj, const char *name, uint8_t val);
OWL_API void owlParamsSet2uc(OWLParams obj, const char *name, uint8_t x, uint8_t y);
OWL_API void owlParamsSet3uc(OWLParams obj, const char *name, uint8_t x, uint8_t y, uint8_t z);
OWL_API void owlParamsSet4uc(OWLParams obj, const char *name, uint8_t x, uint8_t y, uint8_t z, uint8_t w);
OWL_API void owlParamsSet2ucv(OWLParams obj, const char *name, const uint8_t *val);
OWL_API void owlParamsSet3ucv(OWLParams obj, const char *name, const uint8_t *val);
OWL_API void owlParamsSet4ucv(OWLParams obj, const char *name, const uint8_t *val);

// ------------------------------------------------------------------
// setters for variables of type "int16_t"
// ------------------------------------------------------------------

// setters for variables on "RayGen"s
OWL_API void owlRayGenSet1s(OWLRayGen obj, const char *name, int16_t val);
OWL_API void owlRayGenSet2s(OWLRayGen obj, const char *name, int16_t x, int16_t y);
OWL_API void owlRayGenSet3s(OWLRayGen obj, const char *name, int16_t x, int16_t y, int16_t z);
OWL_API void owlRayGenSet4s(OWLRayGen obj, const char *name, int16_t x, int16_t y, int16_t z, int16_t w);
OWL_API void owlRayGenSet2sv(OWLRayGen obj, const char *name, const int16_t *val);
OWL_API void owlRayGenSet3sv(OWLRayGen obj, const char *name, const int16_t *val);
OWL_API void owlRayGenSet4sv(OWLRayGen obj, const char *name, const int16_t *val);

// setters for variables on "MissProg"s
OWL_API void owlMissProgSet1s(OWLMissProg obj, const char *name, int16_t val);
OWL_API void owlMissProgSet2s(OWLMissProg obj, const char *name, int16_t x, int16_t y);
OWL_API void owlMissProgSet3s(OWLMissProg obj, const char *name, int16_t x, int16_t y, int16_t z);
OWL_API void owlMissProgSet4s(OWLMissProg obj, const char *name, int16_t x, int16_t y, int16_t z, int16_t w);
OWL_API void owlMissProgSet2sv(OWLMissProg obj, const char *name, const int16_t *val);
OWL_API void owlMissProgSet3sv(OWLMissProg obj, const char *name, const int16_t *val);
OWL_API void owlMissProgSet4sv(OWLMissProg obj, const char *name, const int16_t *val);

// setters for variables on "Geom"s
OWL_API void owlGeomSet1s(OWLGeom obj, const char *name, int16_t val);
OWL_API void owlGeomSet2s(OWLGeom obj, const char *name, int16_t x, int16_t y);
OWL_API void owlGeomSet3s(OWLGeom obj, const char *name, int16_t x, int16_t y, int16_t z);
OWL_API void owlGeomSet4s(OWLGeom obj, const char *name, int16_t x, int16_t y, int16_t z, int16_t w);
OWL_API void owlGeomSet2sv(OWLGeom obj, const char *name, const int16_t *val);
OWL_API void owlGeomSet3sv(OWLGeom obj, const char *name, const int16_t *val);
OWL_API void owlGeomSet4sv(OWLGeom obj, const char *name, const int16_t *val);

// setters for variables on "Params"s
OWL_API void owlParamsSet1s(OWLParams obj, const char *name, int16_t val);
OWL_API void owlParamsSet2s(OWLParams obj, const char *name, int16_t x, int16_t y);
OWL_API void owlParamsSet3s(OWLParams obj, const char *name, int16_t x, int16_t y, int16_t z);
OWL_API void owlParamsSet4s(OWLParams obj, const char *name, int16_t x, int16_t y, int16_t z, int16_t w);
OWL_API void owlParamsSet2sv(OWLParams obj, const char *name, const int16_t *val);
OWL_API void owlParamsSet3sv(OWLParams obj, const char *name, const int16_t *val);
OWL_API void owlParamsSet4sv(OWLParams obj, const char *name, const int16_t *val);

// ------------------------------------------------------------------
// setters for variables of type "uint16_t"
// ------------------------------------------------------------------

// setters for variables on "RayGen"s
OWL_API void owlRayGenSet1us(OWLRayGen obj, const char *name, uint16_t val);
OWL_API void owlRayGenSet2us(OWLRayGen obj, const char *name, uint16_t x, uint16_t y);
OWL_API void owlRayGenSet3us(OWLRayGen obj, const char *name, uint16_t x, uint16_t y, uint16_t z);
OWL_API void owlRayGenSet4us(OWLRayGen obj, const char *name, uint16_t x, uint16_t y, uint16_t z, uint16_t w);
OWL_API void owlRayGenSet2usv(OWLRayGen obj, const char *name, const uint16_t *val);
OWL_API void owlRayGenSet3usv(OWLRayGen obj, const char *name, const uint16_t *val);
OWL_API void owlRayGenSet4usv(OWLRayGen obj, const char *name, const uint16_t *val);

// setters for variables on "MissProg"s
OWL_API void owlMissProgSet1us(OWLMissProg obj, const char *name, uint16_t val);
OWL_API void owlMissProgSet2us(OWLMissProg obj, const char *name, uint16_t x, uint16_t y);
OWL_API void owlMissProgSet3us(OWLMissProg obj, const char *name, uint16_t x, uint16_t y, uint16_t z);
OWL_API void owlMissProgSet4us(OWLMissProg obj, const char *name, uint16_t x, uint16_t y, uint16_t z, uint16_t w);
OWL_API void owlMissProgSet2usv(OWLMissProg obj, const char *name, const uint16_t *val);
OWL_API void owlMissProgSet3usv(OWLMissProg obj, const char *name, const uint16_t *val);
OWL_API void owlMissProgSet4usv(OWLMissProg obj, const char *name, const uint16_t *val);

// setters for variables on "Geom"s
OWL_API void owlGeomSet1us(OWLGeom obj, const char *name, uint16_t val);
OWL_API void owlGeomSet2us(OWLGeom obj, const char *name, uint16_t x, uint16_t y);
OWL_API void owlGeomSet3us(OWLGeom obj, const char *name, uint16_t x, uint16_t y, uint16_t z);
OWL_API void owlGeomSet4us(OWLGeom obj, const char *name, uint16_t x, uint16_t y, uint16_t z, uint16_t w);
OWL_API void owlGeomSet2usv(OWLGeom obj, const char *name, const uint16_t *val);
OWL_API void owlGeomSet3usv(OWLGeom obj, const char *name, const uint16_t *val);
OWL_API void owlGeomSet4usv(OWLGeom obj, const char *name, const uint16_t *val);

// setters for variables on "Params"s
OWL_API void owlParamsSet1us(OWLParams obj, const char *name, uint16_t val);
OWL_API void owlParamsSet2us(OWLParams obj, const char *name, uint16_t x, uint16_t y);
OWL_API void owlParamsSet3us(OWLParams obj, const char *name, uint16_t x, uint16_t y, uint16_t z);
OWL_API void owlParamsSet4us(OWLParams obj, const char *name, uint16_t x, uint16_t y, uint16_t z, uint16_t w);
OWL_API void owlParamsSet2usv(OWLParams obj, const char *name, const uint16_t *val);
OWL_API void owlParamsSet3usv(OWLParams obj, const char *name, const uint16_t *val);
OWL_API void owlParamsSet4usv(OWLParams obj, const char *name, const uint16_t *val);

// ------------------------------------------------------------------
// setters for variables of type "int"
// ------------------------------------------------------------------

// setters for variables on "RayGen"s
OWL_API void owlRayGenSet1i(OWLRayGen obj, const char *name, int val);
OWL_API void owlRayGenSet2i(OWLRayGen obj, const char *name, int x, int y);
OWL_API void owlRayGenSet3i(OWLRayGen obj, const char *name, int x, int y, int z);
OWL_API void owlRayGenSet4i(OWLRayGen obj, const char *name, int x, int y, int z, int w);
OWL_API void owlRayGenSet2iv(OWLRayGen obj, const char *name, const int *val);
OWL_API void owlRayGenSet3iv(OWLRayGen obj, const char *name, const int *val);
OWL_API void owlRayGenSet4iv(OWLRayGen obj, const char *name, const int *val);

// setters for variables on "MissProg"s
OWL_API void owlMissProgSet1i(OWLMissProg obj, const char *name, int val);
OWL_API void owlMissProgSet2i(OWLMissProg obj, const char *name, int x, int y);
OWL_API void owlMissProgSet3i(OWLMissProg obj, const char *name, int x, int y, int z);
OWL_API void owlMissProgSet4i(OWLMissProg obj, const char *name, int x, int y, int z, int w);
OWL_API void owlMissProgSet2iv(OWLMissProg obj, const char *name, const int *val);
OWL_API void owlMissProgSet3iv(OWLMissProg obj, const char *name, const int *val);
OWL_API void owlMissProgSet4iv(OWLMissProg obj, const char *name, const int *val);

// setters for variables on "Geom"s
OWL_API void owlGeomSet1i(OWLGeom obj, const char *name, int val);
OWL_API void owlGeomSet2i(OWLGeom obj, const char *name, int x, int y);
OWL_API void owlGeomSet3i(OWLGeom obj, const char *name, int x, int y, int z);
OWL_API void owlGeomSet4i(OWLGeom obj, const char *name, int x, int y, int z, int w);
OWL_API void owlGeomSet2iv(OWLGeom obj, const char *name, const int *val);
OWL_API void owlGeomSet3iv(OWLGeom obj, const char *name, const int *val);
OWL_API void owlGeomSet4iv(OWLGeom obj, const char *name, const int *val);

// setters for variables on "Params"s
OWL_API void owlParamsSet1i(OWLParams obj, const char *name, int val);
OWL_API void owlParamsSet2i(OWLParams obj, const char *name, int x, int y);
OWL_API void owlParamsSet3i(OWLParams obj, const char *name, int x, int y, int z);
OWL_API void owlParamsSet4i(OWLParams obj, const char *name, int x, int y, int z, int w);
OWL_API void owlParamsSet2iv(OWLParams obj, const char *name, const int *val);
OWL_API void owlParamsSet3iv(OWLParams obj, const char *name, const int *val);
OWL_API void owlParamsSet4iv(OWLParams obj, const char *name, const int *val);

// ------------------------------------------------------------------
// setters for variables of type "uint32_t"
// ------------------------------------------------------------------

// setters for variables on "RayGen"s
OWL_API void owlRayGenSet1ui(OWLRayGen obj, const char *name, uint32_t val);
OWL_API void owlRayGenSet2ui(OWLRayGen obj, const char *name, uint32_t x, uint32_t y);
OWL_API void owlRayGenSet3ui(OWLRayGen obj, const char *name, uint32_t x, uint32_t y, uint32_t z);
OWL_API void owlRayGenSet4ui(OWLRayGen obj, const char *name, uint32_t x, uint32_t y, uint32_t z, uint32_t w);
OWL_API void owlRayGenSet2uiv(OWLRayGen obj, const char *name, const uint32_t *val);
OWL_API void owlRayGenSet3uiv(OWLRayGen obj, const char *name, const uint32_t *val);
OWL_API void owlRayGenSet4uiv(OWLRayGen obj, const char *name, const uint32_t *val);

// setters for variables on "MissProg"s
OWL_API void owlMissProgSet1ui(OWLMissProg obj, const char *name, uint32_t val);
OWL_API void owlMissProgSet2ui(OWLMissProg obj, const char *name, uint32_t x, uint32_t y);
OWL_API void owlMissProgSet3ui(OWLMissProg obj, const char *name, uint32_t x, uint32_t y, uint32_t z);
OWL_API void owlMissProgSet4ui(OWLMissProg obj, const char *name, uint32_t x, uint32_t y, uint32_t z, uint32_t w);
OWL_API void owlMissProgSet2uiv(OWLMissProg obj, const char *name, const uint32_t *val);
OWL_API void owlMissProgSet3uiv(OWLMissProg obj, const char *name, const uint32_t *val);
OWL_API void owlMissProgSet4uiv(OWLMissProg obj, const char *name, const uint32_t *val);

// setters for variables on "Geom"s
OWL_API void owlGeomSet1ui(OWLGeom obj, const char *name, uint32_t val);
OWL_API void owlGeomSet2ui(OWLGeom obj, const char *name, uint32_t x, uint32_t y);
OWL_API void owlGeomSet3ui(OWLGeom obj, const char *name, uint32_t x, uint32_t y, uint32_t z);
OWL_API void owlGeomSet4ui(OWLGeom obj, const char *name, uint32_t x, uint32_t y, uint32_t z, uint32_t w);
OWL_API void owlGeomSet2uiv(OWLGeom obj, const char *name, const uint32_t *val);
OWL_API void owlGeomSet3uiv(OWLGeom obj, const char *name, const uint32_t *val);
OWL_API void owlGeomSet4uiv(OWLGeom obj, const char *name, const uint32_t *val);

// setters for variables on "Params"s
OWL_API void owlParamsSet1ui(OWLParams obj, const char *name, uint32_t val);
OWL_API void owlParamsSet2ui(OWLParams obj, const char *name, uint32_t x, uint32_t y);
OWL_API void owlParamsSet3ui(OWLParams obj, const char *name, uint32_t x, uint32_t y, uint32_t z);
OWL_API void owlParamsSet4ui(OWLParams obj, const char *name, uint32_t x, uint32_t y, uint32_t z, uint32_t w);
OWL_API void owlParamsSet2uiv(OWLParams obj, const char *name, const uint32_t *val);
OWL_API void owlParamsSet3uiv(OWLParams obj, const char *name, const uint32_t *val);
OWL_API void owlParamsSet4uiv(OWLParams obj, const char *name, const uint32_t *val);

// ------------------------------------------------------------------
// setters for variables of type "float"
// ------------------------------------------------------------------

// setters for variables on "RayGen"s
OWL_API void owlRayGenSet1f(OWLRayGen obj, const char *name, float val);
OWL_API void owlRayGenSet2f(OWLRayGen obj, const char *name, float x, float y);
OWL_API void owlRayGenSet3f(OWLRayGen obj, const char *name, float x, float y, float z);
OWL_API void owlRayGenSet4f(OWLRayGen obj, const char *name, float x, float y, float z, float w);
OWL_API void owlRayGenSet2fv(OWLRayGen obj, const char *name, const float *val);
OWL_API void owlRayGenSet3fv(OWLRayGen obj, const char *name, const float *val);
OWL_API void owlRayGenSet4fv(OWLRayGen obj, const char *name, const float *val);

// setters for variables on "MissProg"s
OWL_API void owlMissProgSet1f(OWLMissProg obj, const char *name, float val);
OWL_API void owlMissProgSet2f(OWLMissProg obj, const char *name, float x, float y);
OWL_API void owlMissProgSet3f(OWLMissProg obj, const char *name, float x, float y, float z);
OWL_API void owlMissProgSet4f(OWLMissProg obj, const char *name, float x, float y, float z, float w);
OWL_API void owlMissProgSet2fv(OWLMissProg obj, const char *name, const float *val);
OWL_API void owlMissProgSet3fv(OWLMissProg obj, const char *name, const float *val);
OWL_API void owlMissProgSet4fv(OWLMissProg obj, const char *name, const float *val);

// setters for variables on "Geom"s
OWL_API void owlGeomSet1f(OWLGeom obj, const char *name, float val);
OWL_API void owlGeomSet2f(OWLGeom obj, const char *name, float x, float y);
OWL_API void owlGeomSet3f(OWLGeom obj, const char *name, float x, float y, float z);
OWL_API void owlGeomSet4f(OWLGeom obj, const char *name, float x, float y, float z, float w);
OWL_API void owlGeomSet2fv(OWLGeom obj, const char *name, const float *val);
OWL_API void owlGeomSet3fv(OWLGeom obj, const char *name, const float *val);
OWL_API void owlGeomSet4fv(OWLGeom obj, const char *name, const float *val);

// setters for variables on "Params"s
OWL_API void owlParamsSet1f(OWLParams obj, const char *name, float val);
OWL_API void owlParamsSet2f(OWLParams obj, const char *name, float x, float y);
OWL_API void owlParamsSet3f(OWLParams obj, const char *name, float x, float y, float z);
OWL_API void owlParamsSet4f(OWLParams obj, const char *name, float x, float y, float z, float w);
OWL_API void owlParamsSet2fv(OWLParams obj, const char *name, const float *val);
OWL_API void owlParamsSet3fv(OWLParams obj, const char *name, const float *val);
OWL_API void owlParamsSet4fv(OWLParams obj, const char *name, const float *val);

// ------------------------------------------------------------------
// setters for variables of type "double"
// ------------------------------------------------------------------

// setters for variables on "RayGen"s
OWL_API void owlRayGenSet1d(OWLRayGen obj, const char *name, double val);
OWL_API void owlRayGenSet2d(OWLRayGen obj, const char *name, double x, double y);
OWL_API void owlRayGenSet3d(OWLRayGen obj, const char *name, double x, double y, double z);
OWL_API void owlRayGenSet4d(OWLRayGen obj, const char *name, double x, double y, double z, double w);
OWL_API void owlRayGenSet2dv(OWLRayGen obj, const char *name, const double *val);
OWL_API void owlRayGenSet3dv(OWLRayGen obj, const char *name, const double *val);
OWL_API void owlRayGenSet4dv(OWLRayGen obj, const char *name, const double *val);

// setters for variables on "MissProg"s
OWL_API void owlMissProgSet1d(OWLMissProg obj, const char *name, double val);
OWL_API void owlMissProgSet2d(OWLMissProg obj, const char *name, double x, double y);
OWL_API void owlMissProgSet3d(OWLMissProg obj, const char *name, double x, double y, double z);
OWL_API void owlMissProgSet4d(OWLMissProg obj, const char *name, double x, double y, double z, double w);
OWL_API void owlMissProgSet2dv(OWLMissProg obj, const char *name, const double *val);
OWL_API void owlMissProgSet3dv(OWLMissProg obj, const char *name, const double *val);
OWL_API void owlMissProgSet4dv(OWLMissProg obj, const char *name, const double *val);

// setters for variables on "Geom"s
OWL_API void owlGeomSet1d(OWLGeom obj, const char *name, double val);
OWL_API void owlGeomSet2d(OWLGeom obj, const char *name, double x, double y);
OWL_API void owlGeomSet3d(OWLGeom obj, const char *name, double x, double y, double z);
OWL_API void owlGeomSet4d(OWLGeom obj, const char *name, double x, double y, double z, double w);
OWL_API void owlGeomSet2dv(OWLGeom obj, const char *name, const double *val);
OWL_API void owlGeomSet3dv(OWLGeom obj, const char *name, const double *val);
OWL_API void owlGeomSet4dv(OWLGeom obj, const char *name, const double *val);

// setters for variables on "Params"s
OWL_API void owlParamsSet1d(OWLParams obj, const char *name, double val);
OWL_API void owlParamsSet2d(OWLParams obj, const char *name, double x, double y);
OWL_API void owlParamsSet3d(OWLParams obj, const char *name, double x, double y, double z);
OWL_API void owlParamsSet4d(OWLParams obj, const char *name, double x, double y, double z, double w);
OWL_API void owlParamsSet2dv(OWLParams obj, const char *name, const double *val);
OWL_API void owlParamsSet3dv(OWLParams obj, const char *name, const double *val);
OWL_API void owlParamsSet4dv(OWLParams obj, const char *name, const double *val);

// ------------------------------------------------------------------
// setters for variables of type "int64_t"
// ------------------------------------------------------------------

// setters for variables on "RayGen"s
OWL_API void owlRayGenSet1l(OWLRayGen obj, const char *name, int64_t val);
OWL_API void owlRayGenSet2l(OWLRayGen obj, const char *name, int64_t x, int64_t y);
OWL_API void owlRayGenSet3l(OWLRayGen obj, const char *name, int64_t x, int64_t y, int64_t z);
OWL_API void owlRayGenSet4l(OWLRayGen obj, const char *name, int64_t x, int64_t y, int64_t z, int64_t w);
OWL_API void owlRayGenSet2lv(OWLRayGen obj, const char *name, const int64_t *val);
OWL_API void owlRayGenSet3lv(OWLRayGen obj, const char *name, const int64_t *val);
OWL_API void owlRayGenSet4lv(OWLRayGen obj, const char *name, const int64_t *val);

// setters for variables on "MissProg"s
OWL_API void owlMissProgSet1l(OWLMissProg obj, const char *name, int64_t val);
OWL_API void owlMissProgSet2l(OWLMissProg obj, const char *name, int64_t x, int64_t y);
OWL_API void owlMissProgSet3l(OWLMissProg obj, const char *name, int64_t x, int64_t y, int64_t z);
OWL_API void owlMissProgSet4l(OWLMissProg obj, const char *name, int64_t x, int64_t y, int64_t z, int64_t w);
OWL_API void owlMissProgSet2lv(OWLMissProg obj, const char *name, const int64_t *val);
OWL_API void owlMissProgSet3lv(OWLMissProg obj, const char *name, const int64_t *val);
OWL_API void owlMissProgSet4lv(OWLMissProg obj, const char *name, const int64_t *val);

// setters for variables on "Geom"s
OWL_API void owlGeomSet1l(OWLGeom obj, const char *name, int64_t val);
OWL_API void owlGeomSet2l(OWLGeom obj, const char *name, int64_t x, int64_t y);
OWL_API void owlGeomSet3l(OWLGeom obj, const char *name, int64_t x, int64_t y, int64_t z);
OWL_API void owlGeomSet4l(OWLGeom obj, const char *name, int64_t x, int64_t y, int64_t z, int64_t w);
OWL_API void owlGeomSet2lv(OWLGeom obj, const char *name, const int64_t *val);
OWL_API void owlGeomSet3lv(OWLGeom obj, const char *name, const int64_t *val);
OWL_API void owlGeomSet4lv(OWLGeom obj, const char *name, const int64_t *val);

// setters for variables on "Params"s
OWL_API void owlParamsSet1l(OWLParams obj, const char *name, int64_t val);
OWL_API void owlParamsSet2l(OWLParams obj, const char *name, int64_t x, int64_t y);
OWL_API void owlParamsSet3l(OWLParams obj, const char *name, int64_t x, int64_t y, int64_t z);
OWL_API void owlParamsSet4l(OWLParams obj, const char *name, int64_t x, int64_t y, int64_t z, int64_t w);
OWL_API void owlParamsSet2lv(OWLParams obj, const char *name, const int64_t *val);
OWL_API void owlParamsSet3lv(OWLParams obj, const char *name, const int64_t *val);
OWL_API void owlParamsSet4lv(OWLParams obj, const char *name, const int64_t *val);

// ------------------------------------------------------------------
// setters for variables of type "uint64_t"
// ------------------------------------------------------------------

// setters for variables on "RayGen"s
OWL_API void owlRayGenSet1ul(OWLRayGen obj, const char *name, uint64_t val);
OWL_API void owlRayGenSet2ul(OWLRayGen obj, const char *name, uint64_t x, uint64_t y);
OWL_API void owlRayGenSet3ul(OWLRayGen obj, const char *name, uint64_t x, uint64_t y, uint64_t z);
OWL_API void owlRayGenSet4ul(OWLRayGen obj, const char *name, uint64_t x, uint64_t y, uint64_t z, uint64_t w);
OWL_API void owlRayGenSet2ulv(OWLRayGen obj, const char *name, const uint64_t *val);
OWL_API void owlRayGenSet3ulv(OWLRayGen obj, const char *name, const uint64_t *val);
OWL_API void owlRayGenSet4ulv(OWLRayGen obj, const char *name, const uint64_t *val);

// setters for variables on "MissProg"s
OWL_API void owlMissProgSet1ul(OWLMissProg obj, const char *name, uint64_t val);
OWL_API void owlMissProgSet2ul(OWLMissProg obj, const char *name, uint64_t x, uint64_t y);
OWL_API void owlMissProgSet3ul(OWLMissProg obj, const char *name, uint64_t x, uint64_t y, uint64_t z);
OWL_API void owlMissProgSet4ul(OWLMissProg obj, const char *name, uint64_t x, uint64_t y, uint64_t z, uint64_t w);
OWL_API void owlMissProgSet2ulv(OWLMissProg obj, const char *name, const uint64_t *val);
OWL_API void owlMissProgSet3ulv(OWLMissProg obj, const char *name, const uint64_t *val);
OWL_API void owlMissProgSet4ulv(OWLMissProg obj, const char *name, const uint64_t *val);

// setters for variables on "Geom"s
OWL_API void owlGeomSet1ul(OWLGeom obj, const char *name, uint64_t val);
OWL_API void owlGeomSet2ul(OWLGeom obj, const char *name, uint64_t x, uint64_t y);
OWL_API void owlGeomSet3ul(OWLGeom obj, const char *name, uint64_t x, uint64_t y, uint64_t z);
OWL_API void owlGeomSet4ul(OWLGeom obj, const char *name, uint64_t x, uint64_t y, uint64_t z, uint64_t w);
OWL_API void owlGeomSet2ulv(OWLGeom obj, const char *name, const uint64_t *val);
OWL_API void owlGeomSet3ulv(OWLGeom obj, const char *name, const uint64_t *val);
OWL_API void owlGeomSet4ulv(OWLGeom obj, const char *name, const uint64_t *val);

// setters for variables on "Params"s
OWL_API void owlParamsSet1ul(OWLParams obj, const char *name, uint64_t val);
OWL_API void owlParamsSet2ul(OWLParams obj, const char *name, uint64_t x, uint64_t y);
OWL_API void owlParamsSet3ul(OWLParams obj, const char *name, uint64_t x, uint64_t y, uint64_t z);
OWL_API void owlParamsSet4ul(OWLParams obj, const char *name, uint64_t x, uint64_t y, uint64_t z, uint64_t w);
OWL_API void owlParamsSet2ulv(OWLParams obj, const char *name, const uint64_t *val);
OWL_API void owlParamsSet3ulv(OWLParams obj, const char *name, const uint64_t *val);
OWL_API void owlParamsSet4ulv(OWLParams obj, const char *name, const uint64_t *val);



// ------------------------------------------------------------------
// setters for "meta" types
// ------------------------------------------------------------------

// setters for variables on "RayGen"s
OWL_API void owlRayGenSetTexture(OWLRayGen obj, const char *name, OWLTexture val);
OWL_API void owlRayGenSetPointer(OWLRayGen obj, const char *name, const void *val);
OWL_API void owlRayGenSetBuffer(OWLRayGen obj, const char *name, OWLBuffer val);
OWL_API void owlRayGenSetGroup(OWLRayGen obj, const char *name, OWLGroup val);
OWL_API void owlRayGenSetRaw(OWLRayGen obj, const char *name, const void *val);

// setters for variables on "Geom"s
OWL_API void owlGeomSetTexture(OWLGeom obj, const char *name, OWLTexture val);
OWL_API void owlGeomSetPointer(OWLGeom obj, const char *name, const void *val);
OWL_API void owlGeomSetBuffer(OWLGeom obj, const char *name, OWLBuffer val);
OWL_API void owlGeomSetGroup(OWLGeom obj, const char *name, OWLGroup val);
OWL_API void owlGeomSetRaw(OWLGeom obj, const char *name, const void *val);

// setters for variables on "Params"s
OWL_API void owlParamsSetTexture(OWLParams obj, const char *name, OWLTexture val);
OWL_API void owlParamsSetPointer(OWLParams obj, const char *name, const void *val);
OWL_API void owlParamsSetBuffer(OWLParams obj, const char *name, OWLBuffer val);
OWL_API void owlParamsSetGroup(OWLParams obj, const char *name, OWLGroup val);
OWL_API void owlParamsSetRaw(OWLParams obj, const char *name, const void *val);

// setters for variables on "MissProg"s
OWL_API void owlMissProgSetTexture(OWLMissProg obj, const char *name, OWLTexture val);
OWL_API void owlMissProgSetPointer(OWLMissProg obj, const char *name, const void *val);
OWL_API void owlMissProgSetBuffer(OWLMissProg obj, const char *name, OWLBuffer val);
OWL_API void owlMissProgSetGroup(OWLMissProg obj, const char *name, OWLGroup val);
OWL_API void owlMissProgSetRaw(OWLMissProg obj, const char *name, const void *val);


// -------------------------------------------------------
// c++ wrappers
// -------------------------------------------------------
#ifdef __cplusplus
// int
inline void owlParamsSet2i(OWLParams obj, const char *name, const owl2i &val)
{ owlParamsSet2i(obj,name,val.x,val.y); }
inline void owlParamsSet3i(OWLParams obj, const char *name, const owl3i &val)
{ owlParamsSet3i(obj,name,val.x,val.y,val.z); }
inline void owlParamsSet4i(OWLParams obj, const char *name, const owl4i &val)
{ owlParamsSet4i(obj,name,val.x,val.y,val.z,val.w); }
// uint
inline void owlParamsSet2ui(OWLParams obj, const char *name, const owl2ui &val)
{ owlParamsSet2ui(obj,name,val.x,val.y); }
inline void owlParamsSet3ui(OWLParams obj, const char *name, const owl3ui &val)
{ owlParamsSet3ui(obj,name,val.x,val.y,val.z); }
inline void owlParamsSet4ui(OWLParams obj, const char *name, const owl4ui &val)
{ owlParamsSet4ui(obj,name,val.x,val.y,val.z,val.w); }
// float
inline void owlParamsSet2f(OWLParams obj, const char *name, const owl2f &val)
{ owlParamsSet2f(obj,name,val.x,val.y); }
inline void owlParamsSet3f(OWLParams obj, const char *name, const owl3f &val)
{ owlParamsSet3f(obj,name,val.x,val.y,val.z); }
inline void owlParamsSet4f(OWLParams obj, const char *name, const owl4f &val)
{ owlParamsSet4f(obj,name,val.x,val.y,val.z,val.w); }

// int
inline void owlGeomSet2i(OWLGeom obj, const char *name, const owl2i &val)
{ owlGeomSet2i(obj,name,val.x,val.y); }
inline void owlGeomSet3i(OWLGeom obj, const char *name, const owl3i &val)
{ owlGeomSet3i(obj,name,val.x,val.y,val.z); }
inline void owlGeomSet4i(OWLGeom obj, const char *name, const owl4i &val)
{ owlGeomSet4i(obj,name,val.x,val.y,val.z,val.w); }
// uint
inline void owlGeomSet2ui(OWLGeom obj, const char *name, const owl2ui &val)
{ owlGeomSet2ui(obj,name,val.x,val.y); }
inline void owlGeomSet3ui(OWLGeom obj, const char *name, const owl3ui &val)
{ owlGeomSet3ui(obj,name,val.x,val.y,val.z); }
inline void owlGeomSet4ui(OWLGeom obj, const char *name, const owl4ui &val)
{ owlGeomSet4ui(obj,name,val.x,val.y,val.z,val.w); }
// float
inline void owlGeomSet2f(OWLGeom obj, const char *name, const owl2f &val)
{ owlGeomSet2f(obj,name,val.x,val.y); }
inline void owlGeomSet3f(OWLGeom obj, const char *name, const owl3f &val)
{ owlGeomSet3f(obj,name,val.x,val.y,val.z); }
inline void owlGeomSet4f(OWLGeom obj, const char *name, const owl4f &val)
{ owlGeomSet4f(obj,name,val.x,val.y,val.z,val.w); }

// int
inline void owlMissProgSet2i(OWLMissProg obj, const char *name, const owl2i &val)
{ owlMissProgSet2i(obj,name,val.x,val.y); }
inline void owlMissProgSet3i(OWLMissProg obj, const char *name, const owl3i &val)
{ owlMissProgSet3i(obj,name,val.x,val.y,val.z); }
inline void owlMissProgSet4i(OWLMissProg obj, const char *name, const owl4i &val)
{ owlMissProgSet4i(obj,name,val.x,val.y,val.z,val.w); }
// uint
inline void owlMissProgSet2ui(OWLMissProg obj, const char *name, const owl2ui &val)
{ owlMissProgSet2ui(obj,name,val.x,val.y); }
inline void owlMissProgSet3ui(OWLMissProg obj, const char *name, const owl3ui &val)
{ owlMissProgSet3ui(obj,name,val.x,val.y,val.z); }
inline void owlMissProgSet4ui(OWLMissProg obj, const char *name, const owl4ui &val)
{ owlMissProgSet4ui(obj,name,val.x,val.y,val.z,val.w); }
// float
inline void owlMissProgSet2f(OWLMissProg obj, const char *name, const owl2f &val)
{ owlMissProgSet2f(obj,name,val.x,val.y); }
inline void owlMissProgSet3f(OWLMissProg obj, const char *name, const owl3f &val)
{ owlMissProgSet3f(obj,name,val.x,val.y,val.z); }
inline void owlMissProgSet4f(OWLMissProg obj, const char *name, const owl4f &val)
{ owlMissProgSet4f(obj,name,val.x,val.y,val.z,val.w); }

// int
inline void owlRayGenSet2i(OWLRayGen obj, const char *name, const owl2i &val)
{ owlRayGenSet2i(obj,name,val.x,val.y); }
inline void owlRayGenSet3i(OWLRayGen obj, const char *name, const owl3i &val)
{ owlRayGenSet3i(obj,name,val.x,val.y,val.z); }
inline void owlRayGenSet4i(OWLRayGen obj, const char *name, const owl4i &val)
{ owlRayGenSet4i(obj,name,val.x,val.y,val.z,val.w); }
// uint
inline void owlRayGenSet2ui(OWLRayGen obj, const char *name, const owl2ui &val)
{ owlRayGenSet2ui(obj,name,val.x,val.y); }
inline void owlRayGenSet3ui(OWLRayGen obj, const char *name, const owl3ui &val)
{ owlRayGenSet3ui(obj,name,val.x,val.y,val.z); }
inline void owlRayGenSet4ui(OWLRayGen obj, const char *name, const owl4ui &val)
{ owlRayGenSet4ui(obj,name,val.x,val.y,val.z,val.w); }
// float
inline void owlRayGenSet2f(OWLRayGen obj, const char *name, const owl2f &val)
{ owlRayGenSet2f(obj,name,val.x,val.y); }
inline void owlRayGenSet3f(OWLRayGen obj, const char *name, const owl3f &val)
{ owlRayGenSet3f(obj,name,val.x,val.y,val.z); }
inline void owlRayGenSet4f(OWLRayGen obj, const char *name, const owl4f &val)
{ owlRayGenSet4f(obj,name,val.x,val.y,val.z,val.w); }
#endif

#ifdef __cplusplus
/*! c++ "convenience variant" of owlInstanceGroupSetTransform that
  also allows passing C++ types) */
  inline void
  owlInstanceGroupSetTransform(OWLGroup group,
                               int childID,
                               const owl4x3f &xfm)
  {
    owlInstanceGroupSetTransform(group,childID,(const float *)&xfm,
                                 OWL_MATRIX_FORMAT_OWL);
  }
/*! c++ "convenience variant" of owlInstanceGroupSetTransform that
  also allows passing C++ types) */
inline void
owlInstanceGroupSetTransform(OWLGroup group,
                             int childID,
                             const owl4x3f *xfm)
{
  owlInstanceGroupSetTransform(group,childID,(const float *)xfm,
                               OWL_MATRIX_FORMAT_OWL);
}

#endif
