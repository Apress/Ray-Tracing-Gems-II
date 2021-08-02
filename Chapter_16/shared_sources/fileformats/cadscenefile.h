/* Copyright (c) 2011-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#ifndef CSF_GLTF2_SUPPORT
#define CSF_GLTF2_SUPPORT 1
#endif

#ifndef CSF_ZIP_SUPPORT
#define CSF_ZIP_SUPPORT 1
#endif

extern "C" {

#ifdef _WIN32
  #if defined(CSFAPI_EXPORTS)
    #define CSFAPI __declspec(dllexport)
  #elif defined(CSFAPI_IMPORTS)
    #define CSFAPI __declspec(dllimport)
  #else
    #define CSFAPI
  #endif
#else
  #define CSFAPI
#endif

  enum {
    CADSCENEFILE_VERSION                = 6,
    // binary compatibility
    CADSCENEFILE_VERSION_COMPAT         = 2,

    CADSCENEFILE_VERSION_BASE           = 1,
    // add support for material meta information
    CADSCENEFILE_VERSION_MATERIAL       = 2,
    // add support for fileflags
    CADSCENEFILE_VERSION_FILEFLAGS      = 3,
    // changes CSFNodePart.lineWidth to CSFNodePart.nodeIDX
    CADSCENEFILE_VERSION_PARTNODEIDX    = 4,
    // adds support for meta pointers
    CADSCENEFILE_VERSION_META           = 5,
    // adds support for vertex and part channels
    CADSCENEFILE_VERSION_GEOMETRYCHANNELS = 6,

    CADSCENEFILE_NOERROR                = 0,
    CADSCENEFILE_ERROR_NOFILE           = 1,
    CADSCENEFILE_ERROR_VERSION          = 2,
    CADSCENEFILE_ERROR_OPERATION        = 3,

    // node tree, no multiple references to same node 
    // always set, as no application supports non-unique case
    CADSCENEFILE_FLAG_UNIQUENODES       = 1,

    // all triangles/lines are using strips instead of index lists
    // never set, only special purpose file/application made use of this
    CADSCENEFILE_FLAG_STRIPS            = 2,

    // file has meta node array
    CADSCENEFILE_FLAG_META_NODE         = 4,
    // file has meta geometry array
    CADSCENEFILE_FLAG_META_GEOMETRY     = 8,
    // file has meta file pointer
    CADSCENEFILE_FLAG_META_FILE         = 16,

    // number of uint32_t per GUID
    CADSCENEFILE_LENGTH_GUID            = 4,
    CADSCENEFILE_LENGTH_STRING          = 128,
  };

  #define CADSCENEFILE_RESTARTINDEX     (~0)

  /*

  Version History
  ---------------

  1 initial
  2 !binary break
    material allows custom payload
    deprecate geometry matrix
    deprecate geometry part vertex
  3 hasUniqueNodes became a bitflag
    added strip indices flag, file is either strip or non-strip
  4 lineWidth changed to nodeIDX, allows per-part sub-transforms. sub-transforms should be below
    object in hierarchy and not affect geometry bbox
  5 meta information handling
  6 vertex channels using deprecated geometry matrix space

  Example Structure
  -----------------
  
  CSFMaterials   
  0 Red          
  1 Green        
  2 Blue         
                
                 
  CSFGeometries (index,vertex & "parts")
  0 Box
  1 Cylinder
  e.g. parts  (CSFGeometryPart defines a region in the indexbuffers of CSFGeometry):
    0 mantle
    1 top cap
    2 bottom cap

  There is no need to have multiple parts, but for experimenting with
  rendering some raw CAD data, having each patch/surface feature individually
  can be useful. A typical CAD file with use one CSFGeometry per Solid (e.g. cube)
  and multiple CSFGeometryParts for each "feature" (face of a cube etc.).

  CSFNodes  (hierarchy of nodes)

  A node can also reference a geometry, this way the same geometry
  data can be instanced multiple times.
  If the node references geometry, then it must have an
  array of "CSFNodePart" matching referenced CSFGeometryParts.
  The CSFNodePart encodes the materials/matrix as well as its
  "visibility" (active) state.


  CSFoffset - nameOFFSET variables
  --------------------------------

  CSFoffset is only indirectly used during save and load operations.
  As end user you can ignore the various "nameOFFSET" variables
  in all the unions, as well as the pointers array.
  */

  typedef struct _CSFLoaderConfig {
    // read only access to loaded csf data
    // means we can use filemappings
    // default = false
    // the primary structs are still allocated for write access, 
    // but all pointers within them are mapped.
    int secondariesReadOnly;

  #if CSF_GLTF2_SUPPORT
    // uses hashes of geometry data to figure out what gltf mesh 
    // data is re-used under different materials, and can 
    // therefore be mapped to a CSF geometry.
    // default = true
    int gltfFindUniqueGeometries;
  #endif
  }CSFLoaderConfig;

  typedef unsigned long long CSFoffset;
  typedef unsigned int CSFGuid[CADSCENEFILE_LENGTH_GUID];

  // optional, if one wants to pack
  // additional meta information into the bytes arrays
  typedef struct _CSFBytePacket {
    CSFGuid               guid;
    int                   numBytes; // includes size of this header
  }CSFBytePacket;

  #define CSFGUID_MATERIAL_GLTF2       {0,0,0,2}

  typedef struct _CSFMaterialGLTF2Texture {
    char                  name[CADSCENEFILE_LENGTH_STRING];
    uint16_t              minFilter;
    uint16_t              magFilter;
    uint16_t              wrapS;
    uint16_t              wrapT;
    float                 scale;
    int                   coord;
    int                   xformUsed;
    int                   xformCoord;
    float                 xformOffset[2];
    float                 xformScale[2];
    float                 xformRotation;
  }CSFMaterialGLTF2Texture;

  typedef struct _CSFMaterialGLTF2Meta {
    CSFBytePacket         packet;

    //-1: unlit
    // 0: metallicRoughness
    // 1: specularGlossiness
    int                   shadingModel;
    int                   doubleSided;
    int                   alphaMode;
    float                 alphaCutoff;
    float                 emissiveFactor[3];

    union {
      struct {
        float             baseColorFactor[4];
        float             metallicFactor;
        float             roughnessFactor;

        _CSFMaterialGLTF2Texture baseColorTexture;
        _CSFMaterialGLTF2Texture metallicRoughnessTexture;
      };

      struct {
        float             diffuseFactor[4];
        float             specularFactor[3];
        float             glossinessFactor;

        _CSFMaterialGLTF2Texture diffuseTexture;
        _CSFMaterialGLTF2Texture specularGlossinessTexture;
      };
    };

    _CSFMaterialGLTF2Texture occlusionTexture;
    _CSFMaterialGLTF2Texture normalTexture;
    _CSFMaterialGLTF2Texture emissiveTexture;
  }CSFMaterialGLTF2Meta;

  typedef struct _CSFMeta {
    char    name[CADSCENEFILE_LENGTH_STRING];
    int     flags;
    CSFoffset             numBytes;
    union {
      CSFoffset           bytesOFFSET;
      unsigned char*      bytes;
    };
  }CSFMeta;

  typedef struct _CSFMaterial {
    char    name[CADSCENEFILE_LENGTH_STRING];
    float   color[4];
    int     type;         // arbitrary data

    // FIXME should move meta outside material, but breaks binary
    // compatibility
    int     numBytes;
    union {
      CSFoffset           bytesOFFSET;
      unsigned char*      bytes;
    };
  }CSFMaterial;

  typedef enum _CSFGeometryPartChannel {
    // CSFGeometryPartChannelBbox
    CSFGEOMETRY_PARTCHANNEL_BBOX,
    CSFGEOMETRY_PARTCHANNELS,
  } CSFGeometryPartChannel;

  typedef struct _CSFGeometryPartBbox {
    float min[3];
    float max[3];
  }CSFGeometryPartBbox;
  
  typedef enum _CSFGeometryNormalChannel {
    // float[3]
    // can extend but must not change order
    CSFGEOMETRY_NORMALCHANNEL_NORMAL,
    CSFGEOMETRY_NORMALCHANNELS,
  }CSFGeometryNormalChannel;

  typedef enum _CSFGeometryTexChannel {
    // float[2]
    // can extend but must not change order
    CSFGEOMETRY_TEXCHANNEL_GENERIC,
    CSFGEOMETRY_TEXCHANNEL_LIGHTMAP,
    CSFGEOMETRY_TEXCHANNELS,
  }CSFGeometryTexChannel;

  typedef enum _CSFGeometryAuxChannel {
    // float[4]
    // can extend but must not change order
    CSFGEOMETRY_AUXCHANNEL_RADIANCE,
    CSFGEOMETRY_AUXCHANNELS,
  }CSFGeometryAuxChannel;

  typedef struct _CSFGeometryPart {
    int     _deprecated;    // deprecated
    int     numIndexSolid;  // number of triangle indices that the part uses
    int     numIndexWire;   // number of line indices that the part uses
  }CSFGeometryPart;

  typedef struct _CSFGeometry{
    /*
    Each Geometry stores:
    - optional index buffer triangles (solid)
    - optional index buffer for lines (wire)
    
      At least one of the index buffers must be
      present.

    - vertex buffer (mandatory)
    - optional vertex attribute (normal,tex,aux) buffers

      Each vertex channel is stored in full for all vertices before
      subsequent channels. Use the channel getter functions.
      Auxiliar data uses the auxStorageOrder array to encode
      what and in which order channels are stored.
    
    - parts array

      index buffer:  { part 0 ....,  part 1.., part 2......., ...}

      Each geometry part represents a range within the
      index buffers. The parts are stored ascending
      in the index buffer. To get the starting 
      offset, use the sum of the previous parts.

    - perpart array

      Allows storing auxiliar per-part channel data (CSFGeometryPartChannel)
      perpartStorageOrder array encodes what and in which
      order the channels are stored.
      Use the channel getter function and size functions.
    */

    // ordering of variable is a bit weird due to keeping binary
    // compatibility with past versions
    float                       _deprecated[4];

    // VERSION < CADSCENEFILE_VERSION
    /////////////////////////////////////////////////
    int                         numNormalChannels;
    int                         numTexChannels;
    int                         numAuxChannels;
    int                         numPartChannels;

    union {
      // numAuxChannels
      CSFoffset                 auxStorageOrderOFFSET;
      _CSFGeometryAuxChannel*   auxStorageOrder;
    };
    union {
      // 4 * numVertices * numAuxiliarChannels
      CSFoffset                 auxOFFSET;
      float*                    aux;
    };

    union {
      // numPartChannels
      CSFoffset                 perpartStorageOrderOFFSET;
      CSFGeometryPartChannel*   perpartStorageOrder;
    };
    union {
      // sized implicitly use CSFGeometry_getPerPartSize functions
      CSFoffset                 perpartOFFSET;
      unsigned char*            perpart;
    };

    // VERSION < CADSCENEFILE_VERSION_GEOMETRYCHANNELS
    /////////////////////////////////////////////////
    int                         numParts;
    int                         numVertices;
    int                         numIndexSolid;
    int                         numIndexWire;

    union {
      // 3 components * numVertices
      CSFoffset                 vertexOFFSET;
      float*                    vertex;
    };

    union {
      // 3 components * numVertices * numNormalChannels
      // canonical order as defined by CSFGeometryNormalChannel
      CSFoffset                 normalOFFSET;
      float*                    normal;
    };

    union {
      // 2 components * numVertices * numTexChannels
      // canonical order is defined in CSFGeometryTexChannel
      CSFoffset                 texOFFSET;
      float*                    tex;
    };

    union {
      CSFoffset                 indexSolidOFFSET;
      unsigned int*             indexSolid;
    };

    union {
      CSFoffset                 indexWireOFFSET;
      unsigned int*             indexWire;
    };

    union {
      CSFoffset                 partsOFFSET;
      CSFGeometryPart*          parts;
    };
  }CSFGeometry;

  typedef struct _CSFNodePart {
    // CSFNodePart defines the state for the corresponding CSFGeometryPart

    // allow setting visibility of a part, 0 or 1
    // should always be 1
    int                   active;

    // index into csf->materials
    // must alwaye be >= 0
    // ideally all parts of a node use the same material
    int                   materialIDX;

    // index into csf->nodes
    // if -1 it uses the matrix of the node it belongs to.
    // This is highly recommended to be used.
    // if >= 0 the nodeIDX should be a child of the 
    // part's node.
    int                   nodeIDX;
  }CSFNodePart;

  typedef struct _CSFNode{
    /*
    CSFNodes form a hierarchy, starting at
    csf->nodes[csf->rootIDX].

    Each node can have children. If CADSCENEFILE_FLAG_UNIQUENODES is set
    the hierarchy is a tree.

    Each Node stores:
    - the object transform (relative to parent)
    - the world transform (final transform to get from object to world space
      node.worldTM = node.parent.worldTM * node.objectTM;
    - optional geometry reference
    - optional array of node children
    - the parts array is mandatory if a geometry is referenced
      and must be sized to form a 1:1 correspondence to the geoemtry's parts.
    */

    float                 objectTM[16];
    float                 worldTM[16];

    // index into csf->geometries
    // can be -1 (no geometry used) or >= 0
    int                   geometryIDX;

    // if geometryIDX >= 0, must match geometry's numParts
    int                   numParts;
    int                   numChildren;
    union {
      // must exist if geometryIDX >= 0, null otherwise
      CSFoffset           partsOFFSET;
      CSFNodePart*        parts;
    };
    union {
      // array of indices into csf->nodes
      // each must be >= 0
      // array must be != null if numChildren is > 0
      CSFoffset           childrenOFFSET;
      int*                children;
    };
  }CSFNode;


  typedef struct _CSFile {
    int                   magic;
    int                   version;
    
    // see CADSCENEFILE_FLAG_??
    unsigned int          fileFlags;

    // used internally for load & save operations, can be ignored
    int                   numPointers;

    int                   numGeometries;
    int                   numMaterials;
    int                   numNodes;

    // index into csf->nodes where the root node is located
    // must be >= 0
    int                   rootIDX;

    union {
      // the pointers are used internally for load & save operations
      // no need to specify prior save
      // no need to access pos load

      CSFoffset           pointersOFFSET;
      CSFoffset*          pointers;
    };

    union {
      CSFoffset           geometriesOFFSET;
      CSFGeometry*        geometries;
    };

    union {
      CSFoffset           materialsOFFSET;
      CSFMaterial*        materials;
    };

    union {
      CSFoffset           nodesOFFSET;
      CSFNode*            nodes;
    };

    //----------------------------------
    // Only available for version >= CADSCENEFILE_VERSION_META and if flag is set.
    // Use the getter functions to access, they return null if the criteria aren't met.
    // Otherwise this memory will overlap with different content.
    
    union {
      // one per node if CADSCENEFILE_FLAG_META_NODE is set
      CSFoffset           nodeMetasOFFSET;
      CSFMeta*            nodeMetas;
    };
    union {
      // one per geometry if CADSCENEFILE_FLAG_META_GEOMETRY is set
      CSFoffset           geometryMetasOFFSET;
      CSFMeta*            geometryMetas;
    };
    union {
      // one per file if CADSCENEFILE_FLAG_META_FILE is set
      CSFoffset           fileMetaOFFSET;
      CSFMeta*            fileMeta;
    };
    //----------------------------------
  }CSFile;

  typedef struct CSFileMemory_s* CSFileMemoryPTR;

  // Internal allocation wrapper
  // also handles details for loading operations
  CSFAPI CSFileMemoryPTR CSFileMemory_new();
  CSFAPI CSFileMemoryPTR CSFileMemory_newCfg(const CSFLoaderConfig* config);

  // alloc functions are thread-safe
  // fill if provided must provide sz bytes
  CSFAPI void*  CSFileMemory_alloc(CSFileMemoryPTR mem, size_t sz, const void*fill);
  // fillPartial if provided must provide szPartial bytes and szPartial <= sz
  CSFAPI void*  CSFileMemory_allocPartial(CSFileMemoryPTR mem, size_t sz, size_t szPartial, const void* fillPartial);
  // all allocations within will be freed
  CSFAPI void   CSFileMemory_delete(CSFileMemoryPTR mem);

  // The data pointed to is modified, therefore the raw load operation can be executed only once.
  // It must be preserved for as long as the csf and its internals are accessed
  CSFAPI int    CSFile_loadRaw (CSFile** outcsf, size_t sz, void* data);

  // All allocations are done within the provided file memory.
  // It must be preserved for as long as the csf and its internals are accessed
  CSFAPI int    CSFile_load    (CSFile** outcsf, const char* filename, CSFileMemoryPTR mem);

  CSFAPI int    CSFile_save    (const CSFile* csf, const char* filename);

  // sets all content of _deprecated to zero, automatically done at load
  // recommended to be done prior safe
  CSFAPI void   CSFile_clearDeprecated(CSFile* csf);

  // sets up single normal/tex channel based on array existence
  CSFAPI void   CSFile_setupDefaultChannels(CSFile* csf);
  CSFAPI void   CSFGeometry_setupDefaultChannels(CSFGeometry* geo);


  // returns vec3*numVertices
  CSFAPI const float* CSFGeometry_getNormalChannel(const CSFGeometry* geo, CSFGeometryNormalChannel channel);
  // returns vec2*numVertices
  CSFAPI const float* CSFGeometry_getTexChannel(const CSFGeometry* geo, CSFGeometryTexChannel texChannel);
  // returns vec4*numVertices
  CSFAPI const float* CSFGeometry_getAuxChannel(const CSFGeometry* geo, CSFGeometryAuxChannel channel);
  // returns arbitrary struct array * numParts
  CSFAPI const void*  CSFGeometry_getPartChannel(const CSFGeometry* geo, CSFGeometryPartChannel channel);

  // accumulates partchannel sizes and multiplies with geo->numParts
  CSFAPI size_t       CSFGeometry_getPerPartSize(const CSFGeometry* geo);
  // accumulates partchannel sizes and multiplies with provided numParts
  CSFAPI size_t       CSFGeometry_getPerPartRequiredSize(const CSFGeometry* geo, int numParts);
  CSFAPI size_t       CSFGeometry_getPerPartRequiredOffset(const CSFGeometry* geo, int numParts, CSFGeometryPartChannel channel);
  // single element size
  CSFAPI size_t       CSFGeometryPartChannel_getSize(CSFGeometryPartChannel channel);


  // safer to use these
  CSFAPI const  CSFMeta*          CSFile_getNodeMetas(const CSFile* csf);
  CSFAPI const  CSFMeta*          CSFile_getGeometryMetas(const CSFile* csf);
  CSFAPI const  CSFMeta*          CSFile_getFileMeta(const CSFile* csf);

  CSFAPI const  CSFBytePacket*    CSFile_getMetaBytePacket(const  CSFMeta* meta, CSFGuid guid);
  CSFAPI const  CSFBytePacket*    CSFile_getMaterialBytePacket(const CSFile* csf, int materialIDX, CSFGuid guid);

  CSFAPI int    CSFile_transform(CSFile *csf);  // requires unique nodes

#if CSF_ZIP_SUPPORT || CSF_GLTF2_SUPPORT
  CSFAPI int    CSFile_loadExt(CSFile** outcsf, const char* filename, CSFileMemoryPTR mem);
#endif
#if CSF_ZIP_SUPPORT 
  CSFAPI int    CSFile_saveExt(CSFile* csf, const char* filename);
#endif

  CSFAPI void   CSFMatrix_identity(float*);
  
};

