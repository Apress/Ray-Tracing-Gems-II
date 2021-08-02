/* Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
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


//////////////////////////////////////////////////////////////////////////

#include <assert.h>
#include <map>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#if CSF_ZIP_SUPPORT
#include <zlib.h>
#endif

#include <mutex>

#include <stddef.h>  // for memcpy
#include <string.h>  // for memcpy

#include "cadscenefile.h"
#include <NvFoundation.h>
#include <nvh/filemapping.hpp>

#define CADSCENEFILE_MAGIC 1567262451

#ifdef WIN32
#define FREAD(a, b, c, d, e) fread_s(a, b, c, d, e)
#else
#define FREAD(a, b, c, d, e) fread(a, c, d, e)
#endif

#if defined(WIN32) && (defined(__amd64__) || defined(__x86_64__) || defined(_M_X64) || defined(__AMD64__))
#define xftell(f) _ftelli64(f)
#define xfseek(f, pos, encoded) _fseeki64(f, pos, encoded)
#else
#define xftell(f) ftell(f)
#define xfseek(f, pos, encoded) fseek(f, (long)pos, encoded)
#endif

struct CSFileMemory_s
{
  CSFLoaderConfig m_config;

  std::vector<void*> m_allocations;
  std::mutex         m_mutex;

  std::vector<nvh::FileReadMapping> m_readMappings;

  void* alloc(size_t size, const void* indata = nullptr, size_t indataSize = 0)
  {
    if(size == 0)
      return nullptr;

    void* data = malloc(size);
    if(indata)
    {
      indataSize = indataSize ? indataSize : size;
      memcpy(data, indata, indataSize);
    }

    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_allocations.push_back(data);
    }
    return data;
  }

  template <typename T>
  T* allocT(size_t size, const T* indata, size_t indataSize = 0)
  {
    return (T*)alloc(size, indata, indataSize);
  }

  CSFileMemory_s()
  {
    m_config.secondariesReadOnly = 0;
#if CSF_GLTF2_SUPPORT
    m_config.gltfFindUniqueGeometries = 1;
#endif
  }

  ~CSFileMemory_s()
  {
    for(size_t i = 0; i < m_allocations.size(); i++)
    {
      free(m_allocations[i]);
    }
    m_readMappings.clear();
  }
};

CSFAPI CSFileMemoryPTR CSFileMemory_new()
{
  return new CSFileMemory_s;
}
CSFAPI CSFileMemoryPTR CSFileMemory_newCfg(const CSFLoaderConfig* config)
{
  CSFileMemoryPTR mem = new CSFileMemory_s;
  mem->m_config       = *config;
  return mem;
}

CSFAPI void CSFileMemory_delete(CSFileMemoryPTR mem)
{
  delete mem;
}

CSFAPI void* CSFileMemory_alloc(CSFileMemoryPTR mem, size_t sz, const void* fill)
{
  return mem->alloc(sz, fill);
}


CSFAPI void* CSFileMemory_allocPartial(CSFileMemoryPTR mem, size_t sz, size_t szPartial, const void* fillPartial)
{
  return mem->alloc(sz, szPartial == 0 ? nullptr : fillPartial, szPartial);
}

static int CSFile_invalidVersion(const CSFile* csf)
{
  return csf->magic != CADSCENEFILE_MAGIC || csf->version < CADSCENEFILE_VERSION_COMPAT || csf->version > CADSCENEFILE_VERSION;
}

static size_t CSFile_getHeaderSize(const CSFile* csf)
{
  if(csf->version >= CADSCENEFILE_VERSION_META)
  {
    return sizeof(CSFile);
  }
  else
  {
    return offsetof(CSFile, nodeMetas);
  }
}

static size_t CSFile_getRawSize(const CSFile* csf)
{
  if(CSFile_invalidVersion(csf))
    return 0;

  return csf->pointersOFFSET + csf->numPointers * sizeof(CSFoffset);
}


template <typename T>
static void fixPointer(T*& ptr, CSFoffset offset, void* base)
{
  if(offset)
  {
    ptr = (T*)(((uint8_t*)base) + offset);
  }
}

static void CSFile_fixSecondaryPointers(CSFile* csf, void* base)
{
  // setup pointers
  for(int m = 0; m < csf->numMaterials; m++)
  {
    CSFMaterial& material = csf->materials[m];
    fixPointer(material.bytes, material.bytesOFFSET, base);
  }
  for(int g = 0; g < csf->numGeometries; g++)
  {
    CSFGeometry& geo = csf->geometries[g];
    fixPointer(geo.vertex, geo.vertexOFFSET, base);
    fixPointer(geo.normal, geo.normalOFFSET, base);
    fixPointer(geo.indexSolid, geo.indexSolidOFFSET, base);
    fixPointer(geo.indexWire, geo.indexWireOFFSET, base);
    fixPointer(geo.parts, geo.partsOFFSET, base);
    fixPointer(geo.auxStorageOrder, geo.auxStorageOrderOFFSET, base);
    fixPointer(geo.aux, geo.auxOFFSET, base);
    fixPointer(geo.perpart, geo.perpartOFFSET, base);
    fixPointer(geo.perpartStorageOrder, geo.perpartStorageOrderOFFSET, base);
  }
  for(int n = 0; n < csf->numNodes; n++)
  {
    CSFNode& node = csf->nodes[n];
    fixPointer(node.children, node.childrenOFFSET, base);
    fixPointer(node.parts, node.partsOFFSET, base);
  }
  if(CSFile_getGeometryMetas(csf))
  {
    for(int g = 0; g < csf->numGeometries; g++)
    {
      CSFMeta& meta = csf->geometryMetas[g];
      fixPointer(meta.bytes, meta.bytesOFFSET, base);
    }
  }
  if(CSFile_getNodeMetas(csf))
  {
    for(int n = 0; n < csf->numNodes; n++)
    {
      CSFMeta& meta = csf->nodeMetas[n];
      fixPointer(meta.bytes, meta.bytesOFFSET, base);
    }
  }
  if(CSFile_getFileMeta(csf))
  {
    CSFMeta& meta = csf->fileMeta[0];
    fixPointer(meta.bytes, meta.bytesOFFSET, base);
  }
}

CSFAPI int CSFile_loadRaw(CSFile** outcsf, size_t size, void* dataraw)
{
  char*   data = (char*)dataraw;
  CSFile* csf  = (CSFile*)data;

  if(size < sizeof(CSFile) || CSFile_invalidVersion(csf))
  {
    *outcsf = 0;
    return CADSCENEFILE_ERROR_VERSION;
  }

  if(size < CSFile_getRawSize((CSFile*)dataraw))
  {
    *outcsf = 0;
    return CADSCENEFILE_ERROR_VERSION;
  }

  if(csf->version < CADSCENEFILE_VERSION_FILEFLAGS)
  {
    csf->fileFlags = csf->fileFlags ? CADSCENEFILE_FLAG_UNIQUENODES : 0;
  }

  csf->pointersOFFSET += (CSFoffset)csf;
  for(int i = 0; i < csf->numPointers; i++)
  {
    CSFoffset* ptr = (CSFoffset*)(data + csf->pointers[i]);
    *(ptr) += (CSFoffset)csf;
  }

  if(csf->version < CADSCENEFILE_VERSION_PARTNODEIDX)
  {
    for(int i = 0; i < csf->numNodes; i++)
    {
      for(int p = 0; p < csf->nodes[i].numParts; p++)
      {
        csf->nodes[i].parts[p].nodeIDX = -1;
      }
    }
  }

  if(csf->version < CADSCENEFILE_VERSION_GEOMETRYCHANNELS)
  {
    CSFile_setupDefaultChannels(csf);
  }

  CSFile_clearDeprecated(csf);


  csf->numPointers = 0;
  csf->pointers    = nullptr;

  *outcsf = csf;

  return CADSCENEFILE_NOERROR;
}

CSFAPI int CSFile_loadReadOnly(CSFile** outcsf, const char* filename, CSFileMemoryPTR mem)
{
  nvh::FileReadMapping file;
  if(!file.open(filename))
  {
    return CADSCENEFILE_ERROR_NOFILE;
  }

  const uint8_t* base = (const uint8_t*)file.data();

  // allocate the primary arrays
  CSFile* csf = mem->allocT(sizeof(CSFile), (const CSFile*)base, sizeof(CSFile));

  csf->materials = mem->allocT(sizeof(CSFMaterial) * csf->numMaterials, (const CSFMaterial*)(base + csf->materialsOFFSET));
  csf->geometries = mem->allocT(sizeof(CSFGeometry) * csf->numGeometries, (const CSFGeometry*)(base + csf->geometriesOFFSET));
  csf->nodes       = mem->allocT(sizeof(CSFNode) * csf->numNodes, (const CSFNode*)(base + csf->nodesOFFSET));
  csf->pointers    = 0;
  csf->numPointers = 0;
  if(CSFile_getGeometryMetas(csf))
  {
    csf->geometryMetas = mem->allocT(sizeof(CSFMeta) * csf->numGeometries, (const CSFMeta*)(base + csf->geometryMetasOFFSET));
  }
  if(CSFile_getNodeMetas(csf))
  {
    csf->nodeMetas = mem->allocT(sizeof(CSFMeta) * csf->numNodes, (const CSFMeta*)(base + csf->nodeMetasOFFSET));
  }
  if(CSFile_getFileMeta(csf))
  {
    csf->fileMeta = mem->allocT(sizeof(CSFMeta), (const CSFMeta*)(base + csf->fileMetaOFFSET));
  }
  if(csf->version < CADSCENEFILE_VERSION_GEOMETRYCHANNELS)
  {
    CSFile_setupDefaultChannels(csf);
  }

  CSFile_fixSecondaryPointers(csf, const_cast<void*>((const void*)base));

  mem->m_readMappings.push_back(std::move(file));

  *outcsf = csf;
  return CADSCENEFILE_NOERROR;
}

CSFAPI int CSFile_load(CSFile** outcsf, const char* filename, CSFileMemoryPTR mem)
{
  if(!filename)
  {
    return CADSCENEFILE_ERROR_NOFILE;
  }

  FILE* file;
#ifdef WIN32
  if(fopen_s(&file, filename, "rb"))
#else
  if((file = fopen(filename, "rb")) == nullptr)
#endif
  {
    *outcsf = 0;
    return CADSCENEFILE_ERROR_NOFILE;
  }

  CSFile header     = {0};
  size_t sizeshould = 0;
  if(!FREAD(&header, sizeof(header), sizeof(header), 1, file) || (sizeshould = CSFile_getRawSize(&header)) == 0)
  {
    fclose(file);
    *outcsf = 0;
    return CADSCENEFILE_ERROR_VERSION;
  }

  if(mem->m_config.secondariesReadOnly)
  {
    fclose(file);
    return CSFile_loadReadOnly(outcsf, filename, mem);
  }

  // load the full file to memory
  xfseek(file, 0, SEEK_END);
  size_t size = (size_t)xftell(file);
  xfseek(file, 0, SEEK_SET);

  if(sizeshould != size)
  {
    fclose(file);
    *outcsf = 0;
    return CADSCENEFILE_ERROR_VERSION;
  }

  char* data = (char*)mem->alloc(size);
  FREAD(data, size, size, 1, file);
  fclose(file);

  return CSFile_loadRaw(outcsf, size, data);
}

#if CSF_GLTF2_SUPPORT
CSFAPI int CSFile_loadGTLF(CSFile** outcsf, const char* filename, CSFileMemoryPTR mem);
#endif

#if CSF_ZIP_SUPPORT || CSF_GLTF2_SUPPORT
CSFAPI int CSFile_loadExt(CSFile** outcsf, const char* filename, CSFileMemoryPTR mem)
{
  if(!filename)
  {
    return CADSCENEFILE_ERROR_NOFILE;
  }
  size_t len = strlen(filename);
#if CSF_ZIP_SUPPORT
  if(len > 3 && strcmp(filename + len - 3, ".gz") == 0)
  {
    gzFile filegz = gzopen(filename, "rb");
    if(!filegz)
    {
      *outcsf = 0;
      return CADSCENEFILE_ERROR_NOFILE;
    }

    CSFile header     = {0};
    size_t sizeshould = 0;
    if(!gzread(filegz, &header, (z_off_t)sizeof(header)) || (sizeshould = CSFile_getRawSize(&header)) == 0)
    {
      gzclose(filegz);
      *outcsf = 0;
      return CADSCENEFILE_ERROR_VERSION;
    }


    gzseek(filegz, 0, SEEK_SET);
    char* data = (char*)CSFileMemory_alloc(mem, sizeshould, 0);
    if(!gzread(filegz, data, (z_off_t)sizeshould))
    {
      gzclose(filegz);
      *outcsf = 0;
      return CADSCENEFILE_ERROR_VERSION;
    }
    gzclose(filegz);

    return CSFile_loadRaw(outcsf, sizeshould, data);
  }
  else
#endif
#if CSF_GLTF2_SUPPORT
      if(len > 5 && strcmp(filename + len - 5, ".gltf") == 0)
  {
    return CSFile_loadGTLF(outcsf, filename, mem);
  }
#endif
  {
    return CSFile_load(outcsf, filename, mem);
  }
}
#endif

struct OutputFILE
{
  FILE* m_file;

  int open(const char* filename)
  {
#ifdef WIN32
    return fopen_s(&m_file, filename, "wb");
#else
    return (m_file = fopen(filename, "wb")) ? 1 : 0;
#endif
  }
  void close() { fclose(m_file); }
  void seek(size_t offset, int pos) { xfseek(m_file, offset, pos); }
  void write(const void* data, size_t dataSize) { fwrite(data, dataSize, 1, m_file); }
};


struct OutputBuf
{
  char*  m_data;
  size_t m_allocated;
  size_t m_used;
  size_t m_cur;

  int open(const char* filename)
  {
    m_allocated = 1024 * 1024;
    m_data      = (char*)malloc(m_allocated);
    m_used      = 0;
    m_cur       = 0;
    return 0;
  }
  void close()
  {
    if(m_data)
    {
      free(m_data);
    }
    m_data      = 0;
    m_allocated = 0;
    m_used      = 0;
    m_cur       = 0;
  }
  void seek(size_t offset, int pos)
  {
    switch(pos)
    {
      case SEEK_CUR:
        m_cur += offset;
        break;
      case SEEK_SET:
        m_cur = offset;
        break;
      case SEEK_END:
        m_cur = m_used;
        break;
    }
  }
  void write(const void* data, size_t dataSize)
  {
    if(m_cur + dataSize > m_used)
    {
      m_used = m_cur + dataSize;
    }

    if(m_cur + dataSize > m_allocated)
    {
      size_t add = m_allocated * 2;
      if(add < dataSize)
        add = dataSize;

      size_t chunk = 1024 * 1024 * 128;
      if(add > chunk && dataSize < chunk)
      {
        add = chunk;
      }
      m_data = (char*)realloc(m_data, m_allocated + add);
      m_allocated += add;
    }
    memcpy(m_data + m_cur, data, dataSize);
    m_cur += dataSize;
  }
};


#if CSF_ZIP_SUPPORT
struct OutputGZ
{
  gzFile    m_file;
  OutputBuf m_buf;

  int open(const char* filename)
  {
    m_buf.open(filename);
    m_file = gzopen(filename, "wb");
    return m_file == 0;
  }
  void close()
  {
    gzwrite(m_file, m_buf.m_data, (z_off_t)m_buf.m_used);
    gzclose(m_file);
    m_buf.close();
  }
  void seek(size_t offset, int pos) { m_buf.seek(offset, pos); }
  void write(const void* data, size_t dataSize) { m_buf.write(data, dataSize); }
};
#endif

template <class T>
struct CSFOffsetMgr
{
  struct Entry
  {
    CSFoffset offset;
    CSFoffset location;
  };
  T&                 m_file;
  std::vector<Entry> m_offsetLocations;
  size_t             m_current;


  CSFOffsetMgr(T& file)
      : m_current(0)
      , m_file(file)
  {
  }

  size_t store(const void* data, size_t dataSize)
  {
    size_t last = m_current;
    m_file.write(data, dataSize);

    m_current += dataSize;
    return last;
  }

  size_t store(size_t location, const void* data, size_t dataSize)
  {
    size_t last = m_current;
    m_file.write(data, dataSize);

    m_current += dataSize;

    Entry entry = {last, location};
    m_offsetLocations.push_back(entry);

    return last;
  }

  void finalize(size_t tableCountLocation, size_t tableLocation)
  {
    m_file.seek(tableCountLocation, SEEK_SET);
    int num = int(m_offsetLocations.size());
    m_file.write(&num, sizeof(int));

    CSFoffset offset = (CSFoffset)m_current;
    m_file.seek(tableLocation, SEEK_SET);
    m_file.write(&offset, sizeof(CSFoffset));

    for(size_t i = 0; i < m_offsetLocations.size(); i++)
    {
      m_file.seek(m_offsetLocations[i].location, SEEK_SET);
      m_file.write(&m_offsetLocations[i].offset, sizeof(CSFoffset));
    }

    // dump table
    m_file.seek(0, SEEK_END);
    for(size_t i = 0; i < m_offsetLocations.size(); i++)
    {
      m_file.write(&m_offsetLocations[i].location, sizeof(CSFoffset));
    }
  }
};

template <class T>
static int CSFile_saveInternal(const CSFile* csf, const char* filename)
{
  T file;
  if(file.open(filename))
  {
    return CADSCENEFILE_ERROR_NOFILE;
  }

  CSFOffsetMgr<T> mgr(file);

  CSFile dump = {0};
  memcpy(&dump, csf, CSFile_getHeaderSize(csf));

  dump.version = CADSCENEFILE_VERSION;
  dump.magic   = CADSCENEFILE_MAGIC;
  // dump main part as is
  mgr.store(&dump, sizeof(CSFile));

  // iterate the objects

  {
    size_t geomOFFSET = mgr.store(offsetof(CSFile, geometriesOFFSET), csf->geometries, sizeof(CSFGeometry) * csf->numGeometries);

    for(int i = 0; i < csf->numGeometries; i++, geomOFFSET += sizeof(CSFGeometry))
    {
      const CSFGeometry* geo = csf->geometries + i;

      if(geo->vertex && geo->numVertices)
      {
        mgr.store(geomOFFSET + offsetof(CSFGeometry, vertexOFFSET), geo->vertex, sizeof(float) * 3 * geo->numVertices);
      }
      if(geo->normal && geo->numVertices)
      {
        mgr.store(geomOFFSET + offsetof(CSFGeometry, normalOFFSET), geo->normal,
                  sizeof(float) * 3 * geo->numVertices * geo->numNormalChannels);
      }
      if(geo->tex && geo->numVertices)
      {
        mgr.store(geomOFFSET + offsetof(CSFGeometry, texOFFSET), geo->tex, sizeof(float) * 2 * geo->numVertices * geo->numTexChannels);
      }

      if(geo->aux && geo->numVertices)
      {
        mgr.store(geomOFFSET + offsetof(CSFGeometry, auxOFFSET), geo->aux, sizeof(float) * 4 * geo->numVertices * geo->numAuxChannels);
      }
      if(geo->auxStorageOrder && geo->numAuxChannels)
      {
        mgr.store(geomOFFSET + offsetof(CSFGeometry, auxStorageOrderOFFSET), geo->auxStorageOrder,
                  sizeof(CSFGeometryAuxChannel) * geo->numAuxChannels);
      }

      if(geo->indexSolid && geo->numIndexSolid)
      {
        mgr.store(geomOFFSET + offsetof(CSFGeometry, indexSolidOFFSET), geo->indexSolid, sizeof(int) * geo->numIndexSolid);
      }
      if(geo->indexWire && geo->numIndexWire)
      {
        mgr.store(geomOFFSET + offsetof(CSFGeometry, indexWireOFFSET), geo->indexWire, sizeof(int) * geo->numIndexWire);
      }

      if(geo->perpartStorageOrder && geo->numPartChannels)
      {
        mgr.store(geomOFFSET + offsetof(CSFGeometry, perpartStorageOrder), geo->perpartStorageOrder,
                  sizeof(CSFGeometryPartChannel) * geo->numPartChannels);
      }
      if(geo->perpart && geo->numPartChannels)
      {
        mgr.store(geomOFFSET + offsetof(CSFGeometry, perpart), geo->perpart, CSFGeometry_getPerPartSize(geo));
      }

      if(geo->parts && geo->numParts)
      {
        mgr.store(geomOFFSET + offsetof(CSFGeometry, partsOFFSET), geo->parts, sizeof(CSFGeometryPart) * geo->numParts);
      }
    }
  }


  {
    size_t matOFFSET = mgr.store(offsetof(CSFile, materialsOFFSET), csf->materials, sizeof(CSFMaterial) * csf->numMaterials);

    for(int i = 0; i < csf->numMaterials; i++, matOFFSET += sizeof(CSFMaterial))
    {
      const CSFMaterial* mat = csf->materials + i;
      if(mat->bytes && mat->numBytes)
      {
        mgr.store(matOFFSET + offsetof(CSFMaterial, bytesOFFSET), mat->bytes, sizeof(unsigned char) * mat->numBytes);
      }
    }
  }

  {
    size_t nodeOFFSET = mgr.store(offsetof(CSFile, nodesOFFSET), csf->nodes, sizeof(CSFNode) * csf->numNodes);

    for(int i = 0; i < csf->numNodes; i++, nodeOFFSET += sizeof(CSFNode))
    {
      const CSFNode* node = csf->nodes + i;
      if(node->parts && node->numParts)
      {
        mgr.store(nodeOFFSET + offsetof(CSFNode, partsOFFSET), node->parts, sizeof(CSFNodePart) * node->numParts);
      }
      if(node->children && node->numChildren)
      {
        mgr.store(nodeOFFSET + offsetof(CSFNode, childrenOFFSET), node->children, sizeof(int) * node->numChildren);
      }
    }
  }

  if(CSFile_getNodeMetas(csf))
  {
    size_t metaOFFSET = mgr.store(offsetof(CSFile, nodeMetasOFFSET), csf->nodeMetas, sizeof(CSFMeta) * csf->numNodes);

    for(int i = 0; i < csf->numNodes; i++, metaOFFSET += sizeof(CSFMeta))
    {
      const CSFMeta* meta = csf->nodeMetas + i;
      if(meta->bytes && meta->numBytes)
      {
        mgr.store(metaOFFSET + offsetof(CSFMeta, bytesOFFSET), meta->bytes, sizeof(unsigned char) * meta->numBytes);
      }
    }
  }

  if(CSFile_getGeometryMetas(csf))
  {
    size_t metaOFFSET = mgr.store(offsetof(CSFile, geometryMetasOFFSET), csf->geometryMetas, sizeof(CSFMeta) * csf->numGeometries);

    for(int i = 0; i < csf->numNodes; i++, metaOFFSET += sizeof(CSFMeta))
    {
      const CSFMeta* meta = csf->geometryMetas + i;
      if(meta->bytes && meta->numBytes)
      {
        mgr.store(metaOFFSET + offsetof(CSFMeta, bytesOFFSET), meta->bytes, sizeof(unsigned char) * meta->numBytes);
      }
    }
  }

  if(CSFile_getFileMeta(csf))
  {
    size_t metaOFFSET = mgr.store(offsetof(CSFile, fileMetaOFFSET), csf->fileMeta, sizeof(CSFMeta));

    {
      const CSFMeta* meta = csf->fileMeta;
      if(meta->bytes && meta->numBytes)
      {
        mgr.store(metaOFFSET + offsetof(CSFMeta, bytesOFFSET), meta->bytes, sizeof(unsigned char) * meta->numBytes);
      }
    }
  }

  mgr.finalize(offsetof(CSFile, numPointers), offsetof(CSFile, pointersOFFSET));

  file.close();

  return CADSCENEFILE_NOERROR;
}

CSFAPI int CSFile_save(const CSFile* csf, const char* filename)
{
  return CSFile_saveInternal<OutputFILE>(csf, filename);
}

#if CSF_ZIP_SUPPORT
CSFAPI int CSFile_saveExt(CSFile* csf, const char* filename)
{
  size_t len = strlen(filename);
  if(strcmp(filename + len - 3, ".gz") == 0)
  {
    return CSFile_saveInternal<OutputGZ>(csf, filename);
  }
  else
  {
    return CSFile_saveInternal<OutputFILE>(csf, filename);
  }
}

#endif

static NV_FORCE_INLINE void Matrix44Copy(float* NV_RESTRICT dst, const float* NV_RESTRICT a)
{
  memcpy(dst, a, sizeof(float) * 16);
}

static NV_FORCE_INLINE void Matrix44MultiplyFull(float* NV_RESTRICT clip, const float* NV_RESTRICT proj, const float* NV_RESTRICT modl)
{

  clip[0] = modl[0] * proj[0] + modl[1] * proj[4] + modl[2] * proj[8] + modl[3] * proj[12];
  clip[1] = modl[0] * proj[1] + modl[1] * proj[5] + modl[2] * proj[9] + modl[3] * proj[13];
  clip[2] = modl[0] * proj[2] + modl[1] * proj[6] + modl[2] * proj[10] + modl[3] * proj[14];
  clip[3] = modl[0] * proj[3] + modl[1] * proj[7] + modl[2] * proj[11] + modl[3] * proj[15];

  clip[4] = modl[4] * proj[0] + modl[5] * proj[4] + modl[6] * proj[8] + modl[7] * proj[12];
  clip[5] = modl[4] * proj[1] + modl[5] * proj[5] + modl[6] * proj[9] + modl[7] * proj[13];
  clip[6] = modl[4] * proj[2] + modl[5] * proj[6] + modl[6] * proj[10] + modl[7] * proj[14];
  clip[7] = modl[4] * proj[3] + modl[5] * proj[7] + modl[6] * proj[11] + modl[7] * proj[15];

  clip[8]  = modl[8] * proj[0] + modl[9] * proj[4] + modl[10] * proj[8] + modl[11] * proj[12];
  clip[9]  = modl[8] * proj[1] + modl[9] * proj[5] + modl[10] * proj[9] + modl[11] * proj[13];
  clip[10] = modl[8] * proj[2] + modl[9] * proj[6] + modl[10] * proj[10] + modl[11] * proj[14];
  clip[11] = modl[8] * proj[3] + modl[9] * proj[7] + modl[10] * proj[11] + modl[11] * proj[15];

  clip[12] = modl[12] * proj[0] + modl[13] * proj[4] + modl[14] * proj[8] + modl[15] * proj[12];
  clip[13] = modl[12] * proj[1] + modl[13] * proj[5] + modl[14] * proj[9] + modl[15] * proj[13];
  clip[14] = modl[12] * proj[2] + modl[13] * proj[6] + modl[14] * proj[10] + modl[15] * proj[14];
  clip[15] = modl[12] * proj[3] + modl[13] * proj[7] + modl[14] * proj[11] + modl[15] * proj[15];
}

static void CSFile_transformHierarchy(CSFile* csf, CSFNode* NV_RESTRICT node, CSFNode* NV_RESTRICT parent)
{
  if(parent)
  {
    Matrix44MultiplyFull(node->worldTM, parent->worldTM, node->objectTM);
  }
  else
  {
    Matrix44Copy(node->worldTM, node->objectTM);
  }

  for(int i = 0; i < node->numChildren; i++)
  {
    CSFNode* NV_RESTRICT child = csf->nodes + node->children[i];
    CSFile_transformHierarchy(csf, child, node);
  }
}

CSFAPI int CSFile_transform(CSFile* csf)
{
  if(!(csf->fileFlags & CADSCENEFILE_FLAG_UNIQUENODES))
    return CADSCENEFILE_ERROR_OPERATION;

  CSFile_transformHierarchy(csf, csf->nodes + csf->rootIDX, nullptr);
  return CADSCENEFILE_NOERROR;
}

CSFAPI const CSFMeta* CSFile_getNodeMetas(const CSFile* csf)
{
  if(csf->version >= CADSCENEFILE_VERSION_META && csf->fileFlags & CADSCENEFILE_FLAG_META_NODE)
  {
    return csf->nodeMetas;
  }

  return nullptr;
}

CSFAPI const CSFMeta* CSFile_getGeometryMetas(const CSFile* csf)
{
  if(csf->version >= CADSCENEFILE_VERSION_META && csf->fileFlags & CADSCENEFILE_FLAG_META_GEOMETRY)
  {
    return csf->geometryMetas;
  }

  return nullptr;
}


CSFAPI const CSFMeta* CSFile_getFileMeta(const CSFile* csf)
{
  if(csf->version >= CADSCENEFILE_VERSION_META && csf->fileFlags & CADSCENEFILE_FLAG_META_FILE)
  {
    return csf->fileMeta;
  }

  return nullptr;
}


CSFAPI const CSFBytePacket* CSFile_getBytePacket(const unsigned char* bytes, CSFoffset numBytes, CSFGuid guid)
{
  if(numBytes < sizeof(CSFBytePacket))
    return nullptr;

  do
  {
    const CSFBytePacket* packet = (const CSFBytePacket*)bytes;
    if(memcmp(guid, packet->guid, sizeof(CSFGuid)) == 0)
    {
      return packet;
    }
    numBytes -= packet->numBytes;
    bytes += packet->numBytes;

  } while(numBytes >= sizeof(CSFBytePacket));

  return nullptr;
}

CSFAPI const CSFBytePacket* CSFile_getMetaBytePacket(const CSFMeta* meta, CSFGuid guid)
{
  return CSFile_getBytePacket(meta->bytes, meta->numBytes, guid);
}

CSFAPI const CSFBytePacket* CSFile_getMaterialBytePacket(const CSFile* csf, int materialIDX, CSFGuid guid)
{
  if(materialIDX < 0 || materialIDX >= csf->numMaterials)
  {
    return nullptr;
  }

  return CSFile_getBytePacket(csf->materials[materialIDX].bytes, csf->materials[materialIDX].numBytes, guid);
}

CSFAPI void CSFMatrix_identity(float* matrix)
{
  memset(matrix, 0, sizeof(float) * 16);
  matrix[0] = matrix[5] = matrix[10] = matrix[15] = 1.0f;
}

CSFAPI void CSFile_clearDeprecated(CSFile* csf)
{
  for(int g = 0; g < csf->numGeometries; g++)
  {
    memset(csf->geometries[g]._deprecated, 0, sizeof(csf->geometries[g]._deprecated));
    for(int p = 0; p < csf->geometries[g].numParts; p++)
    {
      csf->geometries[g].parts[p]._deprecated = 0;
    }
  }
}

CSFAPI void CSFGeometry_setupDefaultChannels(CSFGeometry* geo)
{
  geo->numNormalChannels = geo->normal ? 1 : 0;
  geo->numTexChannels    = geo->tex ? 1 : 0;
  geo->numAuxChannels    = 0;
  geo->numPartChannels   = 0;
  geo->aux               = nullptr;
  geo->auxStorageOrder   = nullptr;
  geo->perpart           = nullptr;
}

CSFAPI void CSFile_setupDefaultChannels(CSFile* csf)
{
  for(int g = 0; g < csf->numGeometries; g++)
  {
    CSFGeometry_setupDefaultChannels(csf->geometries + g);
  }
}

CSFAPI const float* CSFGeometry_getNormalChannel(const CSFGeometry* geo, CSFGeometryNormalChannel channel)
{
  return channel < geo->numNormalChannels ? geo->normal + size_t(geo->numVertices * 3 * channel) : nullptr;
}

CSFAPI const float* CSFGeometry_getTexChannel(const CSFGeometry* geo, CSFGeometryTexChannel channel)
{
  return channel < geo->numTexChannels ? geo->tex + size_t(geo->numVertices * 2 * channel) : nullptr;
}

CSFAPI const float* CSFGeometry_getAuxChannel(const CSFGeometry* geo, CSFGeometryAuxChannel channel)
{
  for(int i = 0; i < geo->numAuxChannels; i++)
  {
    if(geo->auxStorageOrder[i] == channel)
    {
      return geo->aux + size_t(geo->numVertices * 4 * i);
    }
  }

  return nullptr;
}

CSFAPI size_t CSFGeometryPartChannel_getSize(CSFGeometryPartChannel channel)
{
  size_t sizes[CSFGEOMETRY_PARTCHANNELS];
  sizes[CSFGEOMETRY_PARTCHANNEL_BBOX] = sizeof(CSFGeometryPartBbox);

  return sizes[channel];
}

CSFAPI size_t CSFGeometry_getPerPartSize(const CSFGeometry* geo)
{
  size_t size = 0;
  for(int i = 0; i < geo->numPartChannels; i++)
  {
    size += CSFGeometryPartChannel_getSize(geo->perpartStorageOrder[i]) * geo->numParts;
  }
  return size;
}

CSFAPI size_t CSFGeometry_getPerPartRequiredSize(const CSFGeometry* geo, int numParts)
{
  size_t size = 0;
  for(int i = 0; i < geo->numPartChannels; i++)
  {
    size += CSFGeometryPartChannel_getSize(geo->perpartStorageOrder[i]) * numParts;
  }
  return size;
}

CSFAPI size_t CSFGeometry_getPerPartRequiredOffset(const CSFGeometry* geo, int numParts, CSFGeometryPartChannel channel)
{
  size_t offset = 0;
  for(int i = 0; i < geo->numPartChannels; i++)
  {
    if(geo->perpartStorageOrder[i] == channel)
    {
      return offset;
    }
    offset += CSFGeometryPartChannel_getSize(geo->perpartStorageOrder[i]) * numParts;
  }
  return ~0ull;
}

CSFAPI const void* CSFGeometry_getPartChannel(const CSFGeometry* geo, CSFGeometryPartChannel channel)
{
  size_t offset = CSFGeometry_getPerPartRequiredOffset(geo, geo->numParts, channel);
  if(offset != ~0ull)
  {
    return geo->perpart + offset;
  }

  return nullptr;
}


#if CSF_GLTF2_SUPPORT

#include <unordered_map>
#include "cgltf.h"
#include <nvmath/nvmath.h>

void CSFile_countGLTFNodes(CSFile* csf, const cgltf_data* gltfModel, const cgltf_node* node)
{
  csf->numNodes++;
  for (cgltf_size i = 0; i < node->children_count; i++)
  {
    CSFile_countGLTFNodes(csf, gltfModel, node->children[i]);
  }
}

int CSFile_addGLTFNode(CSFile* csf, const cgltf_data* gltfModel, const uint32_t* meshGeometries, CSFileMemoryPTR mem, const cgltf_node* node)
{
  int idx = csf->numNodes++;
  CSFNode& csfnode = csf->nodes[idx];

  CSFMatrix_identity(csfnode.worldTM);
  CSFMatrix_identity(csfnode.objectTM);

  if (node->has_matrix)
  {
    for (int i = 0; i < 16; i++)
    {
      csfnode.objectTM[i] = (float)node->matrix[i];
    }
  }
  else
  {
    nvmath::vec3f translation = { 0, 0, 0 };
    nvmath::quatf rotation = { 0, 0, 0, 0 };
    nvmath::vec3f scale = { 1, 1, 1 };

    if (node->has_translation)
    {
      translation.x = static_cast<float>(node->translation[0]);
      translation.y = static_cast<float>(node->translation[1]);
      translation.z = static_cast<float>(node->translation[2]);
    }

    if (node->has_rotation)
    {
      rotation.x = static_cast<float>(node->rotation[0]);
      rotation.y = static_cast<float>(node->rotation[1]);
      rotation.z = static_cast<float>(node->rotation[2]);
      rotation.w = static_cast<float>(node->rotation[3]);
    }

    if (node->has_scale)
    {
      scale.x = static_cast<float>(node->scale[0]);
      scale.y = static_cast<float>(node->scale[1]);
      scale.z = static_cast<float>(node->scale[2]);
    }


    nvmath::mat4f mtranslation, mscale, mrot;
    nvmath::quatf mrotation;
    mtranslation.as_translation(translation);
    mscale.as_scale(scale);
    rotation.to_matrix(mrot);

    nvmath::mat4f matrix = mtranslation * mrot * mscale;
    for (int i = 0; i < 16; i++)
    {
      csfnode.objectTM[i] = matrix.mat_array[i];
    }
  }

  // setup geometry
  if (node->mesh)
  {
    size_t meshIndex = node->mesh - gltfModel->meshes;

    csfnode.geometryIDX = meshGeometries[meshIndex];
    const cgltf_mesh& mesh = gltfModel->meshes[meshIndex];

    csfnode.numParts = csf->geometries[csfnode.geometryIDX].numParts;
    csfnode.parts = (CSFNodePart*)CSFileMemory_alloc(mem, sizeof(CSFNodePart) * csfnode.numParts, nullptr);

    uint32_t p = 0;
    for (cgltf_size i = 0; i < mesh.primitives_count; i++)
    {
      const cgltf_primitive& primitive = mesh.primitives[i];

      if (primitive.type != cgltf_primitive_type_triangles)
        continue;

      CSFNodePart& csfpart = csfnode.parts[p++];

      csfpart.active = 1;
      csfpart.materialIDX = primitive.material ? int(primitive.material - gltfModel->materials) : 0;
      csfpart.nodeIDX = -1;
    }
  }
  else
  {
    csfnode.geometryIDX = -1;
  }

  csfnode.numChildren = (int)node->children_count;
  csfnode.children = (int*)CSFileMemory_alloc(mem, sizeof(int) * csfnode.numChildren, nullptr);

  for (cgltf_size i = 0; i < node->children_count; i++)
  {
    csfnode.children[i] = CSFile_addGLTFNode(csf, gltfModel, meshGeometries, mem, node->children[i]);
  }

  return idx;
}

//-----------------------------------------------------------------------------
// MurmurHash2A, by Austin Appleby

// This is a variant of MurmurHash2 modified to use the Merkle-Damgard
// construction. Bulk speed should be identical to Murmur2, small-key speed
// will be 10%-20% slower due to the added overhead at the end of the hash.

// This variant fixes a minor issue where null keys were more likely to
// collide with each other than expected, and also makes the algorithm
// more amenable to incremental implementations. All other caveats from
// MurmurHash2 still apply.

#define mmix(h, k)                                                                                                     \
  {                                                                                                                    \
    k *= m;                                                                                                            \
    k ^= k >> r;                                                                                                       \
    k *= m;                                                                                                            \
    h *= m;                                                                                                            \
    h ^= k;                                                                                                            \
  }

static unsigned int strMurmurHash2A(const void* key, size_t len, unsigned int seed)
{
  const unsigned int m = 0x5bd1e995;
  const int          r = 24;
  unsigned int       l = (unsigned int)len;

  const unsigned char* data = (const unsigned char*)key;

  unsigned int h = seed;
  unsigned int t = 0;

  while (len >= 4)
  {
    unsigned int k = *(unsigned int*)data;

    mmix(h, k);

    data += 4;
    len -= 4;
  }


  switch (len)
  {
  case 3:
    t ^= data[2] << 16;
  case 2:
    t ^= data[1] << 8;
  case 1:
    t ^= data[0];
  };

  mmix(h, t);
  mmix(h, l);

  h ^= h >> 13;
  h *= m;
  h ^= h >> 15;

  return h;
}
#undef mmix

struct GLTFGeometryInfo
{
  uint32_t numVertices = 0;
  uint32_t numNormals = 0;
  uint32_t numTexcoords = 0;
  uint32_t numIndices = 0;
  uint32_t numParts = 0;

  uint32_t hashIndex = 0;
  uint32_t hashVertex = 0;
  uint32_t hashNormal = 0;
  uint32_t hashTexcoord = 0;

  uint32_t hashLightVertex = 0;
  uint32_t hashLightNormal = 0;
  uint32_t hashLightTexcoord = 0;

  bool isEqualHash(const GLTFGeometryInfo& other)
  {
    return hashIndex == other.hashIndex && hashVertex == other.hashVertex && hashNormal == other.hashNormal && hashTexcoord == other.hashTexcoord;
  }

  bool isEqualLight(const GLTFGeometryInfo& other)
  {
    return numVertices == other.numVertices && numNormals == other.numNormals && numIndices == other.numIndices
      && numParts == other.numParts && hashLightVertex == other.hashLightVertex && hashLightNormal == other.hashLightNormal && hashLightTexcoord == other.hashLightTexcoord;
  }

  void setup(const cgltf_data* gltfModel, const cgltf_mesh& mesh)
  {
    hashVertex = 0;
    hashNormal = 0;
    hashTexcoord = 0;
    hashIndex = 0;

    hashLightVertex = 0;
    hashLightNormal = 0;
    hashLightTexcoord = 0;

    for (cgltf_size i = 0; i < mesh.primitives_count; i++)
    {
      const cgltf_primitive& primitive = mesh.primitives[i];

      if (primitive.type != cgltf_primitive_type_triangles)
        continue;

      for (cgltf_size a = 0; a < primitive.attributes_count; a++) {
        const cgltf_accessor* accessor = primitive.attributes[a].data;
        const cgltf_buffer_view* view = accessor->buffer_view;
        const uint8_t* data = reinterpret_cast<const uint8_t*>(view->buffer->data);
        data += accessor->offset + view->offset;

        switch (primitive.attributes[a].type) {
        case cgltf_attribute_type_position:
          numVertices += static_cast<uint32_t>(accessor->count);
          hashLightVertex = strMurmurHash2A(data, accessor->stride, hashLightVertex);
          break;
        case cgltf_attribute_type_normal:
          numNormals += static_cast<uint32_t>(accessor->count);
          hashLightNormal = strMurmurHash2A(data, accessor->stride, hashLightNormal);
          break;
        case cgltf_attribute_type_texcoord:
          if (primitive.attributes[a].index == 0) {
            numTexcoords += static_cast<uint32_t>(accessor->count);
            hashLightTexcoord = strMurmurHash2A(data, accessor->stride, hashLightTexcoord);
          }
          break;
        }
      }

      numIndices += static_cast<uint32_t>(primitive.indices->count);
      numParts++;
    }
  }

  bool hasHash() const { return hashIndex != 0 || hashVertex != 0 || hashNormal != 0; }

  void setupHash(const cgltf_data* gltfModel, const cgltf_mesh& mesh)
  {
    for (cgltf_size i = 0; i < mesh.primitives_count; i++)
    {
      const cgltf_primitive& primitive = mesh.primitives[i];

      if (primitive.type != cgltf_primitive_type_triangles)
        continue;

      for (cgltf_size a = 0; a < primitive.attributes_count; a++) {
        const cgltf_accessor* accessor = primitive.attributes[a].data;
        const cgltf_buffer_view* view = accessor->buffer_view;
        const uint8_t* data = reinterpret_cast<const uint8_t*>(view->buffer->data);
        data += accessor->offset + view->offset;

        switch (primitive.attributes[a].type) {
        case cgltf_attribute_type_position:
          hashVertex = strMurmurHash2A(data, accessor->stride * accessor->count, hashVertex);
          break;
        case cgltf_attribute_type_normal:
          hashNormal = strMurmurHash2A(data, accessor->stride * accessor->count, hashNormal);
          break;
        case cgltf_attribute_type_texcoord:
          if (primitive.attributes[a].index == 0) {
            hashTexcoord = strMurmurHash2A(data, accessor->stride * accessor->count, hashTexcoord);
          }
          break;
        }
      }

      {
        const cgltf_accessor* accessor = primitive.indices;
        const cgltf_buffer_view* view = accessor->buffer_view;
        const uint8_t* data = reinterpret_cast<const uint8_t*>(view->buffer->data);
        data += accessor->offset + view->offset;

        hashIndex = strMurmurHash2A(data, accessor->stride * accessor->count, hashIndex);
      }
    }
  }
};

static inline void setupCSFMaterialTexture(CSFMaterialGLTF2Texture& csftex, const cgltf_texture_view& tex)
{
  if (!tex.texture) return;

  const char* uri = tex.texture->image->uri;
  if (uri) {
    strncpy(csftex.name, uri, sizeof(csftex.name));
  }
}

struct MappingList {
  std::unordered_map<std::string, nvh::FileReadMapping> maps;
};

static cgltf_result csf_cgltf_read(const struct cgltf_memory_options* memory_options, const struct cgltf_file_options* file_options, const char* path, cgltf_size* size, void** data)
{
  MappingList* mappings = (MappingList*)file_options->user_data;
  std::string pathStr(path);

  auto it = mappings->maps.find(pathStr);
  if (it != mappings->maps.end()) {
    *data = const_cast<void*>(it->second.data());
    *size = it->second.size();
    return cgltf_result_success;
  }

  nvh::FileReadMapping map;
  if (map.open(path)) {
    *data = const_cast<void*>(map.data());
    *size = map.size();
    mappings->maps.insert({ pathStr,std::move(map) });
    return cgltf_result_success;
  }

  return cgltf_result_io_error;
}

static void csf_cgltf_release(const struct cgltf_memory_options* memory_options, const struct cgltf_file_options* file_options, void* data)
{
  // let MappingList destructor handle it
}

CSFAPI int CSFile_loadGTLF(CSFile** outcsf, const char* filename, CSFileMemoryPTR mem)
{
  if (!filename)
  {
    return CADSCENEFILE_ERROR_NOFILE;
  }

  int findUniqueGeometries = mem->m_config.gltfFindUniqueGeometries;

  cgltf_options       gltfOptions = {};
  cgltf_data*         gltfModel;

  MappingList     mappings;

  gltfOptions.file.read = csf_cgltf_read;
  gltfOptions.file.release = csf_cgltf_release;
  gltfOptions.file.user_data = &mappings;

  *outcsf = NULL;

  cgltf_result result = cgltf_parse_file(&gltfOptions, filename, &gltfModel);
  if (result != cgltf_result_success)
  {
    printf("ERR: cgltf_parse_file: %d\n", result);
    return CADSCENEFILE_ERROR_OPERATION;
  }

  result = cgltf_load_buffers(&gltfOptions, gltfModel, filename);
  if (result != cgltf_result_success)
  {
    printf("ERR: cgltf_load_buffers: %d\n", result);
    cgltf_free(gltfModel);
    return CADSCENEFILE_ERROR_OPERATION;
  }

  CSFile* csf = (CSFile*)CSFileMemory_alloc(mem, sizeof(CSFile), NULL);
  memset(csf, 0, sizeof(CSFile));
  csf->version = CADSCENEFILE_VERSION;
  csf->fileFlags = 0;
  csf->fileFlags |= CADSCENEFILE_FLAG_UNIQUENODES;
  csf->numMaterials = (int)gltfModel->materials_count;
  csf->numNodes = (int)gltfModel->nodes_count;

  csf->materials = (CSFMaterial*)CSFileMemory_alloc(mem, sizeof(CSFMaterial) * csf->numMaterials, NULL);
  memset(csf->materials, 0, sizeof(CSFMaterial) * csf->numMaterials);

  // create materials
  for (cgltf_size materialIdx = 0; materialIdx < gltfModel->materials_count; materialIdx++)
  {
    const cgltf_material& mat = gltfModel->materials[materialIdx];
    CSFMaterial& csfmat = csf->materials[materialIdx];

    if (mat.has_pbr_metallic_roughness)
    {
      csfmat.color[0] = mat.pbr_metallic_roughness.base_color_factor[0];
      csfmat.color[1] = mat.pbr_metallic_roughness.base_color_factor[1];
      csfmat.color[2] = mat.pbr_metallic_roughness.base_color_factor[2];
      csfmat.color[3] = mat.pbr_metallic_roughness.base_color_factor[3];
    }
    else if (mat.has_pbr_specular_glossiness)
    {
      csfmat.color[0] = mat.pbr_specular_glossiness.diffuse_factor[0];
      csfmat.color[1] = mat.pbr_specular_glossiness.diffuse_factor[1];
      csfmat.color[2] = mat.pbr_specular_glossiness.diffuse_factor[2];
      csfmat.color[3] = mat.pbr_specular_glossiness.diffuse_factor[3];
    }
    else
    {
      csfmat.color[0] = 1.0f;
      csfmat.color[1] = 1.0f;
      csfmat.color[2] = 1.0f;
      csfmat.color[3] = 1.0f;
    }

    strncpy(csfmat.name, mat.name, sizeof(csfmat.name));
    csfmat.bytes = nullptr;
    csfmat.numBytes = 0;
    csfmat.type = 0;

    CSFMaterialGLTF2Meta csfmatgltf = { {CSFGUID_MATERIAL_GLTF2, sizeof(CSFMaterialGLTF2Meta)} };

    csfmatgltf.shadingModel = -1;
    csfmatgltf.emissiveFactor[0] = mat.emissive_factor[0];
    csfmatgltf.emissiveFactor[1] = mat.emissive_factor[1];
    csfmatgltf.emissiveFactor[2] = mat.emissive_factor[2];
    csfmatgltf.doubleSided = mat.double_sided ? 1 : 0;
    csfmatgltf.alphaCutoff = mat.alpha_cutoff;
    csfmatgltf.alphaMode = mat.alpha_mode;

    setupCSFMaterialTexture(csfmatgltf.emissiveTexture, mat.emissive_texture);
    setupCSFMaterialTexture(csfmatgltf.normalTexture, mat.normal_texture);
    setupCSFMaterialTexture(csfmatgltf.occlusionTexture, mat.occlusion_texture);

    if (mat.has_pbr_metallic_roughness) {
      csfmatgltf.shadingModel = mat.unlit ? -1 : 0;
      csfmatgltf.baseColorFactor[0] = mat.pbr_metallic_roughness.base_color_factor[0];
      csfmatgltf.baseColorFactor[1] = mat.pbr_metallic_roughness.base_color_factor[1];
      csfmatgltf.baseColorFactor[2] = mat.pbr_metallic_roughness.base_color_factor[2];
      csfmatgltf.baseColorFactor[3] = mat.pbr_metallic_roughness.base_color_factor[3];
      csfmatgltf.roughnessFactor = mat.pbr_metallic_roughness.roughness_factor;
      csfmatgltf.metallicFactor = mat.pbr_metallic_roughness.metallic_factor;
      setupCSFMaterialTexture(csfmatgltf.baseColorTexture, mat.pbr_metallic_roughness.base_color_texture);
      setupCSFMaterialTexture(csfmatgltf.metallicRoughnessTexture, mat.pbr_metallic_roughness.metallic_roughness_texture);
    }
    else if (mat.has_pbr_specular_glossiness) {
      csfmatgltf.shadingModel = 1;
      csfmatgltf.diffuseFactor[0] = mat.pbr_specular_glossiness.diffuse_factor[0];
      csfmatgltf.diffuseFactor[1] = mat.pbr_specular_glossiness.diffuse_factor[1];
      csfmatgltf.diffuseFactor[2] = mat.pbr_specular_glossiness.diffuse_factor[2];
      csfmatgltf.diffuseFactor[3] = mat.pbr_specular_glossiness.diffuse_factor[3];
      csfmatgltf.glossinessFactor = mat.pbr_specular_glossiness.glossiness_factor;
      csfmatgltf.specularFactor[0] = mat.pbr_specular_glossiness.specular_factor[0];
      csfmatgltf.specularFactor[1] = mat.pbr_specular_glossiness.specular_factor[1];
      csfmatgltf.specularFactor[2] = mat.pbr_specular_glossiness.specular_factor[2];
      setupCSFMaterialTexture(csfmatgltf.diffuseTexture, mat.pbr_specular_glossiness.diffuse_texture);
      setupCSFMaterialTexture(csfmatgltf.specularGlossinessTexture, mat.pbr_specular_glossiness.specular_glossiness_texture);
    }

    csfmat.numBytes = sizeof(csfmatgltf);
    csfmat.bytes = (unsigned char*)CSFileMemory_alloc(mem, sizeof(csfmatgltf), &csfmatgltf);
  }

  // find unique geometries
  // many gltf files make improper use of geometry instancing

  std::vector<uint32_t>         meshGeometries;
  std::vector<uint32_t>         geometryMeshes;

  meshGeometries.reserve(gltfModel->meshes_count);
  geometryMeshes.reserve(gltfModel->meshes_count);


  if (findUniqueGeometries) {
    // use some hashing based comparisons to avoid deep comparisons

    std::vector<GLTFGeometryInfo> geometryInfos;
    geometryInfos.reserve(gltfModel->meshes_count);

    uint32_t meshIdx = 0;
    for (cgltf_size m = 0; m < gltfModel->meshes_count; m++)
    {
      const cgltf_mesh& mesh = gltfModel->meshes[m];
      GLTFGeometryInfo geoInfo;

      geoInfo.setup(gltfModel, mesh);

      // compare against existing hashes
      uint32_t found = ~0;
      for (uint32_t i = 0; i < (uint32_t)geometryInfos.size(); i++)
      {
        if (geoInfo.isEqualLight(geometryInfos[i]))
        {
          if (!geometryInfos[i].hasHash())
          {
            geometryInfos[i].setupHash(gltfModel, gltfModel->meshes[geometryMeshes[i]]);
          }

          geoInfo.setupHash(gltfModel, mesh);

          if (geoInfo.isEqualHash(geometryInfos[i]))
          {
            found = i;
            break;
          }
        }
      }
      if (found != ~0)
      {
        meshGeometries.push_back(found);
      }
      else
      {
        meshGeometries.push_back((uint32_t)geometryInfos.size());
        geometryInfos.push_back(geoInfo);
        geometryMeshes.push_back(uint32_t(meshIdx));
      }
      meshIdx++;
    }
  }
  else
  {
    // 1:1 Mesh to CSFGeometry
    for (cgltf_size meshIdx = 0; meshIdx < gltfModel->meshes_count; meshIdx++)
    {
      meshGeometries.push_back(uint32_t(meshIdx));
      geometryMeshes.push_back(uint32_t(meshIdx));
    }
  }

  csf->numGeometries = (int)geometryMeshes.size();
  csf->geometries = (CSFGeometry*)CSFileMemory_alloc(mem, sizeof(CSFGeometry) * csf->numGeometries, NULL);
  memset(csf->geometries, 0, sizeof(CSFGeometry) * csf->numGeometries);


  // create geometries
#pragma omp parallel for
  for (int outIdx = 0; outIdx < csf->numGeometries; outIdx++)
  {
    const cgltf_mesh& mesh = gltfModel->meshes[geometryMeshes[outIdx]];
    CSFGeometry&   csfgeom = csf->geometries[outIdx];

    // count pass
    uint32_t vertexTotCount = 0;
    uint32_t indexTotCount = 0;
    uint32_t partsTotCount = 0;

    bool hasNormals = false;
    bool hasTexcoords = false;
    for (cgltf_size p = 0; p < mesh.primitives_count; p++)
    {
      const cgltf_primitive& primitive = mesh.primitives[p];

      if (primitive.type != cgltf_primitive_type_triangles)
        continue;

      for (cgltf_size a = 0; a < primitive.attributes_count; a++) {
        const cgltf_accessor* accessor = primitive.attributes[a].data;

        switch (primitive.attributes[a].type) {
        case cgltf_attribute_type_position:
          vertexTotCount += uint32_t(accessor->count);
          break;
        case cgltf_attribute_type_normal:
          hasNormals = true;
          break;
        case cgltf_attribute_type_texcoord:
          if (primitive.attributes[a].index == 0) {
            hasTexcoords = true;
          }
          break;
        }
      }
      indexTotCount += uint32_t(primitive.indices->count);
      partsTotCount++;
    }

    // allocate all data
    csfgeom.numVertices = vertexTotCount;
    csfgeom.numParts    = partsTotCount;

    csfgeom.vertex = (float*)CSFileMemory_alloc(mem, sizeof(float) * 3 * vertexTotCount, nullptr);
    if (hasNormals)
    {
      csfgeom.normal = (float*)CSFileMemory_alloc(mem, sizeof(float) * 3 * vertexTotCount, nullptr);
    }
    if (hasTexcoords)
    {
      csfgeom.tex = (float*)CSFileMemory_alloc(mem, sizeof(float) * 2 * vertexTotCount, nullptr);
    }
    csfgeom.indexSolid = (uint32_t*)CSFileMemory_alloc(mem, sizeof(uint32_t) * indexTotCount, nullptr);
    csfgeom.parts = (CSFGeometryPart*)CSFileMemory_alloc(mem, sizeof(CSFGeometryPart) * partsTotCount, nullptr);

    // fill pass
    indexTotCount = 0;
    vertexTotCount = 0;
    partsTotCount = 0;

    for (cgltf_size p = 0; p < mesh.primitives_count; p++)
    {
      const cgltf_primitive& primitive = mesh.primitives[p];

      if (primitive.type != cgltf_primitive_type_triangles)
        continue;

      CSFGeometryPart& csfpart = csfgeom.parts[partsTotCount++];

      uint32_t vertexCount = 0;

      for (cgltf_size a = 0; a < primitive.attributes_count; a++) {
        const cgltf_accessor* accessor = primitive.attributes[a].data;
        const cgltf_buffer_view* view = accessor->buffer_view;
        const uint8_t* data = reinterpret_cast<const uint8_t*>(view->buffer->data);
        data += accessor->offset + view->offset;

        switch (primitive.attributes[a].type) {
        case cgltf_attribute_type_position:
          vertexCount += uint32_t(accessor->count);

          for (cgltf_size i = 0; i < accessor->count; i++) {
            const float* vec = (const float*)(data + i * accessor->stride);
            csfgeom.vertex[(vertexTotCount + i) * 3 + 0] = vec[0];
            csfgeom.vertex[(vertexTotCount + i) * 3 + 1] = vec[1];
            csfgeom.vertex[(vertexTotCount + i) * 3 + 2] = vec[2];
          }
          break;
        case cgltf_attribute_type_normal:
          for (cgltf_size i = 0; i < accessor->count; i++) {
            const float* vec = (const float*)(data + i * accessor->stride);
            csfgeom.normal[(vertexTotCount + i) * 3 + 0] = vec[0];
            csfgeom.normal[(vertexTotCount + i) * 3 + 1] = vec[1];
            csfgeom.normal[(vertexTotCount + i) * 3 + 2] = vec[2];
          }
          hasNormals = true;
          break;
        case cgltf_attribute_type_texcoord:
          if (primitive.attributes[a].index == 0) {
            for (cgltf_size i = 0; i < accessor->count; i++) {
              cgltf_accessor_read_float(accessor, i, csfgeom.tex + (i + vertexTotCount) * 2, 2);
            }
          }
          break;
        }
      }

      {
        const cgltf_accessor* accessor = primitive.indices;
        const cgltf_buffer_view* view = accessor->buffer_view;
        const uint8_t* data = reinterpret_cast<const uint8_t*>(view->buffer->data);
        data += accessor->offset + view->offset;

#define checkDegenerate(index, count)   (index[count-1] == index[count-2] || index[count-2] == index[count-3] || index[count-3] == index[count-1])

        uint32_t indexBegin = indexTotCount;
        switch (accessor->component_type)
        {
        case cgltf_component_type_r_16:
          for (cgltf_size i = 0; i < accessor->count; i++) {
            const uint8_t* in = data + (i * accessor->stride);
            csfgeom.indexSolid[indexTotCount++] = *((const int16_t*)in) + vertexTotCount;
            if (i % 3 == 2 && checkDegenerate(csfgeom.indexSolid, indexTotCount))
            {
              indexTotCount -= 3;
            }
          }
          break;
        case cgltf_component_type_r_16u:
          for (cgltf_size i = 0; i < accessor->count; i++) {
            const uint8_t* in = data + (i * accessor->stride);
            csfgeom.indexSolid[indexTotCount++] = *((const uint16_t*)in) + vertexTotCount;
            if (i % 3 == 2 && checkDegenerate(csfgeom.indexSolid, indexTotCount))
            {
              indexTotCount -= 3;
            }
          }
          break;
        case cgltf_component_type_r_32u:
          for (cgltf_size i = 0; i < accessor->count; i++) {
            const uint8_t* in = data + (i * accessor->stride);
            csfgeom.indexSolid[indexTotCount++] = *((const uint32_t*)in) + vertexTotCount;
            if (i % 3 == 2 && checkDegenerate(csfgeom.indexSolid, indexTotCount))
            {
              indexTotCount -= 3;
            }
          }
          break;
        case cgltf_component_type_r_8:
          for (cgltf_size i = 0; i < accessor->count; i++) {
            const uint8_t* in = data + (i * accessor->stride);
            csfgeom.indexSolid[indexTotCount++] = *((const int8_t*)in) + vertexTotCount;
            if (i % 3 == 2 && checkDegenerate(csfgeom.indexSolid, indexTotCount))
            {
              indexTotCount -= 3;
            }
          }
          break;
        case cgltf_component_type_r_8u:
          for (cgltf_size i = 0; i < accessor->count; i++) {
            const uint8_t* in = data + (i * accessor->stride);
            csfgeom.indexSolid[indexTotCount++] = *((const uint8_t*)in) + vertexTotCount;
            if (i % 3 == 2 && checkDegenerate(csfgeom.indexSolid, indexTotCount))
            {
              indexTotCount -= 3;
            }
          }
          break;
        default:
          assert(0);
          break;
        }

        csfpart.numIndexSolid = indexTotCount - indexBegin;
      }

      vertexTotCount += vertexCount;

      csfpart.numIndexWire = 0;
      csfpart._deprecated = 0;
    }

    csfgeom.numIndexSolid = (int)indexTotCount;

    CSFGeometry_setupDefaultChannels(&csfgeom);
  }

  // create flattened nodes
  csf->numNodes = 1;  // reserve for root
  csf->rootIDX = 0;


  const cgltf_scene* scene = gltfModel->scene;
  for (size_t i = 0; i < scene->nodes_count; i++)
  {
    CSFile_countGLTFNodes(csf, gltfModel, scene->nodes[i]);
  }

  csf->nodes = (CSFNode*)CSFileMemory_alloc(mem, sizeof(CSFNode) * csf->numNodes, nullptr);
  memset(csf->nodes, 0, sizeof(CSFNode) * csf->numNodes);

  csf->numNodes = 1;
  // root setup
  csf->nodes[0].geometryIDX = -1;
  csf->nodes[0].numChildren = (int)scene->nodes_count;
  csf->nodes[0].children = (int*)CSFileMemory_alloc(mem, sizeof(int) * scene->nodes_count, nullptr);
  CSFMatrix_identity(csf->nodes[0].worldTM);
  CSFMatrix_identity(csf->nodes[0].objectTM);

  for (size_t i = 0; i < scene->nodes_count; i++)
  {
    csf->nodes[0].children[i] = CSFile_addGLTFNode(csf, gltfModel, meshGeometries.data(), mem, scene->nodes[i]);
  }

  CSFile_transform(csf);
  cgltf_free(gltfModel);

  *outcsf = csf;
  return CADSCENEFILE_NOERROR;
}


#endif
