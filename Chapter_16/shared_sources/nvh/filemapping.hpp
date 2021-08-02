/* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <cstddef>
#include <utility>

namespace nvh {

class FileMapping
{
public:

  FileMapping(FileMapping&& other)
  {
    this->operator=(std::move(other));
  };

  FileMapping& operator=(FileMapping&& other)
  {
    m_isValid     = other.m_isValid;
    m_fileSize    = other.m_fileSize;
    m_mappingType = other.m_mappingType;
    m_mappingPtr  = other.m_mappingPtr;
    m_mappingSize = other.m_mappingSize;
#ifdef _WIN32
    m_win32.file              = other.m_win32.file;
    m_win32.fileMapping       = other.m_win32.fileMapping;
    other.m_win32.file        = nullptr;
    other.m_win32.fileMapping = nullptr;
#else
    m_unix.file       = other.m_unix.file;
    other.m_unix.file = -1;
#endif
    other.m_isValid    = false;
    other.m_mappingPtr = nullptr;

    return *this;
  }

  FileMapping(const FileMapping&) = delete;
  FileMapping& operator=(const FileMapping& other) = delete;
  FileMapping() {}

  ~FileMapping() { close(); }

  enum MappingType
  {
    MAPPING_READONLY,       // opens existing file for read-only access
    MAPPING_READOVERWRITE,  // creates new file with read/write access, overwriting existing files
  };

  // fileSize only for write access
  bool open(const char* filename, MappingType mappingType, size_t fileSize = 0);
  void close();

  const void* data() const { return m_mappingPtr; }
  void*       data() { return m_mappingPtr; }
  size_t      size() const { return m_mappingSize; }
  bool        valid() const { return m_isValid; }

protected:
  static size_t g_pageSize;

#ifdef _WIN32
  struct
  {
    void* file        = nullptr;
    void* fileMapping = nullptr;
  } m_win32;
#else
  struct
  {
    int file = -1;
  } m_unix;
#endif

  bool        m_isValid  = false;
  size_t      m_fileSize = 0;
  MappingType m_mappingType;
  void*       m_mappingPtr  = nullptr;
  size_t      m_mappingSize = 0;
};

// convenience types
class FileReadMapping : private FileMapping
{
public:
  bool        open(const char* filename) { return FileMapping::open(filename, MAPPING_READONLY, 0); }
  void        close() { FileMapping::close(); }
  const void* data() const { return m_mappingPtr; }
  size_t      size() const { return m_fileSize; }
  bool        valid() const { return m_isValid; }
};

class FileReadOverWriteMapping : private FileMapping
{
public:
  bool open(const char* filename, size_t fileSize)
  {
    return FileMapping::open(filename, MAPPING_READOVERWRITE, fileSize);
  }
  void   close() { FileMapping::close(); }
  void*  data() { return m_mappingPtr; }
  size_t size() const { return m_fileSize; }
  bool   valid() const { return m_isValid; }
};
}  // namespace nvh
