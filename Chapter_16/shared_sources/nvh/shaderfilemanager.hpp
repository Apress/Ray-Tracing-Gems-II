/* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NV_SHADERFILEMANAGER_INCLUDED
#define NV_SHADERFILEMANAGER_INCLUDED


#include <stdio.h>
#include <string>
#include <vector>

namespace nvh {

class ShaderFileManager
{

  //////////////////////////////////////////////////////////////////////////
  /**
    # class nvh::ShaderFileManager

    The ShaderFileManager class is meant to be derived from to create the actual api-specific 
    shader/program managers.

    The ShaderFileManager provides a system to find/load shader files.
    It also allows resolving #include instructions in HLSL/GLSL source files.
    Such includes can be registered before pointing to strings in memory.

    Furthermore it handles injecting prepended strings (typically used for #defines) 
    after the #version statement of GLSL files.

  */

public:
  enum FileType
  {
    FILETYPE_DEFAULT,
    FILETYPE_GLSL,
    FILETYPE_HLSL,
    FILETYPE_SPIRV,
  };

  struct IncludeEntry
  {
    std::string name;
    std::string filename;
    std::string content;
  };

  typedef std::vector<IncludeEntry> IncludeRegistry;

  static std::string format(const char* msg, ...);

public:
  class IncludeID
  {
  public:
    size_t m_value;

    IncludeID()
        : m_value(size_t(~0))
    {
    }

    IncludeID(size_t b)
        : m_value((uint32_t)b)
    {
    }

    IncludeID& operator=(size_t b)
    {
      m_value = b;
      return *this;
    }

    bool isValid() const { return m_value != size_t(~0); }

    operator bool() const { return isValid(); }
    operator size_t() const { return m_value; }

    friend bool operator==(const IncludeID& lhs, const IncludeID& rhs){ return rhs.m_value == lhs.m_value; }
  };

  struct Definition
  {
    Definition()
    {
    }
    Definition(uint32_t type, std::string const& prepend, std::string const& filename)
        : type(type)
        , prepend(prepend)
        , filename(filename)
    {
    }
    Definition(uint32_t type, std::string const& filename)
        : type(type)
        , filename(filename)
    {
    }

    uint32_t    type = 0;
    std::string filename;
    std::string prepend;
    std::string entry = "main";
    FileType    filetype = FILETYPE_DEFAULT;
    std::string filenameFound;
    std::string content;
  };


  // optionally register files to be included, optionally provide content directly rather than from disk
  //
  // name: name used within shader files
  // diskname = filename on disk (defaults to name if not set)
  // content = provide content as string rather than loading from disk

  IncludeID registerInclude(std::string const& name, std::string const& diskname = std::string(), std::string const& content = std::string());

  // Use m_prepend to pass global #defines
  // Derived api classes will use this as global prepend to the per-definition prepends in combination
  // with the source files
  // actualSoure = m_prepend + definition.prepend + definition.content
  std::string m_prepend;

  // per file state, used when FILETYPE_DEFAULT is provided in the Definition
  FileType    m_filetype;

  // add search directories
  void addDirectory(const std::string& dir)
  {
    m_directories.push_back(dir);
  }

  ShaderFileManager()
      : m_forceLineFilenames(false)
      , m_lineMarkers(true)
      , m_forceIncludeContent(false)
      , m_supportsExtendedInclude(false)
      , m_filetype(FILETYPE_GLSL)
  {
    m_directories.push_back(".");
  }

  //////////////////////////////////////////////////////////////////////////

  // in rare cases you may want to access the included content in detail yourself

  IncludeID           findInclude(std::string const& name) const;
  bool                loadIncludeContent(IncludeID);
  const IncludeEntry& getIncludeEntry(IncludeID idx) const;
  
  std::string         getProcessedContent(std::string const& filename, std::string& filenameFound);

protected:
  std::string markerString(int line, std::string const& filename, int fileid);
  std::string getIncludeContent(IncludeID idx, std::string& filenameFound);
  std::string getContent(std::string const& filename, std::string& filenameFound);
  std::string manualInclude(std::string const& filename, std::string& filenameFound, std::string const& prepend, bool foundVersion);

  
  bool m_lineMarkers;
  bool m_forceLineFilenames;
  bool m_forceIncludeContent;
  bool m_supportsExtendedInclude;

  std::vector<std::string>  m_directories;
  IncludeRegistry           m_includes;
};

}  // namespace nvh


#endif  //NV_PROGRAM_INCLUDED
