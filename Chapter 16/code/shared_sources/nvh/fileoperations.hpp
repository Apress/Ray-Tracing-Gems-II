/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <algorithm>  // std::max
#include <fstream>
#include <sstream>
#include <vector>

#include "nvprint.hpp"

/**
  # functions in nvh

  - fileExists : check if file exists
  - findFile : finds filename in provided search directories
  - loadFile : (multiple overloads) loads file as std::string, binary or text, can also search in provided directories
  - getFileName : splits filename from filename with path
  - getFilePath : splits filepath from filename with path
*/

namespace nvh {

inline bool fileExists(const char* filename)
{
  std::ifstream stream;
  stream.open(filename);
  return stream.is_open();
}

// returns first found filename (searches within directories provided)
inline std::string findFile(const std::string& infilename, const std::vector<std::string>& directories, bool warn = false)
{
  std::ifstream stream;

  {
    stream.open(infilename.c_str());
    if(stream.is_open())
      return infilename;
  }

  for(const auto& directory : directories)
  {
    std::string filename = directory + "/" + infilename;
    stream.open(filename.c_str());
    if(stream.is_open())
      return filename;
  }

  if(warn)
  {
    nvprintfLevel(LOGLEVEL_WARNING, "File not found: %s\n", infilename.c_str());
    nvprintfLevel(LOGLEVEL_WARNING, "In directories: \n");
    for(const auto& directory : directories)
    {
      nvprintfLevel(LOGLEVEL_WARNING, " - %s\n", directory.c_str());
    }
    nvprintfLevel(LOGLEVEL_WARNING, "\n");
  }

  return {};
}

inline std::string loadFile(const std::string& filename, bool binary)
{
  std::string   result;
  std::ifstream stream(filename, std::ios::ate | (binary ? std::ios::binary : std::ios_base::openmode(0)));

  if(!stream.is_open())
  {
    return result;
  }

  result.reserve(stream.tellg());
  stream.seekg(0, std::ios::beg);

  result.assign((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  return result;
}

inline std::string loadFile(const char* filename, bool binary)
{
  std::string name(filename);
  return loadFile(name, binary);
}

inline std::string loadFile(const std::string&              filename,
                            bool                            binary,
                            const std::vector<std::string>& directories,
                            std::string&                    filenameFound,
                            bool                            warn = false)
{
  filenameFound = findFile(filename, directories, warn);
  if(filenameFound.empty())
  {
    return {};
  }
  else
  {
    return loadFile(filenameFound, binary);
  }
}

inline std::string loadFile(const std::string filename, bool binary, const std::vector<std::string>& directories, bool warn = false)
{
  std::string filenameFound;
  return loadFile(filename, binary, directories, filenameFound, warn);
}

// splits filename excluding path
inline std::string getFileName(std::string const& fullPath)
{
  size_t istart;
  for(istart = fullPath.size() - 1; istart != -1 && fullPath[istart] != '\\' && fullPath[istart] != '/'; istart--)
    ;
  return std::string(&fullPath[istart + 1]);
}

// splits path from filename
inline std::string getFilePath(const char* filename)
{
  std::string path;
  // find path in filename
  {
    std::string filepath(filename);

    size_t pos0 = filepath.rfind('\\');
    size_t pos1 = filepath.rfind('/');

    pos0 = pos0 == std::string::npos ? 0 : pos0;
    pos1 = pos1 == std::string::npos ? 0 : pos1;

    path = filepath.substr(0, std::max(pos0, pos1));
  }

  if(path.empty())
  {
    path = ".";
  }

  return path;
}

// Return true if the filename ends with ending. i.e. ".png"
inline bool endsWith(std::string const& value, std::string const& ending)
{
  if(ending.size() > value.size())
    return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

}  // namespace nvh
