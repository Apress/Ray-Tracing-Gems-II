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

/*
 * This file contains code derived from glf by Christophe Riccio, www.g-truc.net
 * Copyright (c) 2005 - 2015 G-Truc Creation (www.g-truc.net)
 * https://github.com/g-truc/ogl-samples/blob/master/framework/compiler.cpp
 */

#include "shaderfilemanager.hpp"
#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdarg.h>
#include <stdio.h>

#include "fileoperations.hpp"


namespace nvh {

std::string ShaderFileManager::format(const char* msg, ...)
{
  char    text[8192];
  va_list list;

  if(msg == 0)
    return std::string();

  va_start(list, msg);
  vsnprintf(text, sizeof(text), msg, list);
  va_end(list);

  return std::string(text);
}

inline std::string ShaderFileManager::markerString(int line, std::string const& filename, int fileid)
{
  if(m_supportsExtendedInclude || m_forceLineFilenames)
  {
#if defined(_WIN32) && 1
    std::string fixedname;
    for(size_t i = 0; i < filename.size(); i++)
    {
      char c = filename[i];
      if(c == '/' || c == '\\')
      {
        fixedname.append("\\\\");
      }
      else
      {
        fixedname.append(1, c);
      }
    }
#else
    std::string fixedname = filename;
#endif
    return ShaderFileManager::format("#line %d \"", line) + fixedname + std::string("\"\n");
  }
  else
  {
    return ShaderFileManager::format("#line %d %d\n", line, fileid);
  }
}

std::string ShaderFileManager::getIncludeContent(IncludeID idx, std::string& filename)
{
  IncludeEntry& entry = m_includes[idx];

  filename = entry.filename;

  if(m_forceIncludeContent)
  {
    return entry.content;
  }

  if(!entry.content.empty() && !findFile(entry.filename, m_directories).empty())
  {
    return entry.content;
  }

  std::string content = loadFile(entry.filename, false, m_directories, filename, true);
  return content.empty() ? entry.content : content;
}

std::string ShaderFileManager::getContent(std::string const& filename, std::string& filenameFound)
{
  if(filename.empty())
  {
    return std::string();
  }

  IncludeID idx = findInclude(filename);

  if(idx.isValid())
  {
    return getIncludeContent(idx, filenameFound);
  }

  // fall back
  filenameFound = filename;
  return loadFile(filename, false, m_directories, filenameFound, true);
}

std::string ShaderFileManager::manualInclude(std::string const& filename, std::string& filenameFound, std::string const& prepend, bool foundVersion)
{
  std::string source = getContent(filename, filenameFound);

  if(source.empty())
  {
    return std::string();
  }

  std::stringstream stream;
  stream << source;
  std::string line, text;

  // Handle command line defines
  text += prepend;
  if(m_lineMarkers)
  {
    text += markerString(1, filenameFound, 0);
  }

  int lineCount = 0;
  while(std::getline(stream, line))
  {
    std::size_t offset = 0;
    lineCount++;

    // Version
    offset = line.find("#version");
    if(offset != std::string::npos)
    {
      std::size_t commentOffset = line.find("//");
      if(commentOffset != std::string::npos && commentOffset < offset)
        continue;

      if(foundVersion)
      {
        // someone else already set the version, so just comment out
        text += std::string("//") + line + std::string("\n");
      }
      else
      {
        // Reorder so that the #version line is always the first of a shader text
        text         = line + std::string("\n") + text + std::string("//") + line + std::string("\n");
        foundVersion = true;
      }
      continue;
    }

    // Include
    offset = line.find("#include");
    if(offset != std::string::npos)
    {
      std::size_t commentOffset = line.find("//");
      if(commentOffset != std::string::npos && commentOffset < offset)
        continue;

      size_t firstQuote  = line.find("\"", offset);
      size_t secondQuote = line.find("\"", firstQuote + 1);

      std::string include = line.substr(firstQuote + 1, secondQuote - firstQuote - 1);

      std::string includeFound;
      std::string includeContent = manualInclude(include, includeFound, std::string(), foundVersion);

      if(!includeContent.empty())
      {
        text += includeContent;
        if(m_lineMarkers)
        {
          text += std::string("\n") + markerString(lineCount + 1, filenameFound, 0);
        }
      }

      continue;
    }

    text += line + "\n";
  }

  return text;
}


ShaderFileManager::IncludeID ShaderFileManager::registerInclude(std::string const& name, std::string const& filename, std::string const& content)
{
  // find if already registered
  for(size_t i = 0; i < m_includes.size(); i++)
  {
    if(m_includes[i].name == name)
    {
      m_includes[i].content = content;
      return i;
    }
  }

  IncludeEntry entry;
  entry.name     = name;
  entry.filename = filename.empty() ? name : filename;
  entry.content  = content;

  m_includes.push_back(entry);

  return m_includes.size() - 1;
}


ShaderFileManager::IncludeID ShaderFileManager::findInclude(std::string const& name) const
{
  // check registered includes first
  for(std::size_t i = 0; i < m_includes.size(); ++i)
  {
    if(m_includes[i].name == name)
    {
      return IncludeID(i);
    }
  }

  return IncludeID();
}

bool ShaderFileManager::loadIncludeContent(IncludeID idx)
{
  std::string filenameFound;
  m_includes[idx].content = getIncludeContent(idx, filenameFound);
  return !m_includes[idx].content.empty();
}

const ShaderFileManager::IncludeEntry& ShaderFileManager::getIncludeEntry(IncludeID idx) const
{
  return m_includes[idx];
}

std::string ShaderFileManager::getProcessedContent(std::string const& filename, std::string& filenameFound)
{
  return manualInclude(filename, filenameFound, "", false);
}

}  // namespace nvh
