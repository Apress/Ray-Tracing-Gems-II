/* Copyright (c) 2014-2019, NVIDIA CORPORATION. All rights reserved.
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


//--------------------------------------------------------------------------------------------------
/** 
  # class nvh::InputParser
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
*/

#pragma once
#include <string>
#include <vector>

class InputParser
{
public:
  InputParser(int& argc, char** argv)
  {
    for(int i = 1; i < argc; ++i)
    {
      if(argv[i])
      {
        m_tokens.emplace_back(argv[i]);
      }
    }
  }

  auto findOption(const std::string& option) const { return std::find(m_tokens.begin(), m_tokens.end(), option); }
  const std::string getString(const std::string& option, std::string defaultString = "") const
  {
    if(exist(option))
    {
      auto itr = findOption(option);
      if(itr != m_tokens.end() && ++itr != m_tokens.end())
      {
        return *itr;
      }
    }

    return defaultString;
  }

  std::vector<std::string> getString(const std::string& option, uint32_t nbElem) const
  {
    auto                     itr = findOption(option);
    std::vector<std::string> items;
    while(itr != m_tokens.end() && ++itr != m_tokens.end() && nbElem-- > 0)
    {
      items.push_back((*itr));
    }
    return items;
  }

  int getInt(const std::string& option, int defaultValue = 0) const
  {
    if(exist(option))
      return std::stoi(getString(option));
    return defaultValue;
  }

  auto getInt2(const std::string& option, std::array<int, 2> defaultValues = {0, 0}) const
  {
    if(exist(option))
    {
      auto items = getString(option, 2);
      if(items.size() == 2)
      {
        defaultValues[0] = std::stoi(items[0]);
        defaultValues[1] = std::stoi(items[1]);
      }
    }

    return defaultValues;
  }

  float getFloat(const std::string& option, float defaultValue = 0.0f) const
  {
    if(exist(option))
      return std::stof(getString(option));

    return defaultValue;
  }

  bool exist(const std::string& option) const { return findOption(option) != m_tokens.end(); }

private:
  std::vector<std::string> m_tokens;
};
