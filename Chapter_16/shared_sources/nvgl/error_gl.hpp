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
 
#ifndef NVGL_ERROR_INCLUDED
#define NVGL_ERROR_INCLUDED

#include <include_gl.h>
#include <string>
#include <vector>

#include <nvh/nvprint.hpp>

/**
  # functions in nvgl

  Several utility functions that aid debugging. Check if all bindings
  are cleared, framebuffer complete etc.
*/

namespace nvgl
{
  // tests if version is available
  bool checkGLVersion(GLint MajorVersionRequire, GLint MinorVersionRequire);

  // tests if extension string is available
  bool checkExtension(char const * String);

  // tests against any gl error
  bool checkError(const char* Title);

  // tests for framebuffer incompleteness, also prints errors
  bool checkNamedFramebuffer(GLuint fbo);

  enum CheckBindingBits {
    CHECKBINDING_VAO_BIT = 1<<0,
    CHECKBINDING_FBO_BIT = 1<<1,
    CHECKBINDING_PROGRAM_BIT = 1<<2,
    CHECKBINDING_TEXTURES_BIT = 1<<3,
    CHECKBINDING_SAMPLERS_BIT = 1<<4,
    CHECKBINDING_BUFFERS_BIT = 1<<5,
    CHECKBINDING_IMAGES_BIT = 1<<6,
    CHECKBINDING_XFB_BIT = 1<<7,
    CHECKBINDING_VATTRIBS_BIT = 1<<8,
    CHECKBINDING_ALL_BITS = ~0,
  };

#ifndef NDEBUG
  #define DBG_CHECKBINDINGS(bindingBits)   nvgl::checkBindings(bindingBits, __FILE__, __LINE__);
#else
  #define DBG_CHECKBINDINGS(bindingBits)
#endif

  // tests if the bindings specified by the bits are set to 0
  bool checkBindings(int bindingBits, const char* marker = NULL, int num=0);

  // tests if the texture has the expected target
  bool checkTextureTarget(GLuint texture, GLenum target);
  bool checkTextureTarget(GLuint texture, GLenum target, const char* name, const char* marker = NULL, int num = 0);


  /**
    # template class nvgl::CheckBufferContent
    Utility wrapper to downlad buffer data into a temp vector for debugging
  */
  template<class T>
  class CheckBufferContent {
  public:
    std::vector<T>  content;

    CheckBufferContent(GLuint buffer, size_t offset, size_t size)
    {
      size_t elements = size / sizeof(T);
      content.resize(elements);
      glGetNamedBufferSubData(buffer, offset, elements * sizeof(T), content.data());
      elements = elements;
    }
  };

  /**
    # class nvgl::CheckBufferResidency
    Utility class to test if a certain gpu address is coming from
    a resident buffer. Register the address of buffers in advance.
  */
  class CheckBufferResidency {
  public:
    void addAddress(uint64_t address, uint64_t size);
    void removeAddress(uint64_t address);
    bool checkAddress(uint64_t address);
    size_t getEntryNum() const {
      return m_entries.size();
    }

  private:
    struct Entry {
      uint64_t address;
      uint64_t size;

      Entry() {}
      Entry(uint64_t address, uint64_t size=0) : address(address) , size(size) {}
    };

    size_t find(uint64_t address);

    static bool Entry_compare(const Entry& a, const Entry& b)
    {
      return a.address < b.address;
    }

    void sort();

    bool                m_dirty;
    std::vector<Entry>  m_entries;
  };
}


#endif//NV_ERROR_INCLUDED
