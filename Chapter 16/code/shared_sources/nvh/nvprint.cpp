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

#include "nvprint.hpp"

#include <vector>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#endif

static char   s_logFileNameDefault[] = "log_nvprosample.txt";
static char*  s_logFileName = s_logFileNameDefault;
static size_t s_strBuffer_sz = 0;
static char*  s_strBuffer = nullptr;
static FILE*  s_fd = nullptr;
static bool   s_bLogReady = false;
static bool   s_bPrintLogging = true;
static uint32_t s_bPrintFileLogging = ~0;
static int    s_printLevel = -1; // <0 mean no level prefix
static PFN_NVPRINTCALLBACK s_printCallback = nullptr;
static std::mutex s_mutex;

void nvprintSetLogFileName(const char* name)
{
  std::lock_guard<std::mutex> lockGuard(s_mutex);

  if(name == NULL || strcmp(s_logFileName, name) == 0)
    return;
  
  size_t l = strlen(name) + 1;
  s_logFileName = new char[l];
  strncpy(s_logFileName, name, l);

  if (s_fd) {
    fclose(s_fd);
    s_fd = nullptr;
    s_bLogReady = false;
  }
}
void nvprintSetCallback(PFN_NVPRINTCALLBACK callback)
{
  s_printCallback = callback;
}
void nvprintSetLevel(int l)
{
  s_printLevel = l;
}
int nvprintGetLevel()
{
  return s_printLevel;
}
void nvprintSetLogging(bool b)
{
  s_bPrintLogging = b;
}

void nvprintSetFileLogging(bool state, uint32_t mask)
{
  std::lock_guard<std::mutex> lockGuard(s_mutex);

  if (state) {
    s_bPrintFileLogging |= mask;
  }
  else {
    s_bPrintFileLogging &= ~mask;
  }
}

void nvprintf2(va_list &vlist, const char * fmt, int level)
{
  if (s_bPrintLogging == false) {
    return;
  }

  std::lock_guard<std::mutex> lockGuard(s_mutex);
  if (s_strBuffer_sz == 0) {
    s_strBuffer_sz = 1024;
    s_strBuffer = (char*)malloc(s_strBuffer_sz);
  }
  while ((vsnprintf(s_strBuffer, s_strBuffer_sz, fmt, vlist)) < 0) // means there wasn't enough room
  {
    s_strBuffer_sz *= 2;
    s_strBuffer = (char*)realloc(s_strBuffer, s_strBuffer_sz);
  }
#ifdef WIN32
  OutputDebugStringA(s_strBuffer);
#endif
#if 1
  if (s_bPrintFileLogging & (1 << level)) {
    if (s_bLogReady == false)
    {
      s_fd = fopen(s_logFileName, "wt");
      s_bLogReady = true;
    }
    if (s_fd)
    {
      fprintf(s_fd, s_strBuffer);
    }
  }
#endif
  if (s_printCallback) {
    s_printCallback(level, s_strBuffer);
  }
  ::printf(s_strBuffer);
}
void nvprintf(const char * fmt, ...)
{
  //    int r = 0;
  va_list  vlist;
  va_start(vlist, fmt);
  nvprintf2(vlist, fmt, s_printLevel);
}
void nvprintfLevel(int level, const char * fmt, ...)
{
  va_list  vlist;
  va_start(vlist, fmt);
  nvprintf2(vlist, fmt, level);
}


