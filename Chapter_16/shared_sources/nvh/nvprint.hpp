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

#ifndef __NVPRINT_H__
#define __NVPRINT_H__


#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>


/**
  # global nvprintf functions

  Multiple functions and macros that should be used for logging purposes,
  rather than printf

  - nvprintf : print at default loglevel
  - nvprintfLevel : print at a certain loglevel
  - nvprintSetLevel : sets default loglevel
  - nvprintGetLevel : gets default loglevel
  - nvprintSetLogFileName : sets log filename
  - nvprintSetLogging : sets file logging state
  - nvprintSetCallback : sets custom callback
  - LOGI : macro that does nvprintfLevel(LOGLEVEL_INFO)
  - LOGW : macro that does nvprintfLevel(LOGLEVEL_WARNING)
  - LOGE : macro that does nvprintfLevel(LOGLEVEL_ERROR)
  - LOGE_FILELINE : macro that does nvprintfLevel(LOGLEVEL_ERROR) combined with filename/line
  - LOGD : macro that does nvprintfLevel(LOGLEVEL_DEBUG) (only in debug builds)
  - LOGOK : macro that does nvprintfLevel(LOGLEVEL_OK)
  - LOGSTATS : macro that does nvprintfLevel(LOGLEVEL_STATS)
*/


// trick for pragma message so we can write:
// #pragma message(__FILE__"("S__LINE__"): blah")
#define S__(x) #x
#define S_(x) S__(x)
#define S__LINE__ S_(__LINE__)

#   ifndef  LOGLEVEL_INFO
#   define  LOGLEVEL_INFO     0
#   define  LOGLEVEL_WARNING  1
#   define  LOGLEVEL_ERROR    2
#   define  LOGLEVEL_DEBUG    3
#   define  LOGLEVEL_STATS    4
#   define  LOGLEVEL_OK       7
#   endif

#   define  LOGI(...)  { nvprintfLevel(LOGLEVEL_INFO, __VA_ARGS__); }
#   define  LOGW(...)  { nvprintfLevel(LOGLEVEL_WARNING, __VA_ARGS__); }
#   define  LOGE(...)  { nvprintfLevel(LOGLEVEL_ERROR, __VA_ARGS__ ); }
#   define  LOGE_FILELINE(...)  { nvprintfLevel(LOGLEVEL_ERROR, __FILE__ "(" S__LINE__ "): **ERROR**:\n" __VA_ARGS__ ); }
#ifdef _DEBUG
#   define  LOGD(...)  { nvprintfLevel(LOGLEVEL_DEBUG, __FILE__ "(" S__LINE__ "): Debug Info:\n" __VA_ARGS__ ); }
#else
#   define  LOGD(...)
#endif
#   define  LOGOK(...) { nvprintfLevel(LOGLEVEL_OK, __VA_ARGS__); }
#   define  LOGSTATS(...)  { nvprintfLevel(LOGLEVEL_STATS, __VA_ARGS__); }

#if _MSC_VER
  #define snprintf _snprintf
#endif

typedef void (*PFN_NVPRINTCALLBACK)(int level, const char * fmt);

void nvprintf(const char * fmt, ...);
void nvprintfLevel(int level, const char * fmt, ...);
void nvprintSetLevel(int l);
int  nvprintGetLevel();
void nvprintSetLogFileName(const char* name);
void nvprintSetLogging(bool b);
void nvprintSetFileLogging(bool state, uint32_t mask = ~0);
void nvprintSetCallback(PFN_NVPRINTCALLBACK callback);


#endif

