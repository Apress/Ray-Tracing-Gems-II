/*-----------------------------------------------------------------------
    Copyright (c) 2013, NVIDIA. All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:
     * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
     * Neither the name of its contributors may be used to endorse 
       or promote products derived from this software without specific
       prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
    OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/ //--------------------------------------------------------------------
#pragma once
#include <chrono>
//-----------------------------------------------------------------------------
// TimeSampler work
//-----------------------------------------------------------------------------
struct TimeSampler
{
  bool   bNonStopRendering;
  int    renderCnt;
  double start_time, end_time;
  int    timing_counter;
  int    maxTimeSamples;
  int    frameFPS;
  double frameDT;
  TimeSampler()
  {
    bNonStopRendering = true;
    renderCnt         = 1;
    timing_counter    = 0;
    maxTimeSamples    = 60;
    frameDT           = 1.0 / 60.0;
    frameFPS          = 0;
    start_time = end_time = getTime();
  }
  inline double getTime()
  {
    auto now(std::chrono::system_clock::now());
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000.0;
  }
  inline double getFrameDT() { return frameDT; }
  inline int    getFPS() { return frameFPS; }
  void          resetSampling(int i = 10) { maxTimeSamples = i; }
  bool          update(bool bContinueToRender, bool* glitch = nullptr)
  {
    if(glitch)
      *glitch = false;
    bool updated = false;


    if((timing_counter >= maxTimeSamples) && (maxTimeSamples > 0))
    {
      timing_counter = 0;
      end_time       = getTime();
      frameDT        = (end_time - start_time) / 1000.0;
      // Linux/OSX etc. TODO
      frameDT /= maxTimeSamples;
#define MAXDT (1.0 / 40.0)
#define MINDT (1.0 / 3000.0)
      if(frameDT < MINDT)
      {
        frameDT = MINDT;
      }
      else if(frameDT > MAXDT)
      {
        frameDT = MAXDT;
        if(glitch)
          *glitch = true;
      }
      frameFPS = (int)(1.0 / frameDT);
      // update the amount of samples to average, depending on the speed of the scene
      maxTimeSamples = (int)(0.15 / (frameDT));
      if(maxTimeSamples > 50)
        maxTimeSamples = 50;
      updated = true;
    }
    if(bContinueToRender || bNonStopRendering)
    {
      if(timing_counter == 0)
        start_time = getTime();
      timing_counter++;
    }
    return updated;
    return true;
  }
};
