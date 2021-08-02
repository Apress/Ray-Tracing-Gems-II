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

#include <stdio.h>

inline void saveBMP(const char* bmpfilename, int width, int height, const unsigned char* bgra)
{
#pragma pack(push, 1)
  struct
  {
    unsigned short bfType;
    unsigned int   bfSize;
    unsigned int   bfReserved;
    unsigned int   bfOffBits;

    unsigned int   biSize;
    signed int     biWidth;
    signed int     biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned int   biCompression;
    unsigned int   biSizeImage;
    signed int     biXPelsPerMeter;
    signed int     biYPelsPerMeter;
    unsigned int   biClrUsed;
    unsigned int   biClrImportant;
  } bmpinfo;
#pragma pack(pop)

  bmpinfo.bfType     = 19778;
  bmpinfo.bfSize     = sizeof(bmpinfo) + width * height * 4 * sizeof(unsigned char);
  bmpinfo.bfReserved = 0;
  bmpinfo.bfOffBits  = 54;

  bmpinfo.biSize          = 40;
  bmpinfo.biWidth         = width;
  bmpinfo.biHeight        = height;
  bmpinfo.biPlanes        = 1;
  bmpinfo.biBitCount      = 32;
  bmpinfo.biCompression   = 0;
  bmpinfo.biSizeImage     = 0;
  bmpinfo.biXPelsPerMeter = 0;
  bmpinfo.biYPelsPerMeter = 0;
  bmpinfo.biClrUsed       = 0;
  bmpinfo.biClrImportant  = 0;

  FILE* bmpfile = fopen(bmpfilename, "wb");
  fwrite(&bmpinfo, sizeof(bmpinfo), 1, bmpfile);
  fwrite(bgra, sizeof(char), width * height * 4 * sizeof(unsigned char), bmpfile);
  fclose(bmpfile);
}