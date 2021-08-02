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

#ifndef NV_RADIXSORT_INCLUDED
#define NV_RADIXSORT_INCLUDED

namespace nvh {

    /**
      # function nvh::radixsort

      The radixsort function sorts the provided keys based on
      BYTES many bytes stored inside TKey starting at BYTEOFFSET.
      The sorting result is returned as indices into the keys array.
      
      For example:
      
      ``` c++
      struct MyData {
        uint32_t objectIdentifier;
        uint16_t objectSortKey;
      };
      
      
      // 4-byte offset of objectSortKey within MyData
      // 2-byte size of sorting key
      
      result = radixsort<4,2>(keys, indicesIn, indicesTemp);
      
      // after sorting the following is true
      
      keys[result[i]].objectSortKey < keys[result[i + 1]].objectSortKey

      // result can point either to indicesIn or indicesTemp (we swap the arrays
      // after each byte iteration)
      ```
    */
   
    template<uint32_t BYTEOFFSET, uint32_t BYTES, typename TKey>
    uint32_t* radixsort(uint32_t numIndices, const TKey* keys, uint32_t* indicesIn, uint32_t* indicesTemp) {
      uint32_t histogram[BYTES][256] = { 0 };

      for (uint32_t i = 0; i < numIndices; i++) {
        uint32_t idx = indicesIn[i];
        const uint8_t*  bytes = (const uint8_t*)&keys[idx];
        for (uint32_t p = 0; p < BYTES; p++) {
          uint8_t curbyte = bytes[BYTEOFFSET + p];
          histogram[p][curbyte]++;
        }
      }

      uint32_t* tempIn = indicesIn;
      uint32_t* tempOut = indicesTemp;

      for (uint32_t p = 0; p < BYTES; p++) {
        uint32_t offset = 0;
        for (int32_t i = 0; i < 256; i++) {
          uint32_t numBin = histogram[p][i];
          histogram[p][i] = offset;
          offset += numBin;
        }

        for (uint32_t i = 0; i < numIndices; i++) {
          uint32_t idx = tempIn[i];
          const uint8_t*  bytes = (const uint8_t*)&keys[idx];
          uint8_t curbyte = bytes[BYTEOFFSET + p];
          uint32_t pos = histogram[p][curbyte]++;
          tempOut[pos] = idx;
        }

        assert(histogram[p][255] == offset);

        // swap
        uint32_t *temp = tempIn;
        tempIn = tempOut;
        tempOut = temp;
      }

      // post swap tempIn is last tempOut
      return tempIn;
    }

}

#endif
