// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include <limits>
#include <limits.h>
#ifdef __CUDACC__
#include <math_constants.h>
#endif

#ifndef M_PI
#define M_PI 3.141593f
#endif

namespace owl {
  namespace common {

    static struct ZeroTy
    {
      __both__ operator          double   ( ) const { return 0; }
      __both__ operator          float    ( ) const { return 0; }
      __both__ operator          long long( ) const { return 0; }
      __both__ operator unsigned long long( ) const { return 0; }
      __both__ operator          long     ( ) const { return 0; }
      __both__ operator unsigned long     ( ) const { return 0; }
      __both__ operator          int      ( ) const { return 0; }
      __both__ operator unsigned int      ( ) const { return 0; }
      __both__ operator          short    ( ) const { return 0; }
      __both__ operator unsigned short    ( ) const { return 0; }
      __both__ operator          char     ( ) const { return 0; }
      __both__ operator unsigned char     ( ) const { return 0; }
    } zero MAYBE_UNUSED;

    static struct OneTy
    {
      __both__ operator          double   ( ) const { return 1; }
      __both__ operator          float    ( ) const { return 1; }
      __both__ operator          long long( ) const { return 1; }
      __both__ operator unsigned long long( ) const { return 1; }
      __both__ operator          long     ( ) const { return 1; }
      __both__ operator unsigned long     ( ) const { return 1; }
      __both__ operator          int      ( ) const { return 1; }
      __both__ operator unsigned int      ( ) const { return 1; }
      __both__ operator          short    ( ) const { return 1; }
      __both__ operator unsigned short    ( ) const { return 1; }
      __both__ operator          char     ( ) const { return 1; }
      __both__ operator unsigned char     ( ) const { return 1; }
    } one MAYBE_UNUSED;

    static struct NegInfTy
    {
#ifdef __CUDA_ARCH__
      __device__ operator          double   ( ) const { return -CUDART_INF; }
      __device__ operator          float    ( ) const { return -CUDART_INF_F; }
#else
      __both__ operator          double   ( ) const { return -std::numeric_limits<double>::infinity(); }
      __both__ operator          float    ( ) const { return -std::numeric_limits<float>::infinity(); }
      __both__ operator          long long( ) const { return std::numeric_limits<long long>::min(); }
      __both__ operator unsigned long long( ) const { return std::numeric_limits<unsigned long long>::min(); }
      __both__ operator          long     ( ) const { return std::numeric_limits<long>::min(); }
      __both__ operator unsigned long     ( ) const { return std::numeric_limits<unsigned long>::min(); }
      __both__ operator          int      ( ) const { return std::numeric_limits<int>::min(); }
      __both__ operator unsigned int      ( ) const { return std::numeric_limits<unsigned int>::min(); }
      __both__ operator          short    ( ) const { return std::numeric_limits<short>::min(); }
      __both__ operator unsigned short    ( ) const { return std::numeric_limits<unsigned short>::min(); }
      __both__ operator          char     ( ) const { return std::numeric_limits<char>::min(); }
      __both__ operator unsigned char     ( ) const { return std::numeric_limits<unsigned char>::min(); }
#endif
    } neg_inf MAYBE_UNUSED;

    inline __both__ float infty() {
#if defined(__CUDA_ARCH__)
      return CUDART_INF_F; 
#else
      return std::numeric_limits<float>::infinity(); 
#endif
    }
  
    static struct PosInfTy
    {
#ifdef __CUDA_ARCH__
      __device__ operator          double   ( ) const { return CUDART_INF; }
      __device__ operator          float    ( ) const { return CUDART_INF_F; }
#else
      __both__ operator          double   ( ) const { return std::numeric_limits<double>::infinity(); }
      __both__ operator          float    ( ) const { return std::numeric_limits<float>::infinity(); }
      __both__ operator          long long( ) const { return std::numeric_limits<long long>::max(); }
      __both__ operator unsigned long long( ) const { return std::numeric_limits<unsigned long long>::max(); }
      __both__ operator          long     ( ) const { return std::numeric_limits<long>::max(); }
      __both__ operator unsigned long     ( ) const { return std::numeric_limits<unsigned long>::max(); }
      __both__ operator          int      ( ) const { return std::numeric_limits<int>::max(); }
      __both__ operator unsigned int      ( ) const { return std::numeric_limits<unsigned int>::max(); }
      __both__ operator          short    ( ) const { return std::numeric_limits<short>::max(); }
      __both__ operator unsigned short    ( ) const { return std::numeric_limits<unsigned short>::max(); }
      __both__ operator          char     ( ) const { return std::numeric_limits<char>::max(); }
      __both__ operator unsigned char     ( ) const { return std::numeric_limits<unsigned char>::max(); }
#endif
    } inf MAYBE_UNUSED, pos_inf MAYBE_UNUSED;

    static struct NaNTy
    {
#ifdef __CUDA_ARCH__
      __device__ operator double( ) const { return CUDART_NAN; }
      __device__ operator float ( ) const { return CUDART_NAN_F; }
#else
      __both__ operator double( ) const { return std::numeric_limits<double>::quiet_NaN(); }
      __both__ operator float ( ) const { return std::numeric_limits<float>::quiet_NaN(); }
#endif
    } nan MAYBE_UNUSED;

    static struct UlpTy
    {
#ifdef __CUDA_ARCH__
      // todo
#else
      __both__ operator double( ) const { return std::numeric_limits<double>::epsilon(); }
      __both__ operator float ( ) const { return std::numeric_limits<float>::epsilon(); }
#endif
    } ulp MAYBE_UNUSED;



    template<bool is_integer>
    struct limits_traits;

    template<> struct limits_traits<true> {
      template<typename T> static inline __both__ T value_limits_lower(T) { return std::numeric_limits<T>::min(); }
      template<typename T> static inline __both__ T value_limits_upper(T) { return std::numeric_limits<T>::max(); }
    };
    template<> struct limits_traits<false> {
      template<typename T> static inline __both__ T value_limits_lower(T) { return (T)NegInfTy(); }//{ return -std::numeric_limits<T>::infinity(); }
      template<typename T> static inline __both__ T value_limits_upper(T) { return (T)PosInfTy(); }//{ return +std::numeric_limits<T>::infinity();  }
    };
  
    /*! lower value of a completely *empty* range [+inf..-inf] */
    template<typename T> inline __both__ T empty_bounds_lower()
    {
      return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_upper(T());
    }
  
    /*! upper value of a completely *empty* range [+inf..-inf] */
    template<typename T> inline __both__ T empty_bounds_upper()
    {
      return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_lower(T());
    }

    /*! lower value of a completely *empty* range [+inf..-inf] */
    template<typename T> inline __both__ T empty_range_lower()
    {
      return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_upper(T());
    }
  
    /*! upper value of a completely *empty* range [+inf..-inf] */
    template<typename T> inline __both__ T empty_range_upper()
    {
      return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_lower(T());
    }

    /*! lower value of a completely open range [-inf..+inf] */
    template<typename T> inline __both__ T open_range_lower()
    {
      return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_lower(T());
    }

    /*! upper value of a completely open range [-inf..+inf] */
    template<typename T> inline __both__ T open_range_upper()
    {
      return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_upper(T());
    }

    template<> inline __both__ uint32_t empty_bounds_lower<uint32_t>() 
    { return uint32_t(UINT_MAX); }
    template<> inline __both__ uint32_t empty_bounds_upper<uint32_t>() 
    { return uint32_t(0); }
    template<> inline __both__ uint32_t open_range_lower<uint32_t>() 
    { return uint32_t(0); }
    template<> inline __both__ uint32_t open_range_upper<uint32_t>() 
    { return uint32_t(UINT_MAX); }

    template<> inline __both__ int32_t empty_bounds_lower<int32_t>() 
    { return int32_t(INT_MAX); }
    template<> inline __both__ int32_t empty_bounds_upper<int32_t>() 
    { return int32_t(INT_MIN); }
    template<> inline __both__ int32_t open_range_lower<int32_t>() 
    { return int32_t(INT_MIN); }
    template<> inline __both__ int32_t open_range_upper<int32_t>() 
    { return int32_t(INT_MAX); }

    template<> inline __both__ uint64_t empty_bounds_lower<uint64_t>() 
    { return uint64_t(ULONG_MAX); }
    template<> inline __both__ uint64_t empty_bounds_upper<uint64_t>() 
    { return uint64_t(0); }
    template<> inline __both__ uint64_t open_range_lower<uint64_t>() 
    { return uint64_t(0); }
    template<> inline __both__ uint64_t open_range_upper<uint64_t>() 
    { return uint64_t(ULONG_MAX); }

    template<> inline __both__ int64_t empty_bounds_lower<int64_t>() 
    { return int64_t(LONG_MAX); }
    template<> inline __both__ int64_t empty_bounds_upper<int64_t>() 
    { return int64_t(LONG_MIN); }
    template<> inline __both__ int64_t open_range_lower<int64_t>() 
    { return int64_t(LONG_MIN); }
    template<> inline __both__ int64_t open_range_upper<int64_t>() 
    { return int64_t(LONG_MAX); }


    template<> inline __both__ uint16_t empty_bounds_lower<uint16_t>() 
    { return uint16_t(USHRT_MAX); }
    template<> inline __both__ uint16_t empty_bounds_upper<uint16_t>() 
    { return uint16_t(0); }
    template<> inline __both__ uint16_t open_range_lower<uint16_t>() 
    { return uint16_t(0); }
    template<> inline __both__ uint16_t open_range_upper<uint16_t>() 
    { return uint16_t(USHRT_MAX); }

    template<> inline __both__ int16_t empty_bounds_lower<int16_t>() 
    { return int16_t(SHRT_MAX); }
    template<> inline __both__ int16_t empty_bounds_upper<int16_t>() 
    { return int16_t(SHRT_MIN); }
    template<> inline __both__ int16_t open_range_lower<int16_t>() 
    { return int16_t(SHRT_MIN); }
    template<> inline __both__ int16_t open_range_upper<int16_t>() 
    { return int16_t(SHRT_MAX); }

    template<> inline __both__ uint8_t empty_bounds_lower<uint8_t>() 
    { return uint8_t(CHAR_MAX); }
    template<> inline __both__ uint8_t empty_bounds_upper<uint8_t>() 
    { return uint8_t(CHAR_MIN); }
    template<> inline __both__ uint8_t open_range_lower<uint8_t>() 
    { return uint8_t(CHAR_MIN); }
    template<> inline __both__ uint8_t open_range_upper<uint8_t>() 
    { return uint8_t(CHAR_MAX); }

    template<> inline __both__ int8_t empty_bounds_lower<int8_t>() 
    { return int8_t(SCHAR_MIN); }
    template<> inline __both__ int8_t empty_bounds_upper<int8_t>() 
    { return int8_t(SCHAR_MAX); }
    template<> inline __both__ int8_t open_range_lower<int8_t>() 
    { return int8_t(SCHAR_MAX); }
    template<> inline __both__ int8_t open_range_upper<int8_t>() 
    { return int8_t(SCHAR_MIN); }

  } // ::owl::common
} // ::owl
