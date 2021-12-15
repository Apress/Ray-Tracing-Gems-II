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

/* originally taken (and adapted) from ospray, under following license */

// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
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

#include "../math/vec.h"
#include "../math/Quaternion.h"

namespace owl {
  namespace common {

    ////////////////////////////////////////////////////////////////////////////////
    /// 2D Linear Transform (2x2 Matrix)
    ////////////////////////////////////////////////////////////////////////////////

    template<typename T> struct OWL_INTERFACE LinearSpace2
    {
      using vector_t = T;
      // using Scalar = typename T::scalar_t;
      // using vector_t = T;
      using scalar_t = typename T::scalar_t;
    
      /*! default matrix constructor */
      inline LinearSpace2           ( ) = default;
      inline __both__ LinearSpace2           ( const LinearSpace2& other ) { vx = other.vx; vy = other.vy; }
      inline __both__ LinearSpace2& operator=( const LinearSpace2& other ) { vx = other.vx; vy = other.vy; return *this; }

      template<typename L1> inline __both__ LinearSpace2( const LinearSpace2<L1>& s ) : vx(s.vx), vy(s.vy) {}

      /*! matrix construction from column vectors */
      inline __both__ LinearSpace2(const vector_t& vx, const vector_t& vy)
        : vx(vx), vy(vy) {}

      /*! matrix construction from row mayor data */
      inline __both__ LinearSpace2(const scalar_t& m00, const scalar_t& m01, 
                                   const scalar_t& m10, const scalar_t& m11)
        : vx(m00,m10), vy(m01,m11) {}

      /*! compute the determinant of the matrix */
      inline __both__ const scalar_t det() const { return vx.x*vy.y - vx.y*vy.x; }

      /*! compute adjoint matrix */
      inline __both__ const LinearSpace2 adjoint() const { return LinearSpace2(vy.y,-vy.x,-vx.y,vx.x); }

      /*! compute inverse matrix */
      inline __both__ const LinearSpace2 inverse() const { return adjoint()/det(); }

      /*! compute transposed matrix */
      inline __both__ const LinearSpace2 transposed() const { return LinearSpace2(vx.x,vx.y,vy.x,vy.y); }

      /*! returns first row of matrix */
      inline const vector_t row0() const { return vector_t(vx.x,vy.x); }

      /*! returns second row of matrix */
      inline const vector_t row1() const { return vector_t(vx.y,vy.y); }

      ////////////////////////////////////////////////////////////////////////////////
      /// Constants
      ////////////////////////////////////////////////////////////////////////////////

      inline __both__ LinearSpace2( ZeroTy ) : vx(ZeroTy()), vy(ZeroTy()) {}
      inline __both__ LinearSpace2( OneTy ) : vx(OneTy(), ZeroTy()), vy(ZeroTy(), OneTy()) {}

      /*! return matrix for scaling */
      static inline LinearSpace2 scale(const vector_t& s) {
        return LinearSpace2(s.x,   0,
                            0  , s.y);
      }

      /*! return matrix for rotation */
      static inline LinearSpace2 rotate(const scalar_t& r) {
        scalar_t s = sin(r), c = cos(r);
        return LinearSpace2(c, -s,
                            s,  c);
      }

      /*! return closest orthogonal matrix (i.e. a general rotation including reflection) */
      LinearSpace2 orthogonal() const {
        LinearSpace2 m = *this;

        // mirrored?
        scalar_t mirror{scalar_t(OneTy())};
        if (m.det() < scalar_t(ZeroTy())) {
          m.vx = -m.vx;
          mirror = -mirror;
        }

        // rotation
        for (int i = 0; i < 99; i++) {
          const LinearSpace2 m_next = 0.5 * (m + m.transposed().inverse());
          const LinearSpace2 d = m_next - m;
          m = m_next;
          // norm^2 of difference small enough?
          if (max(dot(d.vx, d.vx), dot(d.vy, d.vy)) < 1e-8)
            break;
        }

        // rotation * mirror_x
        return LinearSpace2(mirror*m.vx, m.vy);
      }

    public:

      /*! the column vectors of the matrix */
      vector_t vx,vy;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Unary Operators
    ////////////////////////////////////////////////////////////////////////////////

    template<typename T> __both__ inline LinearSpace2<T> operator -( const LinearSpace2<T>& a ) { return LinearSpace2<T>(-a.vx,-a.vy); }
    template<typename T> __both__ inline LinearSpace2<T> operator +( const LinearSpace2<T>& a ) { return LinearSpace2<T>(+a.vx,+a.vy); }
    template<typename T> __both__ inline LinearSpace2<T> rcp       ( const LinearSpace2<T>& a ) { return a.inverse(); }

    ////////////////////////////////////////////////////////////////////////////////
    // Binary Operators
    ////////////////////////////////////////////////////////////////////////////////

    template<typename T> inline LinearSpace2<T> operator +( const LinearSpace2<T>& a, const LinearSpace2<T>& b ) { return LinearSpace2<T>(a.vx+b.vx,a.vy+b.vy); }
    template<typename T> inline LinearSpace2<T> operator -( const LinearSpace2<T>& a, const LinearSpace2<T>& b ) { return LinearSpace2<T>(a.vx-b.vx,a.vy-b.vy); }

    template<typename T> inline LinearSpace2<T> operator*(const typename T::scalar_t & a, const LinearSpace2<T>& b) { return LinearSpace2<T>(a*b.vx, a*b.vy); }
    template<typename T> inline T               operator*(const LinearSpace2<T>& a, const T              & b) { return b.x*a.vx + b.y*a.vy; }
    template<typename T> inline LinearSpace2<T> operator*(const LinearSpace2<T>& a, const LinearSpace2<T>& b) { return LinearSpace2<T>(a*b.vx, a*b.vy); }

    template<typename T> inline LinearSpace2<T> operator/(const LinearSpace2<T>& a, const typename T::scalar_t & b) { return LinearSpace2<T>(a.vx/b, a.vy/b); }
    template<typename T> inline LinearSpace2<T> operator/(const LinearSpace2<T>& a, const LinearSpace2<T>& b) { return a * rcp(b); }

    template<typename T> inline LinearSpace2<T>& operator *=( LinearSpace2<T>& a, const LinearSpace2<T>& b ) { return a = a * b; }
    template<typename T> inline LinearSpace2<T>& operator /=( LinearSpace2<T>& a, const LinearSpace2<T>& b ) { return a = a / b; }

    ////////////////////////////////////////////////////////////////////////////////
    /// Comparison Operators
    ////////////////////////////////////////////////////////////////////////////////

    template<typename T> inline bool operator ==( const LinearSpace2<T>& a, const LinearSpace2<T>& b ) { return a.vx == b.vx && a.vy == b.vy; }
    template<typename T> inline bool operator !=( const LinearSpace2<T>& a, const LinearSpace2<T>& b ) { return a.vx != b.vx || a.vy != b.vy; }

    ////////////////////////////////////////////////////////////////////////////////
    /// Output Operators
    ////////////////////////////////////////////////////////////////////////////////

    template<typename T> static std::ostream& operator<<(std::ostream& cout, const LinearSpace2<T>& m) {
      return cout << "{ vx = " << m.vx << ", vy = " << m.vy << "}";
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// 3D Linear Transform (3x3 Matrix)
    ////////////////////////////////////////////////////////////////////////////////

    template<typename T> 
    struct OWL_INTERFACE LinearSpace3
    {
      // using vector_t = T;
      using scalar_t = typename T::scalar_t;
      using vector_t = T;
      // using scalar_t = typename T::scalar_t;

      /*! default matrix constructor */
      // inline LinearSpace3           ( ) = default;
      inline __both__ LinearSpace3()
        : vx(OneTy(),ZeroTy(),ZeroTy()),
        vy(ZeroTy(),OneTy(),ZeroTy()),
        vz(ZeroTy(),ZeroTy(),OneTy())
        {}
        
      inline// __both__
        LinearSpace3           ( const LinearSpace3& other ) = default;
      inline __both__ LinearSpace3& operator=( const LinearSpace3& other ) { vx = other.vx; vy = other.vy; vz = other.vz; return *this; }

      template<typename L1> inline __both__ LinearSpace3( const LinearSpace3<L1>& s ) : vx(s.vx), vy(s.vy), vz(s.vz) {}

      /*! matrix construction from column vectors */
      inline __both__ LinearSpace3(const vector_t& vx, const vector_t& vy, const vector_t& vz)
        : vx(vx), vy(vy), vz(vz) {}

      /*! construction from quaternion */
      inline __both__ LinearSpace3( const QuaternionT<scalar_t>& q )
        : vx((q.r*q.r + q.i*q.i - q.j*q.j - q.k*q.k), 2.0f*(q.i*q.j + q.r*q.k), 2.0f*(q.i*q.k - q.r*q.j))
        , vy(2.0f*(q.i*q.j - q.r*q.k), (q.r*q.r - q.i*q.i + q.j*q.j - q.k*q.k), 2.0f*(q.j*q.k + q.r*q.i))
        , vz(2.0f*(q.i*q.k + q.r*q.j), 2.0f*(q.j*q.k - q.r*q.i), (q.r*q.r - q.i*q.i - q.j*q.j + q.k*q.k)) {}

      /*! matrix construction from row mayor data */
      inline __both__ LinearSpace3(const scalar_t& m00, const scalar_t& m01, const scalar_t& m02,
                                   const scalar_t& m10, const scalar_t& m11, const scalar_t& m12,
                                   const scalar_t& m20, const scalar_t& m21, const scalar_t& m22)
        : vx(m00,m10,m20), vy(m01,m11,m21), vz(m02,m12,m22) {}

      /*! compute the determinant of the matrix */
      inline __both__ const scalar_t det() const { return dot(vx,cross(vy,vz)); }

      /*! compute adjoint matrix */
      inline __both__ const LinearSpace3 adjoint() const { return LinearSpace3(cross(vy,vz),cross(vz,vx),cross(vx,vy)).transposed(); }

      /*! compute inverse matrix */
      inline __both__ const LinearSpace3 inverse() const { return adjoint()/det(); }

      /*! compute transposed matrix */
      inline __both__ const LinearSpace3 transposed() const { return LinearSpace3(vx.x,vx.y,vx.z,vy.x,vy.y,vy.z,vz.x,vz.y,vz.z); }

      /*! returns first row of matrix */
      inline __both__ const vector_t row0() const { return vector_t(vx.x,vy.x,vz.x); }

      /*! returns second row of matrix */
      inline __both__ const vector_t row1() const { return vector_t(vx.y,vy.y,vz.y); }

      /*! returns third row of matrix */
      inline __both__ const vector_t row2() const { return vector_t(vx.z,vy.z,vz.z); }

      ////////////////////////////////////////////////////////////////////////////////
      /// Constants
      ////////////////////////////////////////////////////////////////////////////////

// #ifdef __CUDA_ARCH__
      inline __both__ LinearSpace3( const ZeroTy & )
        : vx(ZeroTy()), vy(ZeroTy()), vz(ZeroTy())
        {}
      inline __both__ LinearSpace3( const OneTy & )
        : vx(OneTy(), ZeroTy(), ZeroTy()),
        vy(ZeroTy(), OneTy(), ZeroTy()),
        vz(ZeroTy(), ZeroTy(), OneTy())
        {}
// #else
//       inline __both__ LinearSpace3( ZeroTy ) : vx(zero), vy(zero), vz(zero) {}
//       inline __both__ LinearSpace3( OneTy ) : vx(one, zero, zero), vy(zero, one, zero), vz(zero, zero, one) {}
// #endif

      /*! return matrix for scaling */
      static inline __both__ LinearSpace3 scale(const vector_t& s) {
        return LinearSpace3(s.x,   0,   0,
                            0  , s.y,   0,
                            0  ,   0, s.z);
      }

      /*! return matrix for rotation around arbitrary axis */
      static inline __both__ LinearSpace3 rotate(const vector_t& _u, const scalar_t& r) {
        vector_t u = normalize(_u);
        scalar_t s = sin(r), c = cos(r);
        return LinearSpace3(u.x*u.x+(1-u.x*u.x)*c,  u.x*u.y*(1-c)-u.z*s,    u.x*u.z*(1-c)+u.y*s,
                            u.x*u.y*(1-c)+u.z*s,    u.y*u.y+(1-u.y*u.y)*c,  u.y*u.z*(1-c)-u.x*s,
                            u.x*u.z*(1-c)-u.y*s,    u.y*u.z*(1-c)+u.x*s,    u.z*u.z+(1-u.z*u.z)*c);
      }

      /*! return quaternion for given rotation matrix */
      static inline __both__ QuaternionT<scalar_t> rotation(const LinearSpace3 &a) {
        scalar_t tr = a.vx.x+a.vy.y+a.vz.z+1;
        vector_t diag(a.vx.x,a.vy.y,a.vz.z);
        if (tr > 1) {
          scalar_t s = owl::common::polymorphic::sqrt(tr) * 2;
          return QuaternionT<scalar_t>(.25f * s,
                                       (a.vz.y-a.vy.z)/s,
                                       (a.vx.z-a.vz.x)/s,
                                       (a.vy.x-a.vx.y)/s);
        } else if (arg_min(diag) == 0) {
          scalar_t s = owl::common::polymorphic::sqrt(1.f+diag.x-diag.y-diag.z)*2.f;
          return QuaternionT<scalar_t>((a.vz.y-a.vy.z)/s,
                                       .25f * s,
                                       (a.vx.y-a.vy.x)/s,
                                       (a.vx.z-a.vz.x)/s);
        } else if (arg_min(diag) == 1) {
          scalar_t s = owl::common::polymorphic::sqrt(1.f+diag.y-diag.x-diag.z)*2.f;
          return QuaternionT<scalar_t>((a.vx.z-a.vz.x)/s,
                                       (a.vx.y-a.vy.x)/s,
                                       .25f * s,
                                       (a.vy.z-a.vz.y)/s);
        } else {
          scalar_t s = owl::common::polymorphic::sqrt(1.f+diag.z-diag.x-diag.y)*2.f;
          return QuaternionT<scalar_t>((a.vy.x-a.vx.y)/s,
                                       (a.vx.z-a.vz.x)/s,
                                       (a.vy.z-a.vz.y)/s,
                                       .25f * s);
        }
      }

    public:

      /*! the column vectors of the matrix */
      T vx,vy,vz;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Unary Operators
    ////////////////////////////////////////////////////////////////////////////////

    template<typename T> inline __both__ LinearSpace3<T> operator -( const LinearSpace3<T>& a ) { return LinearSpace3<T>(-a.vx,-a.vy,-a.vz); }
    template<typename T> inline __both__ LinearSpace3<T> operator +( const LinearSpace3<T>& a ) { return LinearSpace3<T>(+a.vx,+a.vy,+a.vz); }
    template<typename T> inline __both__ LinearSpace3<T> rcp       ( const LinearSpace3<T>& a ) { return a.inverse(); }

    /* constructs a coordinate frame form a normalized normal */
    template<typename T>  
    inline __both__ LinearSpace3<T> frame(const T &N) 
    {
// #ifdef __CUDA_ARCH__
      const T dx0 = cross(T(OneTy(),ZeroTy(),ZeroTy()),N);
      const T dx1 = cross(T(ZeroTy(),OneTy(),ZeroTy()),N);
// #else
//       const T dx0 = cross(T(one,zero,zero),N);
//       const T dx1 = cross(T(zero,one,zero),N);
// #endif
      const T dx = normalize(select(dot(dx0,dx0) > dot(dx1,dx1),dx0,dx1));
      const T dy = normalize(cross(N,dx));
      return LinearSpace3<T>(dx,dy,N);
    }

    /* constructs a coordinate frame from a normal and approximate x-direction */
    template<typename T> inline __both__ LinearSpace3<T> frame(const T& N, const T& dxi)
    {
      if (abs(dot(dxi,N)) > 0.99f) return frame(N); // fallback in case N and dxi are very parallel
      const T dx = normalize(cross(dxi,N));
      const T dy = normalize(cross(N,dx));
      return LinearSpace3<T>(dx,dy,N);
    }
  
    /* clamps linear space to range -1 to +1 */
    template<typename T> inline __both__ LinearSpace3<T> clamp(const LinearSpace3<T>& space) {
      return LinearSpace3<T>(clamp(space.vx,T(-1.0f),T(1.0f)),
                             clamp(space.vy,T(-1.0f),T(1.0f)),
                             clamp(space.vz,T(-1.0f),T(1.0f)));
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Binary Operators
    ////////////////////////////////////////////////////////////////////////////////

    template<typename T> inline __both__ LinearSpace3<T> operator +( const LinearSpace3<T>& a, const LinearSpace3<T>& b ) { return LinearSpace3<T>(a.vx+b.vx,a.vy+b.vy,a.vz+b.vz); }
    template<typename T> inline __both__ LinearSpace3<T> operator -( const LinearSpace3<T>& a, const LinearSpace3<T>& b ) { return LinearSpace3<T>(a.vx-b.vx,a.vy-b.vy,a.vz-b.vz); }

    template<typename T> inline __both__ LinearSpace3<T> operator*(const typename T::scalar_t & a, const LinearSpace3<T>& b) { return LinearSpace3<T>(a*b.vx, a*b.vy, a*b.vz); }
    template<typename T> inline __both__ T               operator*(const LinearSpace3<T>& a, const T              & b) { return b.x*a.vx + b.y*a.vy + b.z*a.vz; }
    template<typename T> inline __both__ LinearSpace3<T> operator*(const LinearSpace3<T>& a, const LinearSpace3<T>& b) { return LinearSpace3<T>(a*b.vx, a*b.vy, a*b.vz); }

    template<typename T> __both__ inline LinearSpace3<T> operator/(const LinearSpace3<T>& a, const typename T::scalar_t & b) { return LinearSpace3<T>(a.vx/b, a.vy/b, a.vz/b); }
  
    template<typename T> __both__ inline LinearSpace3<T> operator/(const LinearSpace3<T>& a, const LinearSpace3<T>& b) { return a * rcp(b); }

    template<typename T> inline LinearSpace3<T>& operator *=( LinearSpace3<T>& a, const LinearSpace3<T>& b ) { return a = a * b; }
    template<typename T> inline LinearSpace3<T>& operator /=( LinearSpace3<T>& a, const LinearSpace3<T>& b ) { return a = a / b; }

    template<typename T> inline __both__ T xfmPoint (const LinearSpace3<T>& s, const T& a) { return madd(T(a.x),s.vx,madd(T(a.y),s.vy,T(a.z*s.vz))); }
    template<typename T> inline __both__ T xfmVector(const LinearSpace3<T>& s, const T& a) { return madd(T(a.x),s.vx,madd(T(a.y),s.vy,T(a.z*s.vz))); }
    template<typename T> inline __both__ T xfmNormal(const LinearSpace3<T>& s, const T& a) { return xfmVector(s.inverse().transposed(),a); }

    ////////////////////////////////////////////////////////////////////////////////
    /// Comparison Operators
    ////////////////////////////////////////////////////////////////////////////////

    template<typename T> inline bool operator ==( const LinearSpace3<T>& a, const LinearSpace3<T>& b ) { return a.vx == b.vx && a.vy == b.vy && a.vz == b.vz; }
    template<typename T> inline bool operator !=( const LinearSpace3<T>& a, const LinearSpace3<T>& b ) { return a.vx != b.vx || a.vy != b.vy || a.vz != b.vz; }

    ////////////////////////////////////////////////////////////////////////////////
    /// Output Operators
    ////////////////////////////////////////////////////////////////////////////////

    template<typename T> inline std::ostream& operator<<(std::ostream& cout, const LinearSpace3<T>& m) {
      return cout << "{ vx = " << m.vx << ", vy = " << m.vy << ", vz = " << m.vz << "}";
    }

    /*! Shortcuts for common linear spaces. */
    using LinearSpace2f  = LinearSpace2<vec2f> ;
    using LinearSpace3f  = LinearSpace3<vec3f> ;
    using LinearSpace3fa = LinearSpace3<vec3fa>;

    using linear2f = LinearSpace2f;
    using linear3f = LinearSpace3f;

  } // ::owl::common
} // ::owl
