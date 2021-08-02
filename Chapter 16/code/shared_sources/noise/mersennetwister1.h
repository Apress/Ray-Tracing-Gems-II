// MersenneTwister.h
// Mersenne Twister random number generator -- a C++ class MTRand
// Based on code by Makoto Matsumoto, Takuji Nishimura, and Shawn Cokus
// Richard J. Wagner v1.0 15 May 2003 rjwagner@writeme.com

// The Mersenne Twister is an algorithm for generating random numbers. It
// was designed with consideration of the flaws in various other generators.
// The period, 2^19937-1, and the order of equidistribution, 623 dimensions,
// are far greater. The generator is also fast; it avoids multiplication and
// division, and it benefits from caches and pipelines. For more information
// see the inventors' web page at http://www.math.keio.ac.jp/~matumoto/emt.html

// Reference
// M. Matsumoto and T. Nishimura, "Mersenne Twister: A 623-Dimensionally
// Equidistributed Uniform Pseudo-Random Number Generator", ACM Transactions on
// Modeling and Computer Simulation, Vol. 8, No. 1, January 1998, pp 3-30.

// Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
// Copyright (C) 2000 - 2003, Richard J. Wagner
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. The names of its contributors may not be used to endorse or promote
// products derived from this software without specific prior written
// permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// The original code included the following notice:
//
// When you use this, send an email to: matumoto@math.keio.ac.jp
// with an appropriate reference to your work.
//
// It would be nice to CC: rjwagner@writeme.com and Cokus@math.washington.edu
// when you write.

#ifndef _MTRAND_H_
#define _MTRAND_H_

#include <climits>
#include <ctime>

#pragma warning (disable : 4146 )

// Not thread safe (unless auto-initialization is avoided and each thread has
// its own MTRand object)

class MTRand {

  public:
    // constructors
    inline MTRand();
    inline MTRand(unsigned int oneSeed);
    inline MTRand(unsigned int bigSeed[], unsigned int seedLength = N);

    // Access to random integers
    inline unsigned int randInt(); // integer in [0, 2^32 - 1]
    inline unsigned int randInt(unsigned int n); // integer in [0, n)
    inline unsigned int operator()(unsigned int n); // integer in [0, n) (for use with STL)

    // Access to 32-bit random numbers
    inline float rand(); // real number in [0,1]
    inline float randExc(); // real number in [0,1)
    inline float randDblExc(); // real number in (0,1)

    // Access to 53-bit random numbers (capacity of IEEE double precision)
    inline double rand53(); // real number in [0,1)

    // Re-seeding functions with same behavior as initializers
    inline void seed();
    inline void seed(unsigned int oneSeed);
    inline void seed(unsigned int bigSeed[], unsigned int seedLength = N);

    const static int N = 624; // length of state vector

  protected:
    const static int M = 397; // period parameter

    unsigned int state[N]; // internal state
    unsigned int *pNext; // next value to get from state
    int left; // number of values left before reload needed

    inline void initialize(unsigned int oneSeed);
    inline void reload();
    inline static unsigned int hash( time_t t, clock_t c );

    inline unsigned int hiBit(unsigned int u) const {
        return u & 0x80000000U;
    }

    inline unsigned int loBit(unsigned int u) const {
        return u & 0x00000001U;
    }

    inline unsigned int loBits(unsigned int u) const {
        return u & 0x7fffffffU;
    }

    inline unsigned int mixBits(unsigned int u, unsigned int v) const {
        return hiBit(u) | loBits(v);
    }

    inline unsigned int twist(unsigned int m, unsigned int s0, unsigned int s1) const {
        return m ^ (mixBits(s0, s1) >> 1) ^ (-loBit(s1) & 0x9908b0dfU);
    }

};


/**
 * Constructors
 **/
inline MTRand::MTRand() {
    seed();
}

inline MTRand::MTRand(unsigned int oneSeed) {
    seed(oneSeed);
}

inline MTRand::MTRand(unsigned int bigSeed[], unsigned int seedLength) {
    seed(bigSeed, seedLength);
}


/**
 * Random integers
 **/

/**
 * Pull a 32-bit integer from the generator state. Every other access function
 * simply transforms the numbers extracted here
 **/
inline unsigned int MTRand::randInt() {
    if (left == 0)
        reload();
    --left;

    unsigned int s1;
    s1 = *pNext++;
    s1 ^= (s1 >> 11);
    s1 ^= (s1 << 7) & 0x9d2c5680U;
    s1 ^= (s1 << 15) & 0xefc60000U;
    return (s1 ^ (s1 >> 18));
}

/**
 * Optimized by Magnus Jonsson (magnus@smartelectronix.com).
 **/
inline unsigned int MTRand::randInt(unsigned int n) {
    // Find which bits are used in n
    unsigned int used = n;
    used |= used >> 1;
    used |= used >> 2;
    used |= used >> 4;
    used |= used >> 8;
    used |= used >> 16;

    // Draw numbers until one is found in [0,n)
    unsigned int i;
    do {
        i = randInt() & used; // toss unused bits to shorten search
    } while (i >= n);
    return i;
}

inline unsigned int MTRand::operator()(unsigned int n) {
    return randInt(n);
}


/**
 * Random floating-point numbers
 **/

inline float MTRand::rand() {
    return static_cast<float>(randInt()) / 4294967295.0f;
}

inline float MTRand::randExc() {
    return static_cast<float>(randInt()) / 4294967296.0f;
}

inline float MTRand::randDblExc() {
    return (static_cast<float>(randInt()) + 0.5f) / 4294967296.0f;
}

/**
 * By Isaku Wada.
 **/
inline double MTRand::rand53() {
    unsigned int a(randInt() >> 5), b(randInt() >> 6);
    return (a * 67108864.0 + b) / 9007199254740992.0;
}


/**
 * Seeding functions
 **/

inline void MTRand::seed() {
    seed(hash(time(NULL), clock()));
}

inline void MTRand::seed(unsigned int oneSeed) {
    initialize(oneSeed);
    reload();
}

/**
 * Seed the generator with an array of unsigned int's. There are 2^19937-1
 * possible initial states. This function allows all of those to be accessed by
 * providing at least 19937 bits (with a default seed length of N = 624 unsigned
 * int's). Any bits above the lower 32 in each element are discarded.
 **/
inline void MTRand::seed(unsigned int bigSeed[], unsigned int seedLength) {
    initialize(19650218UL);
    int i(1);
    unsigned int j(0);
    int k((N > static_cast<int>(seedLength)) ? N : static_cast<int>(seedLength));

    for ( ; k; --k) {
        state[i] = state[i] ^ ((state[i - 1] ^ (state[i - 1] >> 30)) * 1664525U);
        state[i] += (bigSeed[j] & 0xffffffffU) + j;
        state[i] &= 0xffffffffU;
        ++i;
        ++j;
        if (i >= N) {
            state[0] = state[N - 1];
            i = 1;
        }
        if (j >= seedLength)
            j = 0;
    }
    for (k = (N - 1); k; --k) {
        state[i] = state[i] ^ ((state[i - 1] ^ (state[i - 1] >> 30)) * 1566083941U);
        state[i] -= i;
        state[i] &= 0xffffffffU;
        ++i;
        if (i >= N) {
            state[0] = state[N - 1];
            i = 1;
        }
    }
    state[0] = 0x80000000UL; // MSB is 1, assuring non-zero initial array
    reload();
}


/**
 * Bookkeeping functions
 **/

/**
 * Initialize generator state with seed. See Knuth TAOCP Vol 2, 3rd Ed, p.106
 * for multiplier. In previous versions, most significant bits (MSBs) of the
 * seed affect only MSBs of the state array. Modified 9 Jan 2002 by Makoto
 * Matsumoto.
 **/
inline void MTRand::initialize(unsigned int seed) {
    unsigned int *s(state);
    unsigned int *r(state);
    int i(1);
    *s++ = seed & 0xffffffffU;
    for ( ; i < N; ++i) {
        *s++ = (1812433253U * (*r ^ (*r >> 30)) + i) & 0xffffffffU;
        r++;
    }
}

/**
 * Generate N new values in state. Made clearer and faster by Matthew Bellew
 * (matthew.bellew@home.com).
 **/
inline void MTRand::reload() {
    unsigned int *p(state);
    int i;
    for (i = N - M; i--; ++p)
        *p = twist(p[M], p[0], p[1]);
    for (i = M; --i; ++p)
        *p = twist(p[M - N], p[0], p[1]);
    *p = twist(p[M - N], p[0], state[0] );

    left = N;
    pNext = state;
}


inline unsigned int MTRand::hash(time_t t, clock_t c) {
    // used to guarantee time-based seeds will change
    static unsigned int differ(0);

    unsigned int h1(0);
    unsigned char *p(reinterpret_cast<unsigned char *>(&t));
    for (size_t i = 0; i < sizeof(t); ++i) {
        h1 *= UCHAR_MAX + 2U;
        h1 += p[i];
    }

    unsigned int h2(0);
    p = reinterpret_cast<unsigned char *>(&c);
    for (size_t j = 0; j < sizeof(c); ++j) {
        h2 *= UCHAR_MAX + 2U;
        h2 += p[j];
    }

    return (h1 + differ++) ^ h2;
}

#endif
