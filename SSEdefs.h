#ifndef __SSEDEFS_H__
#define __SSEDEFS_H__

#include "cosmoType.h"

// Hardware-intrinsic SIMD only. Every vector op is an SSE/AVX intrinsic
// through Charm's SSE-Float.h / SSE-Double.h wrappers, so no scalar or
// std::experimental fallback runs inside a SIMD kernel.
// configure.ac sets no CMK_USE_* flag, so the arch comes from the target
// flags it already sets: -march=native gives __AVX__, and sse2 adds -mno-avx.
// CMK_USE_AVX also gates the AVX branch in Charm's SSE-Double.h.
#if !defined(CMK_USE_AVX) && !defined(CMK_USE_SSE2)
	#if defined(__AVX__)
		#define CMK_USE_AVX 1
	#else
		#define CMK_USE_SSE2 1
	#endif
#endif
#if CMK_USE_AVX && !defined(__AVX__)
	#undef CMK_USE_AVX
	#define CMK_USE_AVX 0
#endif
#if CMK_USE_SSE2 && !defined(__SSE2__)
	#undef CMK_USE_SSE2
	#define CMK_USE_SSE2 0
#endif

#if CMK_USE_AVX
	#ifdef COSMO_FLOAT
		#error "single-precision AVX is not supported"
	#endif
	#include "SSE-Double.h"
	#define SSE_VECTOR_WIDTH 4
	typedef SSEDouble SSEcosmoType;
	inline SSEcosmoType sseLoad(const double *p) {
	  SSEcosmoType v; v.val = _mm256_loadu_pd(p); return v;
	}
	inline void sseStore(double *p, SSEcosmoType v) {
	  _mm256_storeu_pd(p, v.val);
	}
#elif CMK_USE_SSE2 && defined(COSMO_FLOAT)
	#define SSE_COSMO_FLOAT
	#include "SSE-Float.h"
	#define SSE_VECTOR_WIDTH 4
	typedef SSEFloat SSEcosmoType;
	inline SSEcosmoType sseLoad(const float *p) {
	  SSEcosmoType v; v.val = _mm_loadu_ps(p); return v;
	}
	inline void sseStore(float *p, SSEcosmoType v) {
	  _mm_storeu_ps(p, v.val);
	}
#elif CMK_USE_SSE2
	#include "SSE-Double.h"
	#define SSE_VECTOR_WIDTH 2
	typedef SSEDouble SSEcosmoType;
	inline SSEcosmoType sseLoad(const double *p) {
	  SSEcosmoType v; v.val = _mm_loadu_pd(p); return v;
	}
	inline void sseStore(double *p, SSEcosmoType v) {
	  _mm_storeu_pd(p, v.val);
	}
#else
	#error "SSE not available"
#endif


// A lane mask is a full-width vector whose lanes are all ones or all zeros,
// as produced by the wrapper comparisons.
typedef SSEcosmoType SSEMask;

// Lane-wise m ? t : f using only bitwise intrinsics, so selected values are
// copied bit-exactly.
inline SSEcosmoType sseSelect(SSEMask m, SSEcosmoType t, SSEcosmoType f) {
  return (m & t) | andnot(m, f);
}
// m ? x : +0.0 per lane (zeroes masked-off lanes, which may hold inf/NaN).
inline SSEcosmoType sseAnd(SSEMask m, SSEcosmoType x) {
  return m & x;
}
// True when no lane of the mask is set.
inline bool sseNone(SSEMask m) {
  return movemask(m) == 0;
}
// Lane-wise a > b ? a : b, the same lane semantics as _mm_max_pd(a, b).
inline SSEcosmoType sseMax(SSEcosmoType a, SSEcosmoType b) {
  return sseSelect(a > b, a, b);
}

#endif
