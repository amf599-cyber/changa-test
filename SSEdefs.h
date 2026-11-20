#ifndef __SSEDEFS_H__
#define __SSEDEFS_H__

#include "cosmoType.h"

// Selection flow (all branches are decided at preprocessing time):
//   1. CMK_USE_AVX / CMK_USE_SSE2 are set by configure's legacy modes
//      (--enable-arch avx, sse2, legacy-avx, legacy-sse2). If the compiler
//      lacks __AVX__ / __SSE2__ the flag is cleared below, so a mismatch
//      never silently runs the wrong vector code.
//   2. CMK_SSE is set when either legacy flag survives step 1. It selects the
//      hand-written intrinsic wrappers (SSE-Double.h / SSE-Float.h), which are
//      the original fork code and are kept unchanged for legacy builds.
//   3. Otherwise (CMK_SSE == 0) the native layer at the bottom of this file
//      uses std::experimental::simd. configure picks its width from the host
//      ISA, so no intrinsics appear on this path and no mode flag is needed.
//      Without a C++17 compiler, configure falls back to legacy modes instead.
#if  CMK_USE_AVX
	#if !defined(__AVX__)
		#undef CMK_USE_AVX
		#define CMK_USE_AVX 0
	#endif
#endif

#if CMK_USE_SSE2 && !defined(__SSE2__)
	#undef CMK_USE_SSE2
	#define CMK_USE_SSE2 0
#endif

#if CMK_USE_AVX || CMK_USE_SSE2
	#define CMK_SSE 1
#endif

#if CMK_USE_AVX
	#ifdef COSMO_FLOAT
		#error "single-precision AVX is not supported"
	#else
		#include "SSE-Double.h"
		#define SSE_VECTOR_WIDTH 4
		#define FORCE_INPUT_LIST_PAD 3
		typedef SSEDouble SSEcosmoType;
		#define SSELoad(where, arr, idx, field) where(arr[idx]field, arr[idx+1]field, arr[idx+2]field, arr[idx+3]field)
		#define SSEStore(what, arr, idx, field) { \
		  double p[4];  \
		  storeu(p, what); \
		  arr[idx]field = p[0]; \
		  arr[idx+1]field = p[1]; \
		  arr[idx+2]field = p[2]; \
		  arr[idx+3]field = p[3]; \
		}
		enum {cosmoMask=0xf};
	#endif
#elif CMK_USE_SSE2
	#ifdef COSMO_FLOAT
		#define SSE_COSMO_FLOAT
		#if defined(__SSE2__)
			#include "SSE-Float.h"
			#define SSE_VECTOR_WIDTH 4
			#define FORCE_INPUT_LIST_PAD 3
			typedef SSEFloat SSEcosmoType;
			#define SSELoad(where, arr, idx, field) where(arr[idx]field, arr[idx+1]field, arr[idx+2]field, arr[idx+3]field)
			#define SSEStore(what, arr, idx, field) { \
			  float p[4]; \
			  storeu(p, what); \
			  arr[idx]field = p[0]; \
			  arr[idx+1]field = p[1]; \
			  arr[idx+2]field = p[2]; \
			  arr[idx+3]field = p[3]; \
			}
			enum {cosmoMask=0xf};
                #else
                      #error("SSE not available");
                #endif
        #else
		#if defined(__SSE2__) && !defined(SSE_COSMO_FLOAT)
			#include "SSE-Double.h"
			#define SSE_VECTOR_WIDTH 2
			#define FORCE_INPUT_LIST_PAD 1
			typedef SSEDouble SSEcosmoType;
			#define SSELoad(where, arr, idx, field) where(arr[idx]field, arr[idx+1]field)
			#define SSEStore(what, arr, idx, field) { \
			  storel(&arr[idx]field, what); \
			  storeh(&arr[idx+1]field, what); \
			}
			enum {cosmoMask=0x3};
                #else
                      #error("SSE not available");
		#endif
	#endif
#endif

#if !CMK_SSE

// Native SIMD layer on std::experimental::simd (C++17 Library Fundamentals TS v2).
// Lane count and ABI come from the compiler's target flags (see configure.ac
// --enable-arch): float/double lanes are 16/8 (avx512), 8/4 (avx), 4/2 (sse2, neon).
// No hardware intrinsics are used here.
#include <experimental/simd>

// SSEcosmoType: one vector of cosmoType lanes, width chosen by the target ABI.
// SSEMask: per-lane boolean mask of the same width.
// SSE_VECTOR_WIDTH: lane count, usable as a compile-time constant.
using SSEcosmoType = std::experimental::native_simd<cosmoType>;
using SSEMask = std::experimental::native_simd_mask<cosmoType>;
#define SSE_VECTOR_WIDTH ((int)SSEcosmoType::size())

// Inputs: p, pointer to at least SSE_VECTOR_WIDTH elements (no alignment needed).
// Output: vector holding p[0..width-1].
// Role: gathers a packed block of interaction data into a SIMD register.
inline SSEcosmoType sseLoad(const cosmoType *p) {
  SSEcosmoType v;
  v.copy_from(p, std::experimental::element_aligned);
  return v;
}
// Inputs: p, destination of at least SSE_VECTOR_WIDTH elements; v, vector to write.
// Output: none; writes v's lanes to p[0..width-1].
// Role: spills a SIMD result back to the scalar arrays the tree walk reads.
inline void sseStore(cosmoType *p, SSEcosmoType v) {
  v.copy_to(p, std::experimental::element_aligned);
}
// Inputs: m, lane mask; t, value where m is set; f, value where m is clear.
// Output: per-lane (m ? t : f).
// Role: blends candidate vectors per lane under the walk's selection masks (select0, select1) in gravity.h.
inline SSEcosmoType sseSelect(SSEMask m, SSEcosmoType t, SSEcosmoType f) {
  SSEcosmoType r = f;
  std::experimental::where(m, r) = t;
  return r;
}
// Inputs: m, lane mask; x, value to keep where m is set.
// Output: per-lane (m ? x : +0.0). Masked-off lanes are zeroed even if x holds inf/NaN.
// Role: zeroes lanes outside the selection mask before they are accumulated into forces.
inline SSEcosmoType sseAnd(SSEMask m, SSEcosmoType x) {
  SSEcosmoType r(cosmoType(0));
  std::experimental::where(m, r) = x;
  return r;
}
// Inputs: m, lane mask. Output: true when every lane of m is clear.
// Role: lets the walk skip an interaction block when no lane is selected.
inline bool sseNone(SSEMask m) {
  return !std::experimental::any_of(m);
}
// Inputs: a, b, vectors. Output: per-lane (a > b ? a : b).
// Role: lane-wise running maximum, e.g. idt2 = max(idt2, packedDtGrav) in gravity.h.
inline SSEcosmoType sseMax(SSEcosmoType a, SSEcosmoType b) {
  return sseSelect(a > b, a, b);
}

#endif

#endif
