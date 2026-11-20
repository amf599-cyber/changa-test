#ifndef __GRAVITY_H__
#define __GRAVITY_H__

#include "TreeNode.h"
#include "GenericTreeNode.h"
#include "Space.h"
#include "SSEdefs.h"
#include <vector>

extern cosmoType theta;
extern cosmoType thetaMono;

/// A queued node contribution for a staged bucket. Moments are copied by value
/// so the queue does not depend on the tree node's lifetime.
struct BucketNodeTerm {
  MultipoleMoments m;
  Vector3D<cosmoType> cm;
};

/// A source particle queued for the same batched kernel as nodes.
/// base is the raw source position; off is the periodic offset, applied as
/// -packedPos + off + base to match the fork's SIMD operand order.
struct BucketPartTerm {
  Vector3D<cosmoType> off;
  Vector3D<cosmoType> base;
  cosmoType mass;
  cosmoType soft;
};

/// Queued terms per flush. Scales with the compile-time SIMD width.
/// The fork has no explicit batching. CHANGA_TERM_BATCH_MULT may be
/// overridden at build time to tune the batch size. The default of 8 gives
/// 16 terms per flush for SSE double and 32 for SSE float/AVX double.
#ifndef CHANGA_TERM_BATCH_MULT
#define CHANGA_TERM_BATCH_MULT 8
#endif
enum { TERM_BATCH_MAX = CHANGA_TERM_BATCH_MULT * SSE_VECTOR_WIDTH };

/// FORCE_SOA_FIELDS is the number of per-particle cosmoType arrays in
/// ActiveSoA (px through az), so one bind() buffer holds
/// FORCE_SOA_FIELDS * nPad values.
enum { FORCE_SOA_FIELDS = 10 };

/// Padded SoA of one bucket's active particles, with FORCE_SOA_FIELDS arrays
/// of nPad lanes each. Lanes from n to nPad are zeroed so whole SIMD vectors
/// can be loaded; the kernels' masks keep those lanes from contributing, and
/// suspendBucket reads only the first n lanes back.
/// @var n     Number of active particles.
/// @var nPad  n rounded up to a multiple of SSE_VECTOR_WIDTH.
/// @var buf   Owned storage that the pointers below index into.
/// @var px..az  Pointers into buf: position, softening, mass, and dtGrav
///            inputs; potential and acceleration outputs.
struct ActiveSoA {
  int n;
  int nPad;
  std::vector<cosmoType> buf;
  cosmoType *px, *py, *pz, *soft, *mass, *dtGrav, *potential, *ax, *ay, *az;

  /// Default constructor, initializes n and nPad to zero.
  ActiveSoA() : n(0), nPad(0) {}

  /// Rounds count up to a whole number of SIMD vectors.
  /// @param n     Particle count.
  /// @return      n rounded up to a multiple of SSE_VECTOR_WIDTH.
  static int padded(int n) {
    return ((n + SSE_VECTOR_WIDTH - 1) / SSE_VECTOR_WIDTH) * SSE_VECTOR_WIDTH;
  }

  /// Points the fields at base, which must hold FORCE_SOA_FIELDS *
  /// padded(count) values.
  void bind(cosmoType *base, int count) {
    n = count;
    nPad = padded(count);
    px = base;
    py = px + nPad;
    pz = py + nPad;
    soft = pz + nPad;
    mass = soft + nPad;
    dtGrav = mass + nPad;
    potential = dtGrav + nPad;
    ax = potential + nPad;
    ay = ax + nPad;
    az = ay + nPad;
  }

  /// Owns storage for count particles.
  void allocate(int count) {
    buf.assign(FORCE_SOA_FIELDS * padded(count), 0.0);
    bind(buf.data(), count);
  }

  /// Clears buffer and resets n and nPad to zero after release
  void release() {
    std::vector<cosmoType>().swap(buf);
    n = nPad = 0;
  }

  /// Copies the AoS particles act[0..n) into the SoA in one pass over the
  /// padded lanes, zeroing lanes from n to nPad.
  void gather(GravityParticle **act) {
    for (int i = 0; i < nPad; ++i) {
      const GravityParticle *p = i < n ? act[i] : nullptr;
      px[i] = p ? p->position.x : 0.0;
      py[i] = p ? p->position.y : 0.0;
      pz[i] = p ? p->position.z : 0.0;
      soft[i] = p ? p->soft : 0.0;
      mass[i] = p ? p->mass : 0.0;
      dtGrav[i] = p ? p->dtGrav : 0.0;
      potential[i] = p ? p->potential : 0.0;
      ax[i] = p ? p->treeAcceleration.x : 0.0;
      ay[i] = p ? p->treeAcceleration.y : 0.0;
      az[i] = p ? p->treeAcceleration.z : 0.0;
    }
  }
};

/// Per-bucket staging of active particles. One is owned by each TreePiece
/// bucket. It is gathered once when the bucket is staged and written back
/// when it is suspended or finished (finishBucket).
/// Node and particle terms share one walk-ordered queue: order[k] >= 0
/// indexes nodes, order[k] < 0 indexes parts as ~order[k]. flushTerms
/// applies them in that order, so each particle's accumulation matches the
/// scalar walk.
/// @var staged  True while soa holds gathered particles for this bucket.
/// @var soa     Padded SoA of the active particles.
/// @var nodes   Queued node terms (copied moments, walk order).
/// @var parts   Queued particle terms (offset + position precomputed).
/// @var order   Walk-order index into nodes (>= 0) or parts (~value < 0).
/// @var act     Active particles in gather order; suspendBucket writes back to them.
struct BucketSoA {
  bool staged;
  ActiveSoA soa;
  std::vector<BucketNodeTerm> nodes;
  std::vector<BucketPartTerm> parts;
  std::vector<int> order;
  std::vector<GravityParticle*> act;
  /// AoS treeAcceleration at stage time (x,y,z per particle). Other writers
  /// may add to the AoS while staged, so suspendBucket adds only the gravity delta.
  std::vector<cosmoType> snapAcc;
  BucketSoA() : staged(false) {}
};

/*
** see (A1) and (A2) of TREESPH: A UNIFICATION OF SPH WITH THE 
** HIERARCHICAL TREE METHOD by Lars Hernquist and Neal Katz.
** APJ Supplemant Series 70:416-446, 1989
** 
** Higher derivative terms c and d for use with quadrupole spline
** softening (Joachim Stadel, Dec. 94).
*/
#if !CMK_SSE
inline
void SPLINEQ(cosmoType invr,cosmoType r2,cosmoType twoh,cosmoType& a,
	     cosmoType& b,cosmoType& c,cosmoType& d)
{
  cosmoType u,dih,dir=(invr);
  if ((r2) < (twoh)*(twoh)) {
    dih = COSMO_CONST(2.0)/(twoh);
    u = dih/dir;
    if (u < COSMO_CONST(1.0)) {
      a = dih*(COSMO_CONST(7.0)/COSMO_CONST(5.0)
	       - COSMO_CONST(2.0)/COSMO_CONST(3.0)*u*u
	       + COSMO_CONST(3.0)/COSMO_CONST(10.0)*u*u*u*u
	       - COSMO_CONST(1.0)/COSMO_CONST(10.0)*u*u*u*u*u);
      b = dih*dih*dih*(COSMO_CONST(4.0)/COSMO_CONST(3.0)
		       - COSMO_CONST(6.0)/COSMO_CONST(5.0)*u*u
		       + COSMO_CONST(1.0)/COSMO_CONST(2.0)*u*u*u);
      c = dih*dih*dih*dih*dih*(COSMO_CONST(12.0)/COSMO_CONST(5.0)
			       - COSMO_CONST(3.0)/COSMO_CONST(2.0)*u);
      d = COSMO_CONST(3.0)/COSMO_CONST(2.0)*dih*dih*dih*dih*dih*dih*dir;
    }
    else {
      a = COSMO_CONST(-1.0)/COSMO_CONST(15.0)*dir
	+ dih*(COSMO_CONST(8.0)/COSMO_CONST(5.0)
	       - COSMO_CONST(4.0)/COSMO_CONST(3.0)*u*u + u*u*u
	       - COSMO_CONST(3.0)/COSMO_CONST(10.0)*u*u*u*u
	       + COSMO_CONST(1.0)/COSMO_CONST(30.0)*u*u*u*u*u);
      b = COSMO_CONST(-1.0)/COSMO_CONST(15.0)*dir*dir*dir
	+ dih*dih*dih*(COSMO_CONST(8.0)/COSMO_CONST(3.0) - COSMO_CONST(3.0)*u
		       + COSMO_CONST(6.0)/COSMO_CONST(5.0)*u*u
		       - COSMO_CONST(1.0)/COSMO_CONST(6.0)*u*u*u);
      c = COSMO_CONST(-1.0)/COSMO_CONST(5.0)*dir*dir*dir*dir*dir
	+ COSMO_CONST(3.0)*dih*dih*dih*dih*dir
	+ dih*dih*dih*dih*dih*(COSMO_CONST(-12.0)/COSMO_CONST(5.0)
			       + COSMO_CONST(1.0)/COSMO_CONST(2.0)*u);
      d = -dir*dir*dir*dir*dir*dir*dir
	+ COSMO_CONST(3.0)*dih*dih*dih*dih*dir*dir*dir
	- COSMO_CONST(1.0)/COSMO_CONST(2.0)*dih*dih*dih*dih*dih*dih*dir;
    }
  }
  else {
    a = dir;
    b = a*a*a;
    c = COSMO_CONST(3.0)*b*a*a;
    d = COSMO_CONST(5.0)*c*a*a;
  }
}
#else
inline
void SPLINEQ(SSEcosmoType invr, SSEcosmoType r2, SSEcosmoType twoh,
	     SSEcosmoType& a, SSEcosmoType& b, 
	     SSEcosmoType& c, SSEcosmoType& d)
{
  SSEcosmoType dir;
  dir = invr;
  SSEcosmoType select0 = r2 < twoh * twoh;
  int compare0 = movemask(select0);
  // make common case fast, at the cost of some redundant code
  //   in the infrequent case
  if (!compare0) {
    a = dir;
    b = a*a*a;
    c = COSMO_CONST(3.0)*b*a*a;
    d = COSMO_CONST(5.0)*c*a*a;
  }
  else {
    SSEcosmoType u, dih;
    SSEcosmoType select1;
    SSEcosmoType a0,b0,c0,d0,a1,b1,c1,d1,a2,b2,c2,d2;
    int compare1;

    dih = COSMO_CONST(2.0)/twoh;
    u = dih/dir;

    if ((~compare0) & cosmoMask) {
      a0 = dir;
      b0 = a0*a0*a0;
      c0 = COSMO_CONST(3.0)*b0*a0*a0;
      d0 = COSMO_CONST(5.0)*c0*a0*a0;
    }

    select1 = u < COSMO_CONST(1.0);
    compare1 = movemask(select1);
    if (compare1) {
      a1 = dih*(COSMO_CONST(7.0)/COSMO_CONST(5.0)
	       - COSMO_CONST(2.0)/COSMO_CONST(3.0)*u*u
	       + COSMO_CONST(3.0)/COSMO_CONST(10.0)*u*u*u*u
	       - COSMO_CONST(1.0)/COSMO_CONST(10.0)*u*u*u*u*u);
      b1 = dih*dih*dih*(COSMO_CONST(4.0)/COSMO_CONST(3.0)
		       - COSMO_CONST(6.0)/COSMO_CONST(5.0)*u*u
		       + COSMO_CONST(1.0)/COSMO_CONST(2.0)*u*u*u);
      c1 = dih*dih*dih*dih*dih*(COSMO_CONST(12.0)/COSMO_CONST(5.0)
			       - COSMO_CONST(3.0)/COSMO_CONST(2.0)*u);
      d1 = COSMO_CONST(3.0)/COSMO_CONST(2.0)*dih*dih*dih*dih*dih*dih*dir;
    }
    if ((~compare1) & cosmoMask) {
      a2 = COSMO_CONST(-1.0)/COSMO_CONST(15.0)*dir
	+ dih*(COSMO_CONST(8.0)/COSMO_CONST(5.0)
	       - COSMO_CONST(4.0)/COSMO_CONST(3.0)*u*u + u*u*u
	       - COSMO_CONST(3.0)/COSMO_CONST(10.0)*u*u*u*u
	       + COSMO_CONST(1.0)/COSMO_CONST(30.0)*u*u*u*u*u);
      b2 = COSMO_CONST(-1.0)/COSMO_CONST(15.0)*dir*dir*dir
	+ dih*dih*dih*(COSMO_CONST(8.0)/COSMO_CONST(3.0) - COSMO_CONST(3.0)*u
		       + COSMO_CONST(6.0)/COSMO_CONST(5.0)*u*u
		       - COSMO_CONST(1.0)/COSMO_CONST(6.0)*u*u*u);
      c2 = COSMO_CONST(-1.0)/COSMO_CONST(5.0)*dir*dir*dir*dir*dir
	+ COSMO_CONST(3.0)*dih*dih*dih*dih*dir
	+ dih*dih*dih*dih*dih*(COSMO_CONST(-12.0)/COSMO_CONST(5.0)
			       + COSMO_CONST(1.0)/COSMO_CONST(2.0)*u);
      d2 = -dir*dir*dir*dir*dir*dir*dir
	+ COSMO_CONST(3.0)*dih*dih*dih*dih*dir*dir*dir
	- COSMO_CONST(1.0)/COSMO_CONST(2.0)*dih*dih*dih*dih*dih*dih*dir;
    }

    a = andnot(select0, a0)
      | (select0 & ((select1 & a1) | andnot(select1, a2)));
    b = andnot(select0, b0)
      | (select0 & ((select1 & b1) | andnot(select1, b2)));
    c = andnot(select0, c0)
      | (select0 & ((select1 & c1) | andnot(select1, c2)));
    d = andnot(select0, d0)
      | (select0 & ((select1 & d1) | andnot(select1, d2)));
  }
}
#endif

#if !CMK_SSE
/// Native SPLINEQ on SIMD lanes (std::experimental::simd). Named SPLINEQNative
/// so it does not overload the fork's SPLINEQ above.
/// @param invr   1/r per lane.
/// @param r2     Squared distance per lane.
/// @param twoh   Softening length (2h) per lane.
/// @param a,b,c,d  [out] Potential and derivative factors per lane.
inline
void SPLINEQNative(SSEcosmoType invr, SSEcosmoType r2, SSEcosmoType twoh,
	     SSEcosmoType& a, SSEcosmoType& b,
	     SSEcosmoType& c, SSEcosmoType& d)
{
  SSEcosmoType dir = invr;
  SSEMask select0 = r2 < twoh * twoh;
  // Common case: no lane inside the softening radius. Same arithmetic as the
  // a0..d0 regime below, so results are bit-identical.
  if (sseNone(select0)) {
    a = dir;
    b = dir*dir*dir;
    c = COSMO_CONST(3.0)*b*dir*dir;
    d = COSMO_CONST(5.0)*c*dir*dir;
    return;
  }
  // Every regime is evaluated on all lanes and then selected per lane.
  // Lanes outside a regime may hold inf/NaN; the selects discard them.
  SSEcosmoType dih = COSMO_CONST(2.0)/twoh;
  SSEcosmoType u = dih/dir;
  SSEMask select1 = u < COSMO_CONST(1.0);

  // Operation order matches the fork's SSE SPLINEQ (direct powers, left to
  // right) so results are bit-identical to the fork.
  SSEcosmoType a0 = dir;
  SSEcosmoType b0 = a0*a0*a0;
  SSEcosmoType c0 = COSMO_CONST(3.0)*b0*a0*a0;
  SSEcosmoType d0 = COSMO_CONST(5.0)*c0*a0*a0;

  SSEcosmoType a1 = dih*(COSMO_CONST(7.0)/COSMO_CONST(5.0)
			 - COSMO_CONST(2.0)/COSMO_CONST(3.0)*u*u
			 + COSMO_CONST(3.0)/COSMO_CONST(10.0)*u*u*u*u
			 - COSMO_CONST(1.0)/COSMO_CONST(10.0)*u*u*u*u*u);
  SSEcosmoType b1 = dih*dih*dih*(COSMO_CONST(4.0)/COSMO_CONST(3.0)
				 - COSMO_CONST(6.0)/COSMO_CONST(5.0)*u*u
				 + COSMO_CONST(1.0)/COSMO_CONST(2.0)*u*u*u);
  SSEcosmoType c1 = dih*dih*dih*dih*dih*(COSMO_CONST(12.0)/COSMO_CONST(5.0)
					 - COSMO_CONST(3.0)/COSMO_CONST(2.0)*u);
  SSEcosmoType d1 = COSMO_CONST(3.0)/COSMO_CONST(2.0)*dih*dih*dih*dih*dih
    *dih*dir;

  SSEcosmoType a2 = COSMO_CONST(-1.0)/COSMO_CONST(15.0)*dir
    + dih*(COSMO_CONST(8.0)/COSMO_CONST(5.0)
	   - COSMO_CONST(4.0)/COSMO_CONST(3.0)*u*u + u*u*u
	   - COSMO_CONST(3.0)/COSMO_CONST(10.0)*u*u*u*u
	   + COSMO_CONST(1.0)/COSMO_CONST(30.0)*u*u*u*u*u);
  SSEcosmoType b2 = COSMO_CONST(-1.0)/COSMO_CONST(15.0)*dir*dir*dir
    + dih*dih*dih*(COSMO_CONST(8.0)/COSMO_CONST(3.0) - COSMO_CONST(3.0)*u
		   + COSMO_CONST(6.0)/COSMO_CONST(5.0)*u*u
		   - COSMO_CONST(1.0)/COSMO_CONST(6.0)*u*u*u);
  SSEcosmoType c2 = COSMO_CONST(-1.0)/COSMO_CONST(5.0)*dir*dir*dir*dir*dir
    + COSMO_CONST(3.0)*dih*dih*dih*dih*dir
    + dih*dih*dih*dih*dih*(COSMO_CONST(-12.0)/COSMO_CONST(5.0)
			   + COSMO_CONST(1.0)/COSMO_CONST(2.0)*u);
  SSEcosmoType d2 = -dir*dir*dir*dir*dir*dir*dir
    + COSMO_CONST(3.0)*dih*dih*dih*dih*dir*dir*dir
    - COSMO_CONST(1.0)/COSMO_CONST(2.0)*dih*dih*dih*dih*dih*dih*dir;

  a = sseSelect(select0, sseSelect(select1, a1, a2), a0);
  b = sseSelect(select0, sseSelect(select1, b1, b2), b0);
  c = sseSelect(select0, sseSelect(select1, c1, c2), c0);
  d = sseSelect(select0, sseSelect(select1, d1, d2), d0);
}
#endif

/// Spline softening kernel, single SIMD register. Computes the potential
/// factor a and force factor b for each lane. With u = 2r / twoh, the
/// regimes are r >= twoh (plain 1/r), 1 <= u < 2 (outer spline), and
/// u < 1 (inner spline). Every regime is evaluated and selected per lane.
/// @param r2    Squared distance per lane.
/// @param twoh  Softening length (2h) per lane.
/// @param a     [out] Potential factor per lane.
#if !CMK_SSE
inline
void SPLINE(cosmoType r2, cosmoType twoh, cosmoType &a, cosmoType &b)
{
  cosmoType r, u,dih,dir;
  r = sqrt(r2);

  if (r < (twoh)) {
    dih = COSMO_CONST(2.0)/(twoh);
    u = r*dih;
    if (u < COSMO_CONST(1.0)) {
      a = dih*(COSMO_CONST(7.0)/COSMO_CONST(5.0)
	       - COSMO_CONST(2.0)/COSMO_CONST(3.0)*u*u
	       + COSMO_CONST(3.0)/COSMO_CONST(10.0)*u*u*u*u
	       - COSMO_CONST(1.0)/COSMO_CONST(10.0)*u*u*u*u*u);
      b = dih*dih*dih*(COSMO_CONST(4.0)/COSMO_CONST(3.0)
		       - COSMO_CONST(6.0)/COSMO_CONST(5.0)*u*u
		       + COSMO_CONST(1.0)/COSMO_CONST(2.0)*u*u*u);
    }
    else {
      dir = COSMO_CONST(1.0)/r;
      a = COSMO_CONST(-1.0)/COSMO_CONST(15.0)*dir
	+ dih*(COSMO_CONST(8.0)/COSMO_CONST(5.0)
	       - COSMO_CONST(4.0)/COSMO_CONST(3.0)*u*u + u*u*u
	       - COSMO_CONST(3.0)/COSMO_CONST(10.0)*u*u*u*u
	       + COSMO_CONST(1.0)/COSMO_CONST(30.0)*u*u*u*u*u);
      b = COSMO_CONST(-1.0)/COSMO_CONST(15.0)*dir*dir*dir
	+ dih*dih*dih*(COSMO_CONST(8.0)/COSMO_CONST(3.0) - COSMO_CONST(3.0)*u
		       + COSMO_CONST(6.0)/COSMO_CONST(5.0)*u*u
		       - COSMO_CONST(1.0)/COSMO_CONST(6.0)*u*u*u);
    }
  }
  else {
    a = COSMO_CONST(1.0)/r;
    b = a*a*a;
  }
}
#else
inline
void SPLINE(SSEcosmoType r2, SSEcosmoType twoh,
	    SSEcosmoType &a, SSEcosmoType &b)
{
  SSEcosmoType r;
  r = sqrt(r2);
  SSEcosmoType select0 = r < twoh;
  int compare0 = movemask(select0);
  // make common case fast, at the cost of some redundant code
  //   in the infrequent case
  if (!compare0) {
    a = COSMO_CONST(1.0)/r;
    b = a * a * a;
  }
  else {
    SSEcosmoType u, dih, dir;
    SSEcosmoType a0, b0, a1, b1, a2, b2;
    SSEcosmoType select1;
    int compare1;

    dih = COSMO_CONST(2.0)/twoh;
    u = r * dih;

    if ((~compare0) & cosmoMask) {
      a0 = COSMO_CONST(1.0)/r;
      b0 = a0 * a0 * a0;
    }

    select1 = u < COSMO_CONST(1.0);
    compare1 = movemask(select1);
    if (compare1) {
      a1 = dih*(COSMO_CONST(7.0)/COSMO_CONST(5.0)
		- COSMO_CONST(2.0)/COSMO_CONST(3.0)*u*u
		+ COSMO_CONST(3.0)/COSMO_CONST(10.0)*u*u*u*u
		- COSMO_CONST(1.0)/COSMO_CONST(10.0)*u*u*u*u*u);
      b1 = dih*dih*dih*(COSMO_CONST(4.0)/COSMO_CONST(3.0)
			- COSMO_CONST(6.0)/COSMO_CONST(5.0)*u*u
			+ COSMO_CONST(1.0)/COSMO_CONST(2.0)*u*u*u);
    }
    if ((~compare1) & cosmoMask) {
      dir = COSMO_CONST(1.0)/r;
      a2 = COSMO_CONST(-1.0)/COSMO_CONST(15.0)*dir
	+ dih*(COSMO_CONST(8.0)/COSMO_CONST(5.0)
	       - COSMO_CONST(4.0)/COSMO_CONST(3.0)*u*u + u*u*u
	       - COSMO_CONST(3.0)/COSMO_CONST(10.0)*u*u*u*u
	       + COSMO_CONST(1.0)/COSMO_CONST(30.0)*u*u*u*u*u);
      b2 = COSMO_CONST(-1.0)/COSMO_CONST(15.0)*dir*dir*dir
	+ dih*dih*dih*(COSMO_CONST(8.0)/COSMO_CONST(3.0) - COSMO_CONST(3.0)*u
		       + COSMO_CONST(6.0)/COSMO_CONST(5.0)*u*u
		       - COSMO_CONST(1.0)/COSMO_CONST(6.0)*u*u*u);
    }

    a = andnot(select0, a0)
      | (select0 & ((select1 & a1) | andnot(select1, a2)));
    b = andnot(select0, b0)
      | (select0 & ((select1 & b1) | andnot(select1, b2)));
  }
}
#endif

#if !CMK_SSE
/// Native SPLINE on SIMD lanes (std::experimental::simd). Named SPLINENative
/// so it does not overload the fork's SPLINE above.
/// @param r2    Squared distance per lane.
/// @param twoh  Softening length (2h) per lane.
/// @param a     [out] Potential factor per lane.
/// @param b     [out] Force factor per lane (a*a*a in the plain regime).
inline
void SPLINENative(SSEcosmoType r2, SSEcosmoType twoh,
	    SSEcosmoType &a, SSEcosmoType &b)
{
  SSEcosmoType r = sqrt(r2);
  SSEMask select0 = r < twoh;
  if (sseNone(select0)) {
    a = COSMO_CONST(1.0)/r;
    b = a*a*a;
    return;
  }
  // Every regime is evaluated on all lanes and then selected per lane.
  // Lanes outside a regime may hold inf/NaN; the selects discard them.
  SSEcosmoType dih = COSMO_CONST(2.0)/twoh;
  SSEcosmoType u = r * dih;
  SSEcosmoType dir = COSMO_CONST(1.0)/r;
  SSEMask select1 = u < COSMO_CONST(1.0);

  // Operation order matches the fork's SSE SPLINE (direct powers, left to
  // right) so results are bit-identical to the fork.
  SSEcosmoType a0 = dir;
  SSEcosmoType b0 = a0 * a0 * a0;

  SSEcosmoType a1 = dih*(COSMO_CONST(7.0)/COSMO_CONST(5.0)
			 - COSMO_CONST(2.0)/COSMO_CONST(3.0)*u*u
			 + COSMO_CONST(3.0)/COSMO_CONST(10.0)*u*u*u*u
			 - COSMO_CONST(1.0)/COSMO_CONST(10.0)*u*u*u*u*u);
  SSEcosmoType b1 = dih*dih*dih*(COSMO_CONST(4.0)/COSMO_CONST(3.0)
				 - COSMO_CONST(6.0)/COSMO_CONST(5.0)*u*u
				 + COSMO_CONST(1.0)/COSMO_CONST(2.0)*u*u*u);

  SSEcosmoType a2 = COSMO_CONST(-1.0)/COSMO_CONST(15.0)*dir
    + dih*(COSMO_CONST(8.0)/COSMO_CONST(5.0)
	   - COSMO_CONST(4.0)/COSMO_CONST(3.0)*u*u + u*u*u
	   - COSMO_CONST(3.0)/COSMO_CONST(10.0)*u*u*u*u
	   + COSMO_CONST(1.0)/COSMO_CONST(30.0)*u*u*u*u*u);
  SSEcosmoType b2 = COSMO_CONST(-1.0)/COSMO_CONST(15.0)*dir*dir*dir
    + dih*dih*dih*(COSMO_CONST(8.0)/COSMO_CONST(3.0) - COSMO_CONST(3.0)*u
		   + COSMO_CONST(6.0)/COSMO_CONST(5.0)*u*u
		   - COSMO_CONST(1.0)/COSMO_CONST(6.0)*u*u*u);

  a = sseSelect(select0, sseSelect(select1, a1, a2), a0);
  b = sseSelect(select0, sseSelect(select1, b1, b2), b0);
}
#endif

//
// Return true if the soften nodes overlap, or if the source node's
// softening overlaps the bounding box; i.e. the forces involve softening
// @param node Node to test
// @param myNode
// @param offset Periodic offset applied to node
//
inline
int openSoftening(Tree::GenericTreeNode *node, Tree::GenericTreeNode *myNode,
		  Vector3D<cosmoType> offset)
{
  Sphere<cosmoType> s(node->moments.cm + offset, 2.0*node->moments.soft);
  Sphere<cosmoType> myS(myNode->moments.cm, 2.0*myNode->moments.soft);
  if(Space::intersect(myS, s))
      return true;
  return Space::intersect(myNode->boundingBox, s);
}

#if !CMK_SSE
#ifdef CMK_VERSION_BLUEGENE
static int forProgress = 0;
#endif

// Masked SIMD kernel over padded SoA lanes. Terms are applied in queue
// order for every register, so each lane accumulates exactly as the scalar
// walk does. Node terms follow nodeBucketForce; particle terms follow
// partBucketForce. Positions and accelerations are carried as separate x, y,
// z native_simd registers rather than Vector3D, so the kernel uses no Vector3D.
/// @param nodes  Queued node terms, indexed by order[k] >= 0 (may be null
///               when nt has no nodes).
/// @param parts  Queued particle terms, indexed by ~order[k] < 0.
/// @param order  Walk-order list of term indices, length nt.
/// @param nt     Number of queued terms to apply.
/// @param s      Padded SoA whose force fields are accumulated in place.
/// @param nPad   Lane count of s (a multiple of SSE_VECTOR_WIDTH).
inline void termBatchSoA(const BucketNodeTerm *nodes,
			 const BucketPartTerm *parts,
			 const int *order, int nt,
			 const ActiveSoA &s, int nPad) {
  SSEcosmoType rx, ry, rz;
  SSEcosmoType rsq;
  SSEcosmoType twoh;
  SSEcosmoType a,b,c,d;

  for (int i=0; i<nPad; i+=SSE_VECTOR_WIDTH) {
#ifdef CMK_VERSION_BLUEGENE
    if (++forProgress > 200) {
      forProgress = 0;
#ifdef COSMO_EVENTS
      traceUserEvents(networkProgressUE);
#endif
      CmiNetworkProgress();
    }
#endif
    SSEcosmoType packedX = sseLoad(s.px + i);
    SSEcosmoType packedY = sseLoad(s.py + i);
    SSEcosmoType packedZ = sseLoad(s.pz + i);
    SSEcosmoType accX = sseLoad(s.ax + i);
    SSEcosmoType accY = sseLoad(s.ay + i);
    SSEcosmoType accZ = sseLoad(s.az + i);
    SSEcosmoType packedPotential = sseLoad(s.potential + i);
    SSEcosmoType packedMass = sseLoad(s.mass + i);
    SSEcosmoType packedDtGrav = sseLoad(s.dtGrav + i);
    SSEcosmoType packedSoft = sseLoad(s.soft + i);
    for (int k=0; k<nt; ++k) {
    if (order[k] < 0) {
      const BucketPartTerm &p = parts[~order[k]];
      rx = (-packedX + SSEcosmoType(p.off.x)) + SSEcosmoType(p.base.x);
      ry = (-packedY + SSEcosmoType(p.off.y)) + SSEcosmoType(p.base.y);
      rz = (-packedZ + SSEcosmoType(p.off.z)) + SSEcosmoType(p.base.z);
      rsq = rx*rx + ry*ry + rz*rz;
      twoh = SSEcosmoType(p.soft) + packedSoft;
      SSEMask select = rsq > COSMO_CONST(0.0);
      // Matches partBucketForce: no active lane, so skip the spline and update.
      if (sseNone(select)) continue;
      SPLINENative(rsq, twoh, a, b);
      a = sseAnd(select, a);
      b = sseAnd(select, b);
      SSEcosmoType idt2 = (packedMass + SSEcosmoType(p.mass)) * b;
      idt2 = sseMax(idt2, packedDtGrav);
      SSEcosmoType bm = b * SSEcosmoType(p.mass);
      accX += rx * bm;
      accY += ry * bm;
      accZ += rz * bm;
      packedPotential -= SSEcosmoType(p.mass) * a;
      packedDtGrav = idt2;
      continue;
    }
    const MultipoleMoments &m = nodes[order[k]].m;
    const Vector3D<cosmoType> &cm = nodes[order[k]].cm;
    rx = packedX - cm.x;
    ry = packedY - cm.y;
    rz = packedZ - cm.z;
    rsq = rx*rx + ry*ry + rz*rz;
    SSEMask select = rsq > COSMO_CONST(0.0);
    // Self and padded lanes only: nothing to accumulate.
    if (sseNone(select)) continue;
    /* Masked-off lanes: 1/sqrt(0) is inf for self and padded lanes. */
    SSEcosmoType dir = sseAnd(select, COSMO_CONST(1.0)/sqrt(rsq));
#ifdef HEXADECAPOLE
    SSEcosmoType magai;
    momEvalFmomrcmNative(&m.mom, m.getRadius(), dir, rx, ry, rz,
			 &packedPotential, &accX, &accY, &accZ,
			 &magai);
    SSEcosmoType idt2 = (packedMass + m.totalMass)*dir*dir*dir;
#else
    twoh = CONVERT_TO_COSMO_TYPE m.soft + packedSoft;
    SPLINEQNative(dir, rsq, twoh, a, b, c, d);
    a = sseAnd(select, a);
    b = sseAnd(select, b);
    c = sseAnd(select, c);
    d = sseAnd(select, d);
    SSEcosmoType qix = SSEcosmoType(CONVERT_TO_COSMO_TYPE m.xx)*rx
      + SSEcosmoType(CONVERT_TO_COSMO_TYPE m.xy)*ry
      + SSEcosmoType(CONVERT_TO_COSMO_TYPE m.xz)*rz;
    SSEcosmoType qiy = SSEcosmoType(CONVERT_TO_COSMO_TYPE m.xy)*rx
      + SSEcosmoType(CONVERT_TO_COSMO_TYPE m.yy)*ry
      + SSEcosmoType(CONVERT_TO_COSMO_TYPE m.yz)*rz;
    SSEcosmoType qiz = SSEcosmoType(CONVERT_TO_COSMO_TYPE m.xz)*rx
      + SSEcosmoType(CONVERT_TO_COSMO_TYPE m.yz)*ry
      + SSEcosmoType(CONVERT_TO_COSMO_TYPE m.zz)*rz;
    SSEcosmoType qir = COSMO_CONST(0.5)*(qix*rx + qiy*ry + qiz*rz);
    SSEcosmoType tr = COSMO_CONST(0.5)*(CONVERT_TO_COSMO_TYPE m.xx
					+ CONVERT_TO_COSMO_TYPE m.yy
					+ CONVERT_TO_COSMO_TYPE m.zz);
    SSEcosmoType qir3 = b*CONVERT_TO_COSMO_TYPE m.totalMass + d*qir - c*tr;
    packedPotential -= CONVERT_TO_COSMO_TYPE m.totalMass *a
      + c * qir - b * tr;
    accX -= rx*qir3 - qix*c;
    accY -= ry*qir3 - qiy*c;
    accZ -= rz*qir3 - qiz*c;
    SSEcosmoType idt2 = (packedMass + CONVERT_TO_COSMO_TYPE m.totalMass)*b;
#endif
    idt2 = sseMax(idt2, packedDtGrav);
    packedDtGrav = idt2;
    }
    sseStore(s.ax + i, accX);
    sseStore(s.ay + i, accY);
    sseStore(s.az + i, accZ);
    sseStore(s.potential + i, packedPotential);
    sseStore(s.dtGrav + i, packedDtGrav);
  }
}

/// Applies all queued node and particle terms to the bucket's SoA in walk
/// order, then clears the queues.
/// @param bs  [in/out] Bucket whose term queues are drained into bs.soa.
inline void flushTerms(BucketSoA &bs) {
  if (bs.order.empty())
    return;
  termBatchSoA(bs.nodes.data(), bs.parts.data(), bs.order.data(),
	       (int)bs.order.size(), bs.soa, bs.soa.nPad);
  bs.nodes.clear();
  bs.parts.clear();
  bs.order.clear();
}
#endif

/// @brief Gravity opening criterion for a bucket walk.
/// @param node Source node to be tested
/// @param bucketNode Target bucket
/// @param offset Offset of periodic replica
/// @return True if the node's opening radius intersects the
/// boundingBox of the bucket, i.e. the node needs to be opened.

inline bool
openCriterionBucket(Tree::GenericTreeNode *node,
                   Tree::GenericTreeNode *bucketNode,
                   Vector3D<cosmoType> offset // Offset of node
                   ) {

#if COSMO_STATS > 0
  node->used = true;
#endif
  // Always open node if this many particles or fewer.
  const int nMinParticleNode = 6;
  if(node->particleCount <= nMinParticleNode) {
      return true;
      }

  // Note that some of this could be pre-calculated into an "opening radius"
  cosmoType radius = TreeStuff::opening_geometry_factor * node->moments.getRadius() / theta;
  if(radius < node->moments.getRadius())
      radius = node->moments.getRadius();

  Sphere<cosmoType> s(node->moments.cm + offset, radius);
  
#ifdef HEXADECAPOLE
  if(!Space::intersect(bucketNode->boundingBox, s)) {
      // Well separated, now check softening
      if(!openSoftening(node, bucketNode, offset)) {
	  return false; // passed both tests: will be a Hex interaction
      }
      else {        // Open as monopole?
        radius = TreeStuff::opening_geometry_factor*node->moments.getRadius()/thetaMono;
      Sphere<cosmoType> sM(node->moments.cm + offset, radius);
      return Space::intersect(bucketNode->boundingBox, sM);
      }
      }
  return true;
#else
  return Space::intersect(bucketNode->boundingBox, s);
#endif
}

/// @brief Gravity opening criterion for "double walk".
/// @param node Source node to be tested
/// @param myNode Target node
/// @param offset Offset for periodic replica
/// @return -1 if there is an intersection
///    0 if no intersection
///    +1 if completely contained
///
/// This is the criterion described in Stadel(2001): if no
/// intersection, then the node is valid for all particles in myNode.
/// If completely contained then it needs to be opened for all
/// particles in myNode.  If a partial intersection, then node needs
/// to be checked by the children of myNode.  If myNode is already a
/// bucket, then only returns 0 or 1 are possible.

inline int openCriterionNode(Tree::GenericTreeNode *node,
                    Tree::GenericTreeNode *myNode,
                    Vector3D<cosmoType> offset
                    ) {

#if COSMO_STATS > 0
  node->used = true;
#endif
  // Always open node if this many particles or fewer.
  const int nMinParticleNode = 6;
  if(node->particleCount <= nMinParticleNode) {
      return 1;
      }

  // Note that some of this could be pre-calculated into an "opening radius"
  cosmoType radius = TreeStuff::opening_geometry_factor * node->moments.getRadius() / theta;
  if(radius < node->moments.getRadius())
      radius = node->moments.getRadius();

  Sphere<cosmoType> s(node->moments.cm + offset, radius);

  if(myNode->getType()==Tree::Bucket || myNode->getType()==Tree::CachedBucket || myNode->getType()==Tree::NonLocalBucket){
    if(Space::intersect(myNode->boundingBox, s))
        return 1;
    else
#ifdef HEXADECAPOLE
        {
        // Well separated, now check softening
        if(!openSoftening(node, myNode, offset)) {
            return 0;   // passed both tests: will be a Hex interaction
            }
        else {      // Open as monopole?
          radius = TreeStuff::opening_geometry_factor*node->moments.getRadius()/thetaMono;
            Sphere<cosmoType> sM(node->moments.cm + offset, radius);
            if(Space::intersect(myNode->boundingBox, sM))
                return 1;
            else
                return 0;
            }
        }
#else
    return 0;
#endif
    }
    else{
        if(Space::intersect(myNode->boundingBox, s)){
            if(Space::contained(myNode->boundingBox,s))
                return 1;
            else
                return -1;
        }
        else
#ifdef HEXADECAPOLE
            {
            // Well separated, now check softening
            if(!openSoftening(node, myNode, offset)) {
                return 0;   // passed both tests: will be a Hex interaction
                }
            else {      // Open as monopole?
                radius = TreeStuff::opening_geometry_factor*node->moments.getRadius()/thetaMono;
                Sphere<cosmoType> sM(node->moments.cm + offset, radius);
                if(Space::intersect(myNode->boundingBox, sM))
                    return 1;
                else
                    return 0;
                }
            }
#else
        return 0;
#endif
    }
}

#if !CMK_SSE
// Bucket-level staging. stageBucket gathers the bucket's active particles
// into bs. The bucket overloads below queue terms on that SoA until
// suspendBucket writes the results back to the AoS particles.
// suspendBucket releases the SoA; a later bucketSoAFor restages it from the
// AoS, which already holds the accumulated values.
/// Gathers the bucket's active particles (rung >= activeRung) into bs and
/// snapshots their accelerations so suspendBucket can detect external changes.
/// @param bs         [out] Bucket staging state (SoA, snapshot, term queues).
/// @param req        Bucket node whose particle range is staged.
/// @param particles  AoS particles of the TreePiece.
/// @param activeRung Lowest rung treated as active.
inline void stageBucket(BucketSoA &bs, Tree::GenericTreeNode *req,
			GravityParticle *particles, int activeRung) {
  bs.act.clear();
  bs.nodes.clear();
  bs.parts.clear();
  bs.order.clear();
  bs.nodes.reserve(TERM_BATCH_MAX);
  bs.parts.reserve(TERM_BATCH_MAX);
  bs.order.reserve(TERM_BATCH_MAX);
  for (int j = req->firstParticle; j <= req->lastParticle; ++j) {
    if (particles[j].rung >= activeRung)
      bs.act.push_back(&particles[j]);
  }
  bs.soa.allocate((int)bs.act.size());
  bs.soa.gather(bs.act.data());
  bs.snapAcc.resize(3 * bs.act.size());
  for (size_t i = 0; i < bs.act.size(); ++i) {
    bs.snapAcc[3*i] = bs.act[i]->treeAcceleration.x;
    bs.snapAcc[3*i+1] = bs.act[i]->treeAcceleration.y;
    bs.snapAcc[3*i+2] = bs.act[i]->treeAcceleration.z;
  }
  bs.staged = true;
}

// Flushes queued terms and writes the SoA back to the AoS, releasing the
// SoA. Leaves the bucket restageable so it can be suspended around Ewald,
// which adds to the AoS directly, and finished once the walk ends.
/// @param bs  [in/out] Staged bucket; on return it is unstaged and its SoA
///            released. AoS accelerations, potential and dtGrav are updated.
inline void suspendBucket(BucketSoA &bs) {
  if (bs.staged) {
    flushTerms(bs);
    for (int i = 0; i < bs.soa.n; ++i) {
      GravityParticle *p = bs.act[i];
      // Assign unless the AoS changed while staged; then keep that external delta.
      // Assigning avoids rounding (soa - snap) + aos, which is not bit-exact in float.
      if (p->treeAcceleration.x != bs.snapAcc[3*i])
        p->treeAcceleration.x = bs.soa.ax[i] + (p->treeAcceleration.x - bs.snapAcc[3*i]);
      else p->treeAcceleration.x = bs.soa.ax[i];
      if (p->treeAcceleration.y != bs.snapAcc[3*i+1])
        p->treeAcceleration.y = bs.soa.ay[i] + (p->treeAcceleration.y - bs.snapAcc[3*i+1]);
      else p->treeAcceleration.y = bs.soa.ay[i];
      if (p->treeAcceleration.z != bs.snapAcc[3*i+2])
        p->treeAcceleration.z = bs.soa.az[i] + (p->treeAcceleration.z - bs.snapAcc[3*i+2]);
      else p->treeAcceleration.z = bs.soa.az[i];
      p->potential = bs.soa.potential[i];
      p->dtGrav = bs.soa.dtGrav[i];
    }
  }
  bs.staged = false;
  bs.soa.release();
  std::vector<GravityParticle*>().swap(bs.act);
}
#endif

#if !CMK_SSE
inline int partBucketForce(ExternalGravityParticle *part,
			   Tree::GenericTreeNode *req,
			   GravityParticle *particles,
			   Vector3D<cosmoType> offset, int activeRung) {
  int computed = 0;
  Vector3D<cosmoType> r;
  cosmoType rsq;
  cosmoType twoh, a, b;

  for(int j = req->firstParticle; j <= req->lastParticle; ++j) {
    if (particles[j].rung >= activeRung) {
#ifdef CMK_VERSION_BLUEGENE
      if (++forProgress > 200) {
        forProgress = 0;
#ifdef COSMO_EVENTS
        traceUserEvents(networkProgressUE);
#endif
        CmiNetworkProgress();
      }
#endif
      computed++;
      r = offset + part->position - particles[j].position;
      rsq = r.lengthSquared();
      twoh = part->soft + particles[j].soft;
      if(rsq != 0) {
        SPLINE(rsq, twoh, a, b);
	cosmoType idt2 = (particles[j].mass + part->mass)*b; // (timescale)^-2
	// of interaction
        particles[j].treeAcceleration += r * (b * part->mass);
        particles[j].potential -= part->mass * a;
	if(idt2 > particles[j].dtGrav)
	  particles[j].dtGrav = idt2;
      }
    }
  }
  return computed;
}
#else
inline int partBucketForce(ExternalGravityParticle *part,
			   Tree::GenericTreeNode *req,
			   GravityParticle **activeParticles,
			   Vector3D<cosmoType> offset, int nActiveParts) {
  Vector3D<SSEcosmoType> r;
  SSEcosmoType rsq;
  SSEcosmoType twoh, a, b;

  for (int i=0; i<nActiveParts; i+=SSE_VECTOR_WIDTH) {
#ifdef CMK_VERSION_BLUEGENE
    if (++forProgress > 200) {
      forProgress = 0;
#ifdef COSMO_EVENTS
      traceUserEvents(networkProgressUE);
#endif
      CmiNetworkProgress();
    }
#endif
    Vector3D<SSEcosmoType>
      packedPos(SSELoad(SSEcosmoType, activeParticles, i, ->position.x),
		SSELoad(SSEcosmoType, activeParticles, i, ->position.y),
		SSELoad(SSEcosmoType, activeParticles, i, ->position.z));
    SSELoad(SSEcosmoType packedSoft, activeParticles, i, ->soft);

    r = -packedPos + offset + part->position;
    rsq = r.lengthSquared();
    twoh = part->soft + packedSoft;
    SSEcosmoType select = rsq > COSMO_CONST(0.0);
    int compare = movemask(select);
    if(compare) {
      SPLINE(rsq, twoh, a, b);
      if ((~compare) & cosmoMask) {
	a = select & a;
	b = select & b;
      }
      SSEcosmoType SSELoad(packedMass, activeParticles, i, ->mass);
      SSEcosmoType SSELoad(packedDtGrav, activeParticles, i, ->dtGrav);
      Vector3D<SSEcosmoType>
	packedAcc(SSELoad(SSEcosmoType,
			  activeParticles, i, ->treeAcceleration.x),
		  SSELoad(SSEcosmoType,
			  activeParticles, i, ->treeAcceleration.y),
		  SSELoad(SSEcosmoType,
			  activeParticles, i, ->treeAcceleration.z));
      SSEcosmoType SSELoad(packedPotential, activeParticles, i, ->potential);
      SSEcosmoType idt2 = (packedMass + part->mass) * b;
      idt2 = max(idt2, packedDtGrav);
      packedAcc += r * (b * part->mass);
      packedPotential -= part->mass * a;
      SSEStore(packedAcc.x, activeParticles, i, ->treeAcceleration.x);
      SSEStore(packedAcc.y, activeParticles, i, ->treeAcceleration.y);
      SSEStore(packedAcc.z, activeParticles, i, ->treeAcceleration.z);
      SSEStore(packedPotential, activeParticles, i, ->potential);
      SSEStore(idt2, activeParticles, i, ->dtGrav);
    }
  }
  return nActiveParts;
}

inline int partBucketForce(ExternalGravityParticle *part,
			   Tree::GenericTreeNode *req,
			   GravityParticle *particles,
			   Vector3D<cosmoType> offset, int activeRung) {
  int nActiveParts = 0;
  GravityParticle dummyPart = particles[0];
  dummyPart.soft = 0.0;

  GravityParticle **activeParticles =
    (GravityParticle**)alloca((req->lastParticle - req->firstParticle
			       + 1 + FORCE_INPUT_LIST_PAD)
			      * sizeof(GravityParticle*));
  for (int j=req->firstParticle; j <= req->lastParticle; ++j) {
    if (particles[j].rung >= activeRung)
      activeParticles[nActiveParts++] = &particles[j];
  }

  activeParticles[nActiveParts] = &dummyPart;
#if SSE_VECTOR_WIDTH == 4
  activeParticles[nActiveParts+1] = &dummyPart;
  activeParticles[nActiveParts+2] = &dummyPart;
#endif

  int ret = partBucketForce(part, req, activeParticles, offset, nActiveParts);
  return ret;
}

#endif

#if !CMK_SSE
/// Native bucket overload of the fork's partBucketForce (renamed so the fork
/// overloads stay in place). Queues the source term on the staged
/// bucket (flushed at TERM_BATCH_MAX).
/// @param part       Source particle.
/// @param offset     Periodic offset applied to the source.
/// @param bs         Staged bucket state.
/// @return           Number of active particles in the bucket.
inline int partBucketForceNative(ExternalGravityParticle *part,
			   Vector3D<cosmoType> offset, BucketSoA &bs) {
  bs.order.push_back(~(int)bs.parts.size());
  bs.parts.push_back(BucketPartTerm{offset, part->position,
				    part->mass, part->soft});
  if ((int)bs.order.size() >= TERM_BATCH_MAX)
    flushTerms(bs);
  return bs.soa.n;
}
#endif


//
// Calculated forces on active particles in a bucket due to the
// multipole of a TreeNode.  Return number of multipoles evaluated.
//
#if  !CMK_SSE
inline
int nodeBucketForce(Tree::GenericTreeNode *node,
		    Tree::GenericTreeNode *req,
		    GravityParticle *particles,
		    Vector3D<cosmoType> offset,
		    int activeRung)

{
  int computed = 0;
  Vector3D<cosmoType> r;
  cosmoType rsq;
  MultipoleMoments &m = node->moments;

  Vector3D<cosmoType> cm(m.cm + offset);

#ifdef HEXADECAPOLE
  if(openSoftening(node, req, offset)) {
    ExternalGravityParticle tmpPart;
    tmpPart.mass = m.totalMass;
    tmpPart.soft = m.soft;
    tmpPart.position = m.cm;
    return partBucketForce(&tmpPart, req, particles, offset, activeRung);
  }
#endif
  for(int j = req->firstParticle; j <= req->lastParticle; ++j) {
    if (particles[j].rung >= activeRung) {
      particles[j].interMass += m.totalMass;
#ifdef CMK_VERSION_BLUEGENE
      if (++forProgress > 200) {
        forProgress = 0;
#ifdef COSMO_EVENTS
        traceUserEvents(networkProgressUE);
#endif
        CmiNetworkProgress();
      }
#endif
      computed++;
      r = Vector3D<cosmoType>(particles[j].position) - cm;
      rsq = r.lengthSquared();
      cosmoType dir = COSMO_CONST(1.0)/sqrt(rsq);
#ifdef HEXADECAPOLE
      cosmoType magai;
      momEvalFmomrcm(&m.mom, m.getRadius(), dir, r.x, r.y, r.z,
		  &particles[j].potential,
		  &particles[j].treeAcceleration.x,
		  &particles[j].treeAcceleration.y,
		  &particles[j].treeAcceleration.z, &magai);
      cosmoType idt2 = (particles[j].mass + m.totalMass)*dir*dir*dir;
#else
      cosmoType twoh, a, b, c, d;
      twoh = CONVERT_TO_COSMO_TYPE(m.soft + particles[j].soft);
      SPLINEQ(dir, rsq, twoh, a, b, c, d);
      cosmoType qirx = CONVERT_TO_COSMO_TYPE m.xx*r.x
	+ CONVERT_TO_COSMO_TYPE m.xy*r.y + CONVERT_TO_COSMO_TYPE m.xz*r.z;
      cosmoType qiry = CONVERT_TO_COSMO_TYPE m.xy*r.x
	+ CONVERT_TO_COSMO_TYPE m.yy*r.y + CONVERT_TO_COSMO_TYPE m.yz*r.z;
      cosmoType qirz = CONVERT_TO_COSMO_TYPE m.xz*r.x
	+ CONVERT_TO_COSMO_TYPE m.yz*r.y + CONVERT_TO_COSMO_TYPE m.zz*r.z;
      cosmoType qir = COSMO_CONST(0.5)*(qirx*r.x + qiry*r.y + qirz*r.z);
      cosmoType tr = COSMO_CONST(0.5)*(CONVERT_TO_COSMO_TYPE m.xx
				       + CONVERT_TO_COSMO_TYPE m.yy
				       + CONVERT_TO_COSMO_TYPE m.zz);
      cosmoType qir3 = b*CONVERT_TO_COSMO_TYPE m.totalMass + d*qir - c*tr;
      particles[j].potential -= CONVERT_TO_COSMO_TYPE m.totalMass * a
	+ c*qir - b*tr;
      particles[j].treeAcceleration.x -= qir3*r.x - c*qirx;
      particles[j].treeAcceleration.y -= qir3*r.y - c*qiry;
      particles[j].treeAcceleration.z -= qir3*r.z - c*qirz;
      cosmoType idt2 = (CONVERT_TO_COSMO_TYPE particles[j].mass
			+ CONVERT_TO_COSMO_TYPE m.totalMass)*b;
#endif
      if(idt2 > particles[j].dtGrav)
        particles[j].dtGrav = idt2;
    }
  }
  return computed;
}

#elif  CMK_SSE
inline
int nodeBucketForce(Tree::GenericTreeNode *node,
		    Tree::GenericTreeNode *req,
		    GravityParticle *particles,
		    Vector3D<cosmoType> offset,
		    int activeRung)

{
  Vector3D<SSEcosmoType> r;
  SSEcosmoType rsq;
  SSEcosmoType twoh;
  SSEcosmoType a,b,c,d;
  MultipoleMoments m = node->moments;
  Vector3D<cosmoType> cm(m.cm + offset);
  int nActiveParts = 0;
  GravityParticle dummyPart = particles[0];
  dummyPart.soft = 0.0;

  GravityParticle **activeParticles =
    (GravityParticle**)alloca((req->lastParticle - req->firstParticle
			       + 1 + FORCE_INPUT_LIST_PAD)
			      * sizeof(GravityParticle*));
  for (int j=req->firstParticle; j <= req->lastParticle; ++j) {
    if (particles[j].rung >= activeRung)
      activeParticles[nActiveParts++] = &particles[j];
  }

  activeParticles[nActiveParts] = &dummyPart;
#if SSE_VECTOR_WIDTH == 4
  activeParticles[nActiveParts+1] = &dummyPart;
  activeParticles[nActiveParts+2] = &dummyPart;
#endif

#ifdef HEXADECAPOLE
  if(openSoftening(node, req, offset)) {
    ExternalGravityParticle tmpPart;
    tmpPart.mass = m.totalMass;
    tmpPart.soft = m.soft;
    tmpPart.position = m.cm;
    int ret = partBucketForce(&tmpPart, req, activeParticles,
			   offset, nActiveParts);
    return ret;
    }
#endif
  for (int i=0; i<nActiveParts; i+=SSE_VECTOR_WIDTH) {
#ifdef CMK_VERSION_BLUEGENE
    if (++forProgress > 200) {
      forProgress = 0;
#ifdef COSMO_EVENTS
      traceUserEvents(networkProgressUE);
#endif
      CmiNetworkProgress();
    }
#endif
    Vector3D<SSEcosmoType>
      packedPos(SSELoad(SSEcosmoType, activeParticles, i, ->position.x),
		SSELoad(SSEcosmoType, activeParticles, i, ->position.y),
		SSELoad(SSEcosmoType, activeParticles, i, ->position.z));
    r = packedPos - cm;
    rsq = r.lengthSquared();
    SSEcosmoType dir = COSMO_CONST(1.0)/sqrt(rsq);
    Vector3D<SSEcosmoType>
      packedAcc(SSELoad(SSEcosmoType,
			activeParticles, i, ->treeAcceleration.x),
		SSELoad(SSEcosmoType,
			activeParticles, i, ->treeAcceleration.y),
		SSELoad(SSEcosmoType,
			activeParticles, i, ->treeAcceleration.z));
    SSEcosmoType SSELoad(packedPotential, activeParticles, i, ->potential);
    SSEcosmoType SSELoad(packedMass, activeParticles, i, ->mass);
    SSEcosmoType SSELoad(packedDtGrav, activeParticles, i, ->dtGrav);
#ifdef HEXADECAPOLE
    SSEcosmoType magai;
    momEvalFmomrcm(&m.mom, m.getRadius(), dir, r.x, r.y, r.z,
		   &packedPotential,
		   &packedAcc.x,
		   &packedAcc.y,
		   &packedAcc.z, &magai);
    SSEcosmoType idt2 = (packedMass + m.totalMass)*dir*dir*dir;
#else
    SSELoad(SSEcosmoType packedSoft, activeParticles, i, ->soft);
    twoh = CONVERT_TO_COSMO_TYPE m.soft + packedSoft;
    SPLINEQ(dir, rsq, twoh, a, b, c, d);
    SSEcosmoType qirx = CONVERT_TO_COSMO_TYPE m.xx*r.x
      + CONVERT_TO_COSMO_TYPE m.xy*r.y + CONVERT_TO_COSMO_TYPE m.xz*r.z;
    SSEcosmoType qiry = CONVERT_TO_COSMO_TYPE m.xy*r.x
      + CONVERT_TO_COSMO_TYPE m.yy*r.y + CONVERT_TO_COSMO_TYPE m.yz*r.z;
    SSEcosmoType qirz = CONVERT_TO_COSMO_TYPE m.xz*r.x
      + CONVERT_TO_COSMO_TYPE m.yz*r.y + CONVERT_TO_COSMO_TYPE m.zz*r.z;
    SSEcosmoType qir = COSMO_CONST(0.5)*(qirx*r.x + qiry*r.y + qirz*r.z);
    SSEcosmoType tr = COSMO_CONST(0.5)*(CONVERT_TO_COSMO_TYPE m.xx
					+ CONVERT_TO_COSMO_TYPE m.yy
					+ CONVERT_TO_COSMO_TYPE m.zz);
    SSEcosmoType qir3 = b*CONVERT_TO_COSMO_TYPE m.totalMass + d*qir - c*tr;
    packedPotential -= CONVERT_TO_COSMO_TYPE m.totalMass *a
      + c * qir - b * tr;
    packedAcc.x -= qir3*r.x - c*qirx;
    packedAcc.y -= qir3*r.y - c*qiry;
    packedAcc.z -= qir3*r.z - c*qirz;
    SSEcosmoType idt2 = (packedMass + CONVERT_TO_COSMO_TYPE m.totalMass)*b;
#endif
    SSEStore(packedAcc.x, activeParticles, i, ->treeAcceleration.x);
    SSEStore(packedAcc.y, activeParticles, i, ->treeAcceleration.y);
    SSEStore(packedAcc.z, activeParticles, i, ->treeAcceleration.z);
    SSEStore(packedPotential, activeParticles, i, ->potential);
    idt2 = max(idt2, packedDtGrav);
    SSEStore(idt2, activeParticles, i, ->dtGrav);
  }
  return nActiveParts;
}
#endif

#if !CMK_SSE
/// Native bucket overload of the fork's nodeBucketForce (renamed so the fork
/// overloads stay in place). Queues the node moments on the staged bucket,
/// or a particle term if softening opens the node.
/// @param node       Source tree node.
/// @param req        Bucket node the walk is evaluating.
/// @param offset     Periodic offset applied to the node.
/// @param bs         Staged bucket state.
/// @return           Number of active particles in the bucket.
inline int nodeBucketForceNative(Tree::GenericTreeNode *node,
			   Tree::GenericTreeNode *req,
			   Vector3D<cosmoType> offset, BucketSoA &bs) {
  const MultipoleMoments &m = node->moments;
  Vector3D<cosmoType> cm(m.cm + offset);
#ifdef HEXADECAPOLE
  if(openSoftening(node, req, offset)) {
    bs.order.push_back(~(int)bs.parts.size());
    bs.parts.push_back(BucketPartTerm{offset, m.cm,
				      m.totalMass, m.soft});
    if ((int)bs.order.size() >= TERM_BATCH_MAX)
      flushTerms(bs);
    return bs.soa.n;
  }
#endif
  bs.order.push_back((int)bs.nodes.size());
  bs.nodes.push_back(BucketNodeTerm{m, cm});
  if ((int)bs.order.size() >= TERM_BATCH_MAX)
    flushTerms(bs);
  return bs.soa.n;
}
#endif

#endif
