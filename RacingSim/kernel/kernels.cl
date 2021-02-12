#define positions(r, c) positions[c + r*n_c]
#define randoms(r, c) randoms[c + r*n_c]
#ifndef dt10_mwc64x_rng_cl
#define dt10_mwc64x_rng_cl

#include "RacingSim/kernel/skip_mwc.cl"

//! Represents the state of a particular generator
typedef struct{ uint x; uint c; } mwc64x_state_t;

enum{ MWC64X_A = 4294883355U };
enum{ MWC64X_M = 18446383549859758079UL };

void MWC64X_Step(mwc64x_state_t *s)
{
	uint X=s->x, C=s->c;
	
	uint Xn=MWC64X_A*X+C;
	uint carry=(uint)(Xn<C);				// The (Xn<C) will be zero or one for scalar
	uint Cn=mad_hi(MWC64X_A,X,carry);  
	
	s->x=Xn;
	s->c=Cn;
}

void MWC64X_Skip(mwc64x_state_t *s, ulong distance)
{
	uint2 tmp=MWC_SkipImpl_Mod64((uint2)(s->x,s->c), MWC64X_A, MWC64X_M, distance);
	s->x=tmp.x;
	s->c=tmp.y;
}

void MWC64X_SeedStreams(mwc64x_state_t *s, ulong baseOffset, ulong perStreamOffset)
{
	uint2 tmp=MWC_SeedImpl_Mod64(MWC64X_A, MWC64X_M, 1, 0, baseOffset, perStreamOffset);
	s->x=tmp.x;
	s->c=tmp.y;
}

//! Return a 32-bit integer in the range [0..2^32)
uint MWC64X_NextUint(mwc64x_state_t *s)
{
	uint res=s->x ^ s->c;
	MWC64X_Step(s);
	return res;
}

#endif

__kernel void generate_randoms(
    global int* randoms
)
{
    int ii = get_global_id(0);
    mwc64x_state_t rng;
    MWC64X_SeedStreams(&rng, offset, 2);
    randoms[ii] = MWC64X_NextUint(&rng);
}

__kernel void update_positions(
    global int* randoms,
    global float* positions)
{
    int r = get_global_id(0);
    
    // Update each competetor
    for (uchar c = 0; c < n_c; c++) {
        positions(r, c) = randoms(r, c) % 10;
    }
}

