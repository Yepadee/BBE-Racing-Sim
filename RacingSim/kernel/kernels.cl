#include "RacingSim/kernel/mwc64x_rng.cl"

#define positions(r, c) positions[c + r*n_c]
#define randoms(r, c) randoms[c + r*n_c]


__kernel void generate_randoms(
    global double* randoms
)
{
    int ii = get_global_id(0);
    mwc64x_state_t rng;
    MWC64X_SeedStreams(&rng, offset, 2);
    randoms[ii] = MWC64X_NextUint(&rng) / (4294967295.0);
}

__kernel void update_positions(
    global double* randoms,
    global double* positions)
{
    int r = get_global_id(0);
    // Update each competetor
    for (uchar c = 0; c < n_c; c++) {
        positions(r, c) = randoms(r, c);
    }
}

