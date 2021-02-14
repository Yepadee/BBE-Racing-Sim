#include "RacingSim/kernel/mwc64x_rng.cl"

#define positions(r, c) positions[c + r*n_c]
#define randoms(r, c) randoms[c + r*n_c]


__kernel void generate_randoms(
    ulong offset, 
    global float* randoms
)
{
    int ii = get_global_id(0);
    mwc64x_state_t rng;
    MWC64X_SeedStreams(&rng, offset, 2);
    randoms[ii] = (float) (MWC64X_NextUint(&rng) / (4294967295.0));
}

inline double u(float rdm, float min, float max)
{
    float diff = max - min;
    return min + diff * rdm;
}

__kernel void update_positions(
    global float* rng_mins,
    global float* rng_maxs,
    global float* randoms,
    global float* positions,
    global uchar* winners)
{
    int r = get_global_id(0);
    // Update each competetor
    uchar winner = winners[r];
    if (winner <= 0) {
        for (int c = 0; c < n_c; c++) {
            float diff = rng_maxs[c] - rng_mins[c];
            float new_pos = positions(r, c) + u(randoms(r, c), rng_mins[c], rng_maxs[c]);
            positions(r, c) = new_pos;
            if (new_pos >= l) winners[r] = (c + 1);
        }
    }
}

