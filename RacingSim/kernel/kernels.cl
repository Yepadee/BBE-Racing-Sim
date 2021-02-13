#include "RacingSim/kernel/mwc64x_rng.cl"

#define positions(r, c) positions[c + r*n_c]
#define randoms(r, c) randoms[c + r*n_c]


__kernel void generate_randoms(
    ulong offset, 
    global double* randoms
)
{
    int ii = get_global_id(0);
    mwc64x_state_t rng;
    MWC64X_SeedStreams(&rng, offset, 2);
    randoms[ii] = MWC64X_NextUint(&rng) / (4294967295.0);
}

inline double u(double rdm, double min, double max)
{
    double diff = max - min;
    return min + diff * rdm;
}

__kernel void update_positions(
    global double* rng_mins,
    global double* rng_maxs,
    global double* randoms,
    global double* positions)
{
    int r = get_global_id(0);
    // Update each competetor
    for (int c = 0; c < n_c; c++) {
        double diff = rng_maxs[c] - rng_mins[c];
        positions(r, c) += u(randoms(r, c), rng_mins[c], rng_maxs[c]);
    }
}

