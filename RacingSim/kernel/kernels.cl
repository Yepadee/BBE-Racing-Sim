#include "RacingSim/kernel/mwc64x_rng.cl"

#define positions(r, c) positions[c + r*n_c]
#define randoms(n, r, c) randoms[c + r*n_c + n_c*n_r*n]


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

float g(global float* positions, global float* randoms, int r, int c) {
    float pos = positions(r, c);
    float total_distances = 0;
    int num_close_infront = 0;
    for (int c2 = 0; c2 < n_c; ++ c2) {
        float distance_from_c2 = positions(r,c2) - pos;

        // If c2 is infront of c, and is closer than clean_air_dist
        if (distance_from_c2 > 0 && distance_from_c2 < clean_air_dist) {
            num_close_infront++;
            total_distances += distance_from_c2;
        }
    }
    float avg_distance = total_distances / (float) num_close_infront;
    float blockage_factor = (clean_air_dist - avg_distance) / clean_air_dist; // Between 0 and 1
    float prob = (float) num_close_infront / w;
    float rdm = randoms(1, r, c);
    return 1.0f - ((rdm < prob) ? blockage_factor : 0.0f);
}

__kernel void update_positions(
    global float* rng_mins,
    global float* rng_maxs,
    global float* randoms,
    global float* positions,
    global uchar* winners)
{
    int r = get_global_id(0);
    int c = get_global_id(1);
    // Update each competetor
    uchar winner = winners[r];
    float no_winner_mask = winner == 0;
    positions(r, c) += no_winner_mask * g(positions, randoms, r, c) * u(randoms(0, r, c), rng_mins[c], rng_maxs[c]);
    if (positions(r, c) >= l) winners[r] = (c + 1);
}

