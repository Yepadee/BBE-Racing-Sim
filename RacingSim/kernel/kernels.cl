#include "RacingSim/kernel/mwc64x_rng.cl"

#define positions(r, c) positions[c + r*n_c]
#define tmp_positions(r, c) tmp_positions[c + r*n_c]
#define randoms(n, r, c) randoms[c + r*n_c + n_c*n_r*n]

inline float u(float rdm, float min, float max)
{
    float diff = max - min;
    return min + diff * rdm;
}

inline float g(global float* positions, int r, int c, float rdm) {
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

    float avg_distance = (num_close_infront > 0) ? (total_distances / (float) num_close_infront) : 0.0f;
    float blockage_factor = (clean_air_dist - avg_distance) / clean_air_dist; // Between 0 and 1
    float prob = (float) num_close_infront / w;

    return 1.0f - ((rdm < prob) ? blockage_factor : 0.0f);
}

__kernel void update_positions(
    global float* preferences,
    global float* rng_mins,
    global float* rng_maxs,
    global float* positions,
    global float* tmp_positions,
    global uchar* winners,
    ulong offset
    )
{
    int r = get_global_id(0);
    int c = get_global_id(1);

    mwc64x_state_t rng;
    MWC64X_SeedStreams(&rng, offset, 2);
    float rdm1 = (float) (MWC64X_NextUint(&rng) / (4294967295.0));
    float rdm2 = (float) (MWC64X_NextUint(&rng) / (4294967295.0));

    // Update each competetor
    uchar winner = winners[r];
    float no_winner_mask = winner == 0;
    tmp_positions(r, c) = positions(r, c) + no_winner_mask * preferences[c] * g(positions, r, c, rdm1) * u(rdm2, rng_mins[c], rng_maxs[c]);
    if (tmp_positions(r, c) >= l) winners[r] = (c + 1);
}

