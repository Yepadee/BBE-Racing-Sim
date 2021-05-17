#include "<kernel_location>/mwc64x_rng.cl"
#define positions(r, c) positions[r + c*n_r]
#define tmp_positions(r, c) tmp_positions[r + c*n_r]

#define rngs(c, n) rngs[n + 2*c]
#define rs(c, n) rs[n + 3*c]
#define ts(c, n) ts[n + 2*c]

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

inline float resp(int c, float pos, global float* rs, global float* ts) {
    float percent_complete = pos / (float) l;
    float r;
    float t1 = ts(c, 0);
    float t2 = 1.0f - ts(c, 1);
    if (percent_complete < t1) r = rs(c, 0);
    else if (percent_complete < t2) r = rs(c, 1);
    else r = rs(c, 2);
    return r;
}

__kernel void update_positions(
    global float* preferences,
    global float* rngs,
    global float* rs,
    global float* ts,
    global float* positions,
    global float* tmp_positions,
    global long* winners,
    ulong offset
    )
{
    int r = get_global_id(0);
    mwc64x_state_t rng;
    MWC64X_SeedStreams(&rng, offset, 2*n_c); // We pick two random numbers for each competetor.
    if (winners[r] == 0) {
        for (int c = 0; c < n_c; c++) {
            // Generate two random floats between 0 and 1
            float rdm1 = (MWC64X_NextUint(&rng) / (4294967295.0));  
            float rdm2 = (MWC64X_NextUint(&rng) / (4294967295.0));

            float pos = positions(r, c);
            float u1 = rngs(c, 0);
            float u2 = rngs(c, 1);

            tmp_positions(r, c) = pos + preferences[c] * u(rdm1, u1, u2) *
                                        g(positions, r, c, rdm2) *
                                        resp(c, pos, rs, ts);

            if (tmp_positions(r, c) >= l) winners[r] = (1 << c) | winners[r];
        }
    }  
}

