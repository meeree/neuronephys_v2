#pragma once
#include <stddef.h>
#include <cstdio>
#include <cstdint>

namespace Graphics
{
    int setup(float*& voltages, float const* positions, size_t N, size_t render_N);

    bool render(float const* voltages, void const* slice, int const slice_width, int const slice_height, bool& pause);
    void terminate_graphics();

    // Sets size of maximum raw video size. If we exceed this, the raw video is converted to an mp4 and overwritten. 
    void set_max_raw_video_size(int const size_in_gb);

    // Writes a frame to raw video file. If file size starts to exceed limit, it is converted to an mp4 and will be overwritten.
    void write_frame(FILE* out, uint8_t* data);

    // replay utilities. Allows user to record a replay of voltages and replay it. 
    void setup_replay(int frame_cache_count); // Allocate frame_cache_count frames to store replay. 
    void record_for_replay(float const* voltages);

    void set_window_title(const char* title);

    // helper functions
    int get_width ();
    int get_height ();

    int get_slice_index();
};