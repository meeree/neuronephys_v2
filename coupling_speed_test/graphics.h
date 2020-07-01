#pragma once
#include <stddef.h>
#include <cstdio>
#include <cstdint>

namespace Graphics
{
    int setup(float*& voltages, float const* positions, size_t N, size_t render_N);

    bool render(float const* voltages, void const* slice, int const slice_width, int const slice_height, bool& pause);
    void terminate_graphics();
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