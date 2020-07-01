// morphology.h : Handles generation and storage of neuron positions. 
#include <array>
#pragma once

enum MorphologyType : unsigned char
{
    eBrain = 0, eBrain2, eSphere, eSCN, eCount, eUndefined
};

namespace Morphology
{
    std::pair<bool,std::array<float,3>> initialize_positions(MorphologyType const type, int const N, const std::string coupling_filename);

    // Accessors
    float const* get_positions();
    float const* get_world_min();
    float const* get_world_max();
};