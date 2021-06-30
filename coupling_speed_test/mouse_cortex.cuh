#pragma once

#include "network_factory.cuh"
#include "morphology.h"
#include "electrode_net.cuh"
#include "ephys.cuh"

//! Mouse cortex model including morphology, connectivity, and underlying neuron model.
class MouseCortex
{
public:
    std::unique_ptr<Morphology> morphology;
    std::unique_ptr<Ephys> ephys;

    std::unique_ptr<AtomicCoupling> white_ampa_connectivity; // Long range AMPA connections.
    std::unique_ptr<AtomicCoupling> white_gaba_connectivity; // Long range GABA connections.
    std::unique_ptr<AtomicCoupling> gray_gaba_connectivity; // Short range GABA connections.

    //! Construct the mouse brain model.
    //! \param[in] N Number of neurons to use for simulations. These will be sampled uniformly from the full population of 13 million neurons.
    //! \param[in] neuron_model One of HH or SAN. Default value is HH.
    MouseCortex(int N, std::string neuron_model = "HH");

    //! Simulate the whole cortex! This consists of first handling synaptic coupling then numerically integrating each individual neuron.
    int Simulate();
};
