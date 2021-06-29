#include "mouse_cortex.cuh"

MouseCortex::MouseCortex(int N, std::string neuron_model)
{
	// Create and generate morphology.
	morphology.reset(new FileMorphology("cortex_connectivity_1m_1k_inh.h5", true));
    morphology->Generate(N);
	morphology->Rotate(glm::radians(90.0f), glm::vec3(1, 0, 0));
	morphology->Rotate(glm::radians(-90.0f), glm::vec3(0, 0, 1));

    // Create electrophysiology.
    if (neuron_model == "HH")
        ephys.reset(new HHEphys(N));
    else if (neuron_model == "SAN")
        ephys.reset(new SANEphys(N));
    else
        throw std::runtime_error("Error: invalid neuron model string: " + neuron_model);

    // Generate connectivity.
    AtomicCoupling loc_coupling(N), e_coupling(N), i_coupling(N);
    NormalNetwork loc_net(morphology.get(), 0.03);
    FileNetwork e_net("cortex_connectivity_1m_1k_exc.h5"), i_net("cortex_connectivity_1m_1k_inh.h5");
    loc_net.Generate(N); e_net.Generate(N); i_net.Generate(N);

    // This code duplicates the exact same thing because I need different namespaces. This is ugly. 
    auto sim_vars_cpu = ephys->SimVarsCpu();
    if (neuron_model == "HH")
    {
        using namespace HH;
        AdjacencyList e_list{ e_net.ToAdjacencyList(sim_vars_cpu[Params::ECOUNT]) };
        AdjacencyList loc_list{ loc_net.ToAdjacencyList(sim_vars_cpu[Params::LOCAL_COUNT]) };
        AdjacencyList i_list{ i_net.ToAdjacencyList(sim_vars_cpu[Params::ICOUNT]) };
        gray_gaba_connectivity.reset(
            new CouplingStruct{ loc_list, loc_coupling, sim_vars_cpu[Params::LOCAL_IN], sim_vars_cpu[Params::LOCAL_OUTPUT] }
        );
        white_ampa_connectivity.reset(
            new CouplingStruct{ e_list, e_coupling, sim_vars_cpu[Params::EIN], sim_vars_cpu[Params::EOUTPUT] }
        );
        white_gaba_connectivity.reset(
            new CouplingStruct{ i_list, i_coupling, sim_vars_cpu[Params::IIN], sim_vars_cpu[Params::IOUTPUT] }
        );
    }
    else
    {
        using namespace SAN;
        AdjacencyList e_list{ e_net.ToAdjacencyList(sim_vars_cpu[Params::ECOUNT]) };
        AdjacencyList loc_list{ loc_net.ToAdjacencyList(sim_vars_cpu[Params::LOCAL_COUNT]) };
        AdjacencyList i_list{ i_net.ToAdjacencyList(sim_vars_cpu[Params::ICOUNT]) };
        gray_gaba_connectivity.reset(
            new CouplingStruct{ loc_list, loc_coupling, sim_vars_cpu[Params::LOCAL_IN], sim_vars_cpu[Params::LOCAL_OUTPUT] }
        );
        white_ampa_connectivity.reset(
            new CouplingStruct{ e_list, e_coupling, sim_vars_cpu[Params::EIN], sim_vars_cpu[Params::EOUTPUT] }
        );
        white_gaba_connectivity.reset(
            new CouplingStruct{ i_list, i_coupling, sim_vars_cpu[Params::IIN], sim_vars_cpu[Params::IOUTPUT] }
        );
    }
}

int MouseCortex::Simulate()
{
    // Perform coupling. The number of neurons firing is returned by each of these calls. 
    gray_gaba_connectivity->PerformCoupling();
    white_gaba_connectivity->PerformCoupling();
    int n_firing = white_ampa_connectivity->PerformCoupling();

    // Update ODE. 
    ephys->SimulateEphys();

    // Return number of neurons firing.
    return n_firing;
}
