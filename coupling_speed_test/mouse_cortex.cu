#include "mouse_cortex.cuh"

std::string MouseCortex::excitatory_conns_file_name = "isocortex_rho_1000.h5";
std::string MouseCortex::inhibitory_conns_file_name = "cortex_connectivity_1m_1k_inh.h5";
int MouseCortex::uniform_random_Ncouple = 1000;
bool MouseCortex::use_weights_network = false;

MouseCortex::MouseCortex(int N, std::string neuron_model, bool uniform_random_white, bool use_gray)
{
	// Create and generate morphology.
	morphology.reset(new FileMorphology(excitatory_conns_file_name, true));
    morphology->Generate(N);
//	morphology->Rotate(glm::radians(90.0f), glm::vec3(1, 0, 0));
//	morphology->Rotate(glm::radians(-90.0f), glm::vec3(0, 0, 1));
//	morphology->Rotate(glm::radians(90.0f), glm::vec3(0, 1, 0));

    // Create electrophysiology.
    if (neuron_model == "HH")
        ephys.reset(new HHEphys(N));
    else if (neuron_model == "SAN")
        ephys.reset(new SANEphys(N));
    else
        throw std::runtime_error("Error: invalid neuron model string: " + neuron_model);

    // Generate connectivity.
    Float stddev = 100.0; // \mu m 
    KNNNetwork loc_e_net(100, morphology.get());//, loc_i_net(morphology.get(), stddev, 0.1);
    FileNetwork realistic_e_net(excitatory_conns_file_name); //, realistic_i_net(inhibitory_conns_file_name);
    UniformRandomNetwork uniform_e_net(0.9 * uniform_random_Ncouple), uniform_i_net(0.1 * uniform_random_Ncouple);
    if (uniform_random_white)
    {
        uniform_e_net.Generate(N);
        uniform_i_net.Generate(N);
    }
    else
    {
        realistic_e_net.Generate(N);
        //realistic_i_net.Generate(N);
    }

    if (use_gray)
    {
        loc_e_net.Generate(N);
        //loc_i_net.Generate(N);
    }

    auto sim_vars_cpu = ephys->SimVarsCpu();
    if (neuron_model == "HH")
    {
        using namespace HH;
        AdjacencyList e_list, i_list;
        if (uniform_random_white)
        {
            e_list = uniform_e_net.ToAdjacencyList(sim_vars_cpu[Params::ECOUNT], use_weights_network);
            //i_list = uniform_i_net.ToAdjacencyList(sim_vars_cpu[Params::ICOUNT], use_weights_network);
        }
        else
        {
            e_list = realistic_e_net.ToAdjacencyList(sim_vars_cpu[Params::ECOUNT], use_weights_network);
            //i_list = realistic_i_net.ToAdjacencyList(sim_vars_cpu[Params::ICOUNT], use_weights_network);
        }
        white_ampa_connectivity.reset(
            new AtomicCoupling( N, e_list, sim_vars_cpu[Params::EIN], sim_vars_cpu[Params::EOUTPUT] )
        );
        //white_gaba_connectivity.reset(
        //    new AtomicCoupling( N, i_list, sim_vars_cpu[Params::IIN], sim_vars_cpu[Params::IOUTPUT] )
        //);
        
        if (use_gray)
        {
            AdjacencyList loc_e_list{ loc_e_net.ToAdjacencyList(sim_vars_cpu[Params::LOCAL_ECOUNT], use_weights_network) };
            gray_ampa_connectivity.reset(
                new AtomicCoupling( N, loc_e_list, sim_vars_cpu[Params::LOCAL_EIN], sim_vars_cpu[Params::LOCAL_EOUTPUT] )
            );

            //AdjacencyList loc_i_list{ loc_i_net.ToAdjacencyList(sim_vars_cpu[Params::LOCAL_ICOUNT], use_weights_network) };
            //gray_gaba_connectivity.reset(
            //    new AtomicCoupling( N, loc_i_list, sim_vars_cpu[Params::LOCAL_IIN], sim_vars_cpu[Params::LOCAL_IOUTPUT] )
            //);
        }
    }
}

int MouseCortex::Simulate()
{
    // Perform coupling. The number of neurons firing is returned by each of these calls. 
    int n_firing;
    if(gray_ampa_connectivity)
        n_firing = gray_ampa_connectivity->PerformCoupling();
   // if(gray_gaba_connectivity)
   //     n_firing = gray_gaba_connectivity->PerformCoupling();
   // if(white_gaba_connectivity)
   //     n_firing = white_gaba_connectivity->PerformCoupling();
    if(white_ampa_connectivity)
        n_firing = white_ampa_connectivity->PerformCoupling();

    // Update ODE. 
    ephys->SimulateEphys();

    // Return number of neurons firing.
    return n_firing;
}
