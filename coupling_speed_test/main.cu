#include "network_factory.cuh"
#include "graphics.h"
#include "morphology.h"
#include <set>
#include "electrode_net.cuh"
#include <glm/gtx/norm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "ephys.cuh"

#define OUTPUT_ELECTRODES_FILE

// This is simply a struct that contains all the relevant stuff for coupling. 
// It can be called which will perform coupling and store it. 
// Description:
// - list Adjacency list containing connectivity.
// - coupling Coupling class that can be called to do coupling with a variety of connectivity structs.
// - downstream_inputs Input to downstream neurons after coupling.
// - neuron_outputs Output of neurons.
struct CouplingStruct 
{
    AdjacencyList& list;
    AtomicCoupling& coupling;
    Float* downstream_inputs;
    Float* neuron_outputs;
    
    // Performs coupling and stores results in downstream_inputs. 
    // Returns number of neurons firing.
    int operator() () 
    {
        return coupling(list, downstream_inputs, neuron_outputs);
    }
};

void _RunAndPrintSimulation(int const nincs, int const print_incs, std::ostream& fout, std::ostream& chi_out,
    int const N, Ephys* ephys,
    std::vector<CouplingStruct>& coupling_structs,
    Float* V_ptr,
    float* graphics_voltages_handle = nullptr, int const render_N = -1,
    ElectrodeNet* electrodes = nullptr,
    Morphology* morphology = nullptr,
    std::vector<int> electrode_inds = {},
    SynchronyNet* synchrony_net = nullptr)
{
    auto sim_vars = ephys->SimVars();
    auto sim_vars_cpu = ephys->SimVarsCpu();

    // If graphics handle is non-null, allocate CPU mem to copy voltages to.
    uint8_t* frame_data;
    FILE* replay_out;
    bool pause{ false };
    float* tex;
    int tex_dim = 100;
    if (graphics_voltages_handle)
    {
        frame_data = (uint8_t*)malloc(3 * sizeof(uint8_t) * Graphics::get_width() * Graphics::get_height());
        replay_out = fopen("REPLAY.raw", "wb");
        tex = (float*)malloc(tex_dim * tex_dim * sizeof(float));
    }

//    thrust::host_vector<Float> cpu_arr(N);
//    thrust::fill_n(cpu_arr.begin(), N, 739.09);
//    cudaMemcpy(sim_vars_cpu[SAN::Params::TAUCA], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
//    float n_incs_second = 1000 / ephys->Dt();
    for (int i = 0; i < nincs; ++i)
    {
        if (i % (nincs / 10) == 0)
            std::cout << "\t" << 100 * (i / (float)nincs) << "%" << std::endl;

        Float t = ephys->Dt() * i;
        if (i % print_incs == 0)
        {
            // Update graphics if the graphics handle is non-null
            if (graphics_voltages_handle)
            {
                if(t > 1000) // Skip transient of 4 seconds.
                    synchrony_net->MeasureVariances(V_ptr);
                electrodes->DoElectrodeMeasurements(V_ptr);
                Float const* avg_V_handle = electrodes->ElectrodeOutputHandle();

                for (int k = 0; k < render_N; ++k)
                    graphics_voltages_handle[k] = avg_V_handle[k];
                               
#ifdef OUTPUT_ELECTRODES_FILE
                for(int k = 0; k < render_N; ++k)
                    fout << avg_V_handle[k] << ',';
#endif
                Float V0;
                cudaMemcpy(&V0, V_ptr, sizeof(Float), cudaMemcpyDeviceToHost);
                fout << V0 << std::endl;

#ifndef OUTPUT_ELECTRODES_FILE
//                std::fill_n(tex, tex_dim * tex_dim, 0);
//                for (int k = 0; k < render_N; ++k)
//                {
//                    auto& pos{ morphology->GetPositions()[electrode_inds[k]] };
//                    int tex_i = tex_dim * (pos.x + 1) / 2;
//                    int tex_j = tex_dim * (pos.y + 1) / 2;
//                    tex[tex_i + tex_j * tex_dim] += (avg_V_handle[k] - Graphics::get_color_offset()) / Graphics::get_color_divisor();
//                }
//                // div is average number of neurons per grid cell (assuming uniformality).
//                float div = render_N / (float)(tex_dim * tex_dim);
//                div = *std::max_element(tex, tex + tex_dim * tex_dim);
//                for (int k = 0; k < tex_dim * tex_dim; ++k)
//                    tex[k] /= div;
                pause = Graphics::render(graphics_voltages_handle, nullptr, tex_dim, tex_dim, pause);
#endif
            }
        }

        if (pause)
        {
            i -= 1;
            continue;
        }

        // Perform coupling. The number of neurons firing is returned by each of these calls. 
        int n_firing;
        for(auto& coup_struct: coupling_structs)
            n_firing = coup_struct();

 //       fout << n_firing << std::endl;

        // Update ODE 
        ephys->SimulateEphys();
    }
    std::vector<double> chi_sub = synchrony_net->GetSynchronyMeasures();
    for (auto& chi : chi_sub)
        chi_out << chi << std::endl;

    if (graphics_voltages_handle)
    {
        free(frame_data);
        fclose(replay_out);
        free(tex);
    }
}


bool length_predicate(float const* p1, float const* p2)
{
	return true;
}

static std::vector<glm::vec3> electrode_positions{
	{-0.30186574,  0.68659384, 0},
	{ 0.30186574,  0.68659384, 0},
	{-0.30186574,  0.47586892, 0},
	{ 0.30186574,  0.47586892, 0},
	{-0.44360421,  0.47586892, 0},
	{ 0.44360421,  0.47586892, 0},
	{-0.30186574,  0.22462761, 0},
	{ 0.30186574,  0.22462761, 0},
	{-0.49505155,  0.22462761, 0},
	{ 0.49505155,  0.22462761, 0},
	{-0.35331308,  0.        , 0},
	{ 0.35331308,  0.        , 0},
	{-0.67005517,  0.        , 0},
	{ 0.67005517,  0.        , 0},
	{-0.32211409, -0.19642502, 0},
	{ 0.32211409, -0.19642502, 0},
	{-0.51901899, -0.19642502, 0},
	{ 0.51901899, -0.19642502, 0},
	{-0.7252216 , -0.19642502, 0},
	{ 0.7252216 , -0.19642502, 0},
	{-0.34422199, -0.44766634, 0},
	{ 0.34422199, -0.44766634, 0},
	{-0.55228414, -0.44766634, 0},
	{ 0.55228414, -0.44766634, 0},
	{-0.75828013, -0.44766634, 0},
	{ 0.75828013, -0.44766634, 0},
	{-0.34794107, -0.66176763, 0},
	{ 0.34794107, -0.66176763, 0},
	{-0.60931011, -0.66176763, 0},
	{ 0.60931011, -0.66176763, 0},
	{-0.8283229 , -0.66176763, 0},
	{ 0.8283229 , -0.66176763, 0},
	{-0.34608153, -0.93247269, 0},
	{ 0.34608153, -0.93247269, 0},
	{-0.54112688, -0.93247269, 0},
	{ 0.54112688, -0.93247269, 0}};

static std::vector<glm::vec3> projected_electrode_positions{
    {-0.301866, 0.786594, -0.421104},
    {0.301866, 0.786594, -0.421104},
    {-0.301866, 0.575869, -0.607844},
    {0.301866, 0.575869, -0.533148},
    {-0.443604, 0.575869, -0.421104},
    {0.443604, 0.575869, -0.458452},
    {-0.301866, 0.324628, -0.645193},
    {0.301866, 0.324628, -0.645193},
    {-0.495052, 0.324628, -0.533148},
    {0.495052, 0.324628, -0.4958},
    {-0.353313, 0.1, -0.757237},
    {0.353313, 0.1, -0.757237},
    {-0.670055, 0.1, -0.458452},
    {0.670055, 0.1, -0.533148},
    {-0.322114, -0.096425, -0.869281},
    {0.322114, -0.096425, -0.831933},
    {-0.519019, -0.096425, -0.757237},
    {0.519019, -0.096425, -0.757237},
    {-0.725222, -0.096425, -0.607844},
    {0.725222, -0.096425, -0.570496},
    {-0.344222, -0.347666, -0.90663},
    {0.344222, -0.347666, -0.90663},
    {-0.552284, -0.347666, -0.831933},
    {0.552284, -0.347666, -0.794585},
    {-0.75828, -0.347666, -0.570496},
    {0.75828, -0.347666, -0.607844},
    {-0.347941, -0.561768, -0.943978},
    {0.347941, -0.561768, -0.943978},
    {-0.60931, -0.561768, -0.757237},
    {0.60931, -0.561768, -0.719889},
    {-0.828323, -0.561768, -0.421104},
    {0.828323, -0.561768, -0.383756},
    {-0.346082, -0.832473, -0.831933},
    {0.346082, -0.832473, -0.869281},
    {-0.541127, -0.832473, -0.682541},
    { 0.541127, -0.832473, -0.682541 }};

// N : number of neurons to simulate
// samp_freq : sampling frequency (Hz)
// samp_len : How long you record the activity of the model neuron (sec)
void sanmodel_inhexc_benchmark(int const N = 1, int const samp_freq = 1000, float const samp_len = 10, int render_N = -1)
{
    using namespace SAN;
    // Set up graphics and morphology
	FileMorphology morphology("cortex_connectivity_1m_1k_inh.h5", true);
    morphology.Generate(N);
	morphology.Rotate(glm::radians(90.0f), glm::vec3(1, 0, 0));
	morphology.Rotate(glm::radians(-90.0f), glm::vec3(0, 0, 1));
//    morphology.ProjectNeuronsToPlane(1);
//	morphology.Rotate(glm::radians(90.0f), glm::vec3(0, 0, 1));
//    UnitSquare morphology;
//    morphology.Generate(N);

    UserPointCloud electrodes_cloud(projected_electrode_positions);
//    electrodes_cloud.ProjectNeuronsToPlane(1);

    float* graphics_voltage_handle;
    Graphics::setup(graphics_voltage_handle, morphology.GetPositionsRaw(), N, render_N);
    Graphics::set_color_params(2, -65, 0.8);

    std::ofstream fout("OUT_AN.txt");
    SANEphys ephys(N);
    auto sim_vars_cpu = ephys.SimVarsCpu();

    // Generate coupling
    AtomicCoupling loc_coupling(N), e_coupling(N), i_coupling(N);
    NormalNetwork loc_net(&morphology, 0.03);
    FileNetwork e_net("cortex_connectivity_1m_1k_exc.h5"), i_net("cortex_connectivity_1m_1k_inh.h5");
    loc_net.Generate(N); e_net.Generate(N); i_net.Generate(N);
    AdjacencyList e_list{ e_net.ToAdjacencyList(sim_vars_cpu[Params::ECOUNT]) };
    AdjacencyList loc_list{ loc_net.ToAdjacencyList(sim_vars_cpu[Params::LOCAL_COUNT]) };
    AdjacencyList i_list{ i_net.ToAdjacencyList(sim_vars_cpu[Params::ICOUNT]) };

    std::vector<CouplingStruct> coupling_structs
    {
        {loc_list, loc_coupling, sim_vars_cpu[Params::LOCAL_IN], sim_vars_cpu[Params::LOCAL_OUTPUT]},
        {e_list, e_coupling, sim_vars_cpu[Params::EIN], sim_vars_cpu[Params::EOUTPUT]},
        {i_list, i_coupling, sim_vars_cpu[Params::IIN], sim_vars_cpu[Params::IOUTPUT]}
    };

    // Setup electrodes randomly distributed.
    thrust::host_vector<int> h_regions, h_regions_inds;
    std::vector<int> electrode_inds(render_N);
    for(int i = 0; i < render_N; ++i)
        electrode_inds[i] = (N / render_N) * i;

    int grid_dim = 10;
    int grid_N = grid_dim * grid_dim;
    std::vector<glm::vec3> eeg_pos(grid_N);
    for (int i = 0; i < grid_dim; ++i)
        for (int j = 0; j < grid_dim; ++j)
            eeg_pos[i + j * grid_dim] = glm::vec3(-1.0f, 1.0f, 0.0f) + glm::vec3((i + 0.5f) / float(grid_dim), -(j + 0.5f) / float(grid_dim), 0.0f) * 2.0f;

#ifdef OUTPUT_ELECTRODES_FILE
//    morphology.GenSphericalRegions(h_regions, h_regions_inds, { {-0.5, -0.5, 0.0}, {-0.5, 0.5, 0.0}, {0.5, -0.5, 0.0}, {0.5, 0.5, 0.0} }, N/4);
//    morphology.GenSphericalRegions(h_regions, h_regions_inds, eeg_pos, 1000);
    morphology.GenSphericalRegions(h_regions, h_regions_inds, projected_electrode_positions, 1000);
#else
    morphology.GenSphericalRegions(h_regions, h_regions_inds, electrode_inds, 500);

    // Send the connectivity to graphics.
    Graphics::load_in_connectivity(
        loc_net.GetCpuIndices().data(),
        loc_net.GetCpuOffsetsIntoIndices().data(),
        N, 1000,
        length_predicate);
#endif
    ElectrodeNet electrode_net(h_regions, h_regions_inds);

    thrust::host_vector<int> synchrony_region_inds(32768);
    for(int i = 0; i < synchrony_region_inds.size(); ++i)
        synchrony_region_inds[i] = (N / synchrony_region_inds.size()) * i;
    SynchronyNet synchrony_net(std::move(synchrony_region_inds), { 32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64 });

    // Note, we multiply by 1000 to convert from sec to ms.
    int const nincs = (1000 * samp_len / rk4_dt);
    int const print_incs = (int)((1000.0 / samp_freq) / rk4_dt + 0.5);

    // Run for an intiial transient second.
    for(int i = 0; i < 1000 / ephys.Dt(); ++i)
        ephys.SimulateEphys();

    std::vector<Float> smpls{
0.001,0.00127427,0.00162378,0.00206914,0.00263665,0.00335982,
0.00428133,0.00545559,0.00695193,0.00885867,0.01128838,0.0143845,
0.01832981,0.02335721,0.02976351,0.0379269,0.0483293,0.06158482,
0.078476,0.1};
    int n_smpls = smpls.size();
    thrust::host_vector<Float> cpu_arr(N);
    for (int i = 0; i < n_smpls; ++i)
    {
        for (int j = 0; j < n_smpls; ++j)
        {
            std::ofstream fout("SAN_OUT_" + std::to_string(i) + "_" + std::to_string(j) + ".txt");
            std::ofstream chi_out("SAN_CHI_OUT_" + std::to_string(i) + "_" + std::to_string(j) + ".txt");
        //    ephys.Reset();
            Float g_white = smpls[i];
            thrust::fill_n(cpu_arr.begin(), N, g_white);
            cudaMemcpy(sim_vars_cpu[Params::GGABA_WHITE], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

            Float g_grey = smpls[j];
            thrust::fill_n(cpu_arr.begin(), N, g_grey);
            cudaMemcpy(sim_vars_cpu[Params::GGABA_GREY], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

            std::cout << "White and grey matter parameters: " << g_white << ", " << g_grey << std::endl;
            _RunAndPrintSimulation(nincs, print_incs, fout, chi_out,
                N, &ephys,
                coupling_structs,
                sim_vars_cpu[Params::V],
                graphics_voltage_handle,
                render_N,
                &electrode_net,
                &morphology,
                {},
                &synchrony_net);
            fout.close();
            chi_out.close();
        }
    }

    for (Float*& arr : sim_vars_cpu)
        cudaFree(arr);
    fout.close();
}

// N : number of neurons to simulate
// samp_freq : sampling frequency (Hz)
// samp_len : How long you record the activity of the model neuron (sec)
void hh_benchmark(int const N = 1, int const samp_freq = 1000, float const samp_len = 10, int render_N = -1)
{
    using namespace HH;

    // Set up graphics and morphology
    FileMorphology morphology("cortex_connectivity_1m_1k_inh.h5", true);
    morphology.Generate(N);
    morphology.Rotate(glm::radians(90.0f), glm::vec3(1, 0, 0));
    morphology.Rotate(glm::radians(-90.0f), glm::vec3(0, 0, 1));
    //    morphology.ProjectNeuronsToPlane(1);
    //	morphology.Rotate(glm::radians(90.0f), glm::vec3(0, 0, 1));

    UserPointCloud electrodes_cloud(projected_electrode_positions);
    //    electrodes_cloud.ProjectNeuronsToPlane(1);

    float* graphics_voltage_handle;
#ifdef OUTPUT_ELECTRODES_FILE
    Graphics::setup(graphics_voltage_handle, (float*)projected_electrode_positions.data(), 36, 36);
#else
    Graphics::setup(graphics_voltage_handle, morphology.GetPositionsRaw(), N, render_N);
#endif
    Graphics::set_color_params(15, -60, 0.0);

    HHEphys ephys(N);
    auto sim_vars_cpu = ephys.SimVarsCpu();

    // Generate coupling
    AtomicCoupling loc_coupling(N), e_coupling(N), i_coupling(N);
    NormalNetwork loc_net(&morphology, 0.03);
    FileNetwork e_net("cortex_connectivity_1m_1k_exc.h5"), i_net("cortex_connectivity_1m_1k_inh.h5");
    loc_net.Generate(N); e_net.Generate(N); i_net.Generate(N);
    AdjacencyList e_list{ e_net.ToAdjacencyList(sim_vars_cpu[Params::ECOUNT]) };
    AdjacencyList loc_list{ loc_net.ToAdjacencyList(sim_vars_cpu[Params::LOCAL_COUNT]) };
    AdjacencyList i_list{ i_net.ToAdjacencyList(sim_vars_cpu[Params::ICOUNT]) };

    std::vector<CouplingStruct> coupling_structs
    {
        {loc_list, loc_coupling, sim_vars_cpu[Params::LOCAL_IN], sim_vars_cpu[Params::LOCAL_OUTPUT]},
        {e_list, e_coupling, sim_vars_cpu[Params::EIN], sim_vars_cpu[Params::EOUTPUT]},
        {i_list, i_coupling, sim_vars_cpu[Params::IIN], sim_vars_cpu[Params::IOUTPUT]}
    };

    // Setup electrodes based on mouse 2D data from 
    // https://gin.g-node.org/hiobeen/Mouse_hdEEG_ASSR_Hwang_et_al/src/master/montage.csv.
    thrust::host_vector<int> h_regions, h_regions_inds;
    std::vector<int> electrode_inds(render_N);
#ifdef OUTPUT_ELECTRODES_FILE
    morphology.GenSphericalRegions(h_regions, h_regions_inds, electrodes_cloud.GetPositions(), 1000);
    //   morphology.GenRegionsFromIDs(h_regions, h_regions_inds);
   //    morphology.GenCustomRegions([](glm::vec3 const& pos) {return int(2 * (pos.x + 1)); }, 4, h_regions, h_regions_inds);
#else
    for (int i = 0; i < render_N; ++i)
        electrode_inds[i] = (N / render_N) * i;
    morphology.GenSphericalRegions(h_regions, h_regions_inds, electrode_inds, 1);
    //    morphology.GenRegionsFromIDs(h_regions, h_regions_inds);

        // Send the connectivity to graphics.
    Graphics::load_in_connectivity(
        i_net.GetCpuIndices().data(),
        i_net.GetCpuOffsetsIntoIndices().data(),
        N, 1000,
        length_predicate);
#endif
    ElectrodeNet electrode_net(h_regions, h_regions_inds);

    // Note, we multiply by 1000 to convert from sec to ms.
    int const nincs = (1000 * samp_len / leapfrog_dt);
    int const print_incs = (int)((1000.0 / samp_freq) / leapfrog_dt + 0.5);

//    // Run for an intiial transient second.
    for(int i = 0; i < 1000 / ephys.Dt(); ++i)
        ephys.SimulateEphys();

    thrust::host_vector<int> synchrony_region_inds(8192);
    for(int i = 0; i < synchrony_region_inds.size(); ++i)
        synchrony_region_inds[i] = (N / synchrony_region_inds.size()) * i;
    SynchronyNet synchrony_net(std::move(synchrony_region_inds), { 8192, 4096, 2048, 1024, 512, 256, 128, 64 });

    std::vector<Float> smpls{
        0.01,
        0.01274275, 0.01623777, 0.02069138, 0.02636651, 0.03359818,
 0.04281332, 0.05455595, 0.06951928, 0.08858668, 0.11288379, 0.14384499,
 0.18329807, 0.23357215, 0.29763514, 0.37926902, 0.48329302, 0.61584821,
 0.78475997, 1. };
    int n_smpls = smpls.size();
    thrust::host_vector<Float> cpu_arr(N);
    for (int i = 0; i < n_smpls; ++i)
    {
        for (int j = 0; j < n_smpls; ++j)
        {
            std::ofstream fout("HH_OUT_" + std::to_string(i) + "_" + std::to_string(j) + ".txt");
            std::ofstream chi_out("HH_CHI_OUT_" + std::to_string(i) + "_" + std::to_string(j) + ".txt");
        //    ephys.Reset();
            Float g_white = smpls[i];
            thrust::fill_n(cpu_arr.begin(), N, g_white);
            cudaMemcpy(sim_vars_cpu[Params::GAMPA_WHITE], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

            Float g_grey = smpls[j];
            thrust::fill_n(cpu_arr.begin(), N, g_grey);
            cudaMemcpy(sim_vars_cpu[Params::GAMPA_GREY], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

            std::cout << "White and grey matter parameters: " << g_white << ", " << g_grey << std::endl;
            _RunAndPrintSimulation(nincs, print_incs, fout, chi_out,
                N, &ephys,
                coupling_structs,
                sim_vars_cpu[Params::V],
                graphics_voltage_handle,
                render_N,
                &electrode_net,
                &morphology,
                electrode_inds,
                &synchrony_net);
            fout.close();
            chi_out.close();
        }
    }


    for (Float*& arr : sim_vars_cpu)
        cudaFree(arr);
}

int main(int argc, char** argv)
{
//    if (argc < 3)
//    {
//        std::cerr << " Please provide ampa conductance for white and grey matter, respectively." << std::endl;
//        return -1;
//    }

//    Float g_white = std::atof(argv[1]);
//    Float g_grey = std::atof(argv[2]);

#ifdef OUTPUT_ELECTRODES_FILE
    int N_electrodes = 100; // 239;
#else
    int N_electrodes = 1e5;
#endif
    hh_benchmark(1e6, 2500, 8, N_electrodes);
    Graphics::terminate_graphics();
}
