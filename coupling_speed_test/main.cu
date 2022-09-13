#include "mouse_cortex.cuh"
#include "graphics.h"
#include <set>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "regions_tree.h"
#include "hebbian.cuh"
#include "paracrine.cuh"

using namespace thrust::placeholders;


int NumIncsSecond(float dt=HH::leapfrog_dt)
{
    return int(1000 / dt); 
}

void ElectrodesExample(int N)
{
    MouseCortex mouse_cortex(N);

    int KNN_value = 1000;

    // Setup 5 electrodes.
    std::vector<int> electrode_inds = { 0, 1, 2, 3, N - 1 };
    
    // Generate spherical regions at indices.
    thrust::host_vector<int> h_regions, h_regions_inds;
    mouse_cortex.morphology->GenSphericalRegions(h_regions, h_regions_inds, electrode_inds, KNN_value);

    // Construct ElectrodeNet class.
    ElectrodeNet electrode_net(h_regions, h_regions_inds);

    // Get pointer to voltages. 
    Float const* V_ptr = mouse_cortex.ephys->SimVarsCpu()[HH::Params::V];

    // Output file.
    std::ofstream fout("ELECTRODES_OUT.txt");
    for(int i = 0; i < NumIncsSecond() * 3; ++i)
    {
        mouse_cortex.Simulate();

        // Measure average for each individual electrode.
        electrode_net.DoElectrodeMeasurements(V_ptr, mouse_cortex.white_ampa_connectivity->is_firing);

        // Write electrode measurements to file.
        Float const* handle = electrode_net.ElectrodeOutputHandle();
        for (int j = 0; j < electrode_inds.size(); ++j)
            fout << handle[j] << ',';
        fout << std::endl;
    }
    fout.close();
}

void ParamSweep(int N)
{
    MouseCortex mouse_cortex(N);

    // Get pointer to voltages. 
    Float const* V_ptr = mouse_cortex.ephys->SimVarsCpu()[HH::Params::V];

    // Output file.
    std::ofstream fout("SWEEP_OUT_ENTIRE_NETWORK_RUNMISSING_" + std::to_string(N) + ".txt");
    
    // Vary conductances logarithmically.
    std::vector<Float> g_vals(20);
    Float g_min = -2, g_max = 0;
    for (int i = 0; i < g_vals.size(); ++i)
        g_vals[i] = pow(10, g_min + (g_max - g_min) * i / float(g_vals.size() - 1));
    
    for (int i = 0; i < g_vals.size(); ++i)
    {
        for (int j = 0; j < g_vals.size(); ++j)
        {
            std::cout << i << ',' << j << std::endl;
            // Randomize initial conditions.
            mouse_cortex.ephys->SetVariablesRandomly();

            // Set constants.
//            thrust::fill_n(mouse_cortex.white_ampa_connectivity->list.weights.begin(), N, g_vals[i]);
//            thrust::fill_n(mouse_cortex.gray_gaba_connectivity->list.weights.begin(), N, g_vals[j]);
            thrust::fill_n(thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::G_AMPA_GREY]), N, g_vals[i]);
            thrust::fill_n(thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::G_AMPA_WHITE]), N, g_vals[j]);

            // Write constants to file.
            fout << g_vals[i] << ',' << g_vals[j] << ',';

            // Transient.
            for(int n = 0; n < NumIncsSecond(); ++n)
                mouse_cortex.Simulate();

            // Simulate and record row for file.
            for (int n = 0; n < 5 * NumIncsSecond(); ++n)
            {
                mouse_cortex.Simulate();

                if (n % 10 == 0)
                {
                    // Measure average for each individual electrode.
//                    electrode_net.DoElectrodeMeasurements(V_ptr, mouse_cortex.white_ampa_connectivity->is_firing);
                    // Measure average for ENTIRE cortex.
                    Float avg_entire = ElectrodeNet::MeasureAverageEntireNetwork(V_ptr, N);

                    // Write electrode measurement to file.
//                    Float const* handle = electrode_net.ElectrodeOutputHandle();
                    fout << avg_entire << ',';
                }
            }
            fout << '\n';
        }
    }
    fout.close();
}

void MichiganMathSecretCode(int N, int white_ind, int grey_ind)
{
    MouseCortex mouse_cortex(N);

    // Get pointer to voltages. 
    Float const* V_ptr = mouse_cortex.ephys->SimVarsCpu()[HH::Params::V];

    std::vector<int> electrode_inds(N);

    int electrodeK = 30;
    // Generate electrodes by sampling render_N neurons uniformly from the N and creating electrodes at every such neuron.
    for (int i = 0; i < N; ++i)
        electrode_inds[i] = i;
    thrust::host_vector<int> h_regions, h_regions_inds;
    mouse_cortex.morphology->GenSphericalRegions(h_regions, h_regions_inds, electrode_inds, electrodeK);
    ElectrodeNet electrode_net(h_regions, h_regions_inds);
    Float const* avg_V_handle = electrode_net.ElectrodeOutputHandle();
    
    // Vary conductances logarithmically.
    std::vector<Float> g_vals(20);
    Float g_min = -2, g_max = 0;
    for (int i = 0; i < g_vals.size(); ++i)
        g_vals[i] = pow(10, g_min + (g_max - g_min) * i / float(g_vals.size() - 1));

    Float grey_val = g_vals[grey_ind];
    Float white_val = g_vals[white_ind];
    thrust::fill_n(thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::G_AMPA_GREY]), N, grey_val);
    thrust::fill_n(thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::G_AMPA_WHITE]), N, white_val);

    float* graphics_voltage_handle;
    Graphics::setup(graphics_voltage_handle, mouse_cortex.morphology->GetPositionsRaw(), N, N);
    // Hodgkin-Huxley normalized coloring.
    //Graphics::set_color_params(30, -60, 0.0);

    // Transient.
//    for(int n = 0; n < NumIncsSecond(); ++n)
//        mouse_cortex.Simulate();

    bool pause = false;
    std::ofstream fout("GUANHUA_DATA_noise6_wakecoupling.csv");
    for (int n = 0; n < 5 * NumIncsSecond(); ++n)
    {
        mouse_cortex.Simulate();
        
        for (int i = 0; i < 100; ++i)
            fout << graphics_voltage_handle[i] << ',';
        fout << std::endl;
        cudaMemcpy(graphics_voltage_handle, V_ptr, N, cudaMemcpyDeviceToHost);
        if (n % 400 == 0)
        {
           std::cout << n * HH::leapfrog_dt << std::endl;
           //electrode_net.DoElectrodeMeasurements(V_ptr, mouse_cortex.white_ampa_connectivity->is_firing);
           // for (int j = 0; j < h_regions_inds.size() - 1; ++j)
           //     graphics_voltage_handle[j] = avg_V_handle[j];
           // Graphics::set_window_title(std::to_string(n / float(NumIncsSecond())).c_str());
           // Graphics::render(graphics_voltage_handle, nullptr, 0, 0, pause);
        }
    }
    fout.close();
}

// Render the indivudal regions of the mouse cortex! 
void RenderRegions(int N)
{
    MouseCortex mouse_cortex(N, "HH", false, false);
    float* graphics_voltage_handle;
    Graphics::setup(graphics_voltage_handle, mouse_cortex.morphology->GetPositionsRaw(), N, N);

    // Use super-regions at a specific depth in the regions json tree.
    thrust::host_vector<int> h_regions, h_regions_inds, associated_ids;
    int ID_VTA, ID_STR;
    try
    {
        RegionsTree tree("CUBIC_regionID_lookup_no_comma.json");
        mouse_cortex.morphology->SpecfiyRegionsDepth(6, tree);
        mouse_cortex.morphology->GenRegionsFromIDs(h_regions, h_regions_inds, associated_ids);

        ID_VTA = tree.GetIdOfRegionByName("VISC");
        std::cout << "VTA region ID: " << ID_VTA << std::endl;

        ID_STR = tree.GetIdOfRegionByName("VIS");
        std::cout << "Striatum region ID: " << ID_STR << std::endl;
    }
    catch (std::runtime_error& e)
    {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    // EXPLANATION. 
    // We use a trick in the visualzation here. Because the visualization just takes a float per neuron, we send a float 
    // corresponding to the region index and we set the normalization parameters to be 1/N_regions!
    int N_regions = h_regions_inds.size() - 1;
    Graphics::set_color_params(N_regions, 0.0, 0.1);

    bool pause = false;
    int n_vta = 0, n_str = 0;
    for (int j = 0; j < N; ++j)
    {
        int id = mouse_cortex.morphology->GetRegions()[j];
        std::cout << id << std::endl;
        int unique_region_idx = std::find(associated_ids.begin(), associated_ids.end(), id) - associated_ids.begin();
        if (id == ID_VTA)
        {
            unique_region_idx = N_regions - 1;
            n_vta += 1;
        }
        else if (id == ID_STR)
        {
            unique_region_idx = N_regions * 0.3;
            n_str += 1;
        }
        else
            unique_region_idx = 0;
        graphics_voltage_handle[j] = unique_region_idx;
    }
    std::cout << "VTA count " << n_vta << ", STR count " << n_str << std::endl;
    for(;;)
        Graphics::render(graphics_voltage_handle, nullptr, 0, 0, pause);
}


// A simple test where the visual regions are stimulated. This is a GOOD REFERENCE for visualization and for setting up regions.
void VisualizationTest(int N, int render_N=-1, int electrodeK=10)// electrodeK sets the number of neurons to sample for electrode
{
    if (render_N < 0)
        render_N = N;

    MouseCortex mouse_cortex(N);
    std::vector<int> electrode_inds(render_N);
    int const render_stride = N / render_N;

    // Generate electrodes by sampling render_N neurons uniformly from the N and creating electrodes at every such neuron.
    for (int i = 0; i < render_N; ++i)
        electrode_inds[i] = render_stride * i;
    thrust::host_vector<int> h_regions, h_regions_inds;
    mouse_cortex.morphology->GenSphericalRegions(h_regions, h_regions_inds, electrode_inds, electrodeK);

    ElectrodeNet electrode_net(h_regions, h_regions_inds);
    Float const* avg_V_handle = electrode_net.ElectrodeOutputHandle();
    Float const* V_ptr = mouse_cortex.ephys->SimVarsCpu()[HH::Params::V];

    float* graphics_voltage_handle;
    Graphics::setup(graphics_voltage_handle, mouse_cortex.morphology->GetPositionsRaw(), N, render_N);
    // Hodgkin-Huxley normalized coloring.
    Graphics::set_color_params(15, -60, 0.0);

    // Use super-regions at a specific depth in the regions json tree.
    int visual_region_idx;
    thrust::host_vector<int> h_regions_allen, h_regions_inds_allen, associated_ids;
    try
    {
        RegionsTree tree("CUBIC_regionID_lookup_no_comma.json");
        mouse_cortex.morphology->SpecfiyRegionsDepth(6, tree);
        mouse_cortex.morphology->GenRegionsFromIDs(h_regions_allen, h_regions_inds_allen, associated_ids);

        // Find region corresponding to the visual region in h_regions.
        int ID = tree.GetIdOfRegionByName("VIS");
        std::cout << "Visual region ID: " << ID << std::endl;
        visual_region_idx = std::find(associated_ids.begin(), associated_ids.end(), ID) - associated_ids.begin();

        // Set applied current to zero everywhere but the visual regions.
        thrust::host_vector<Float> appcur(N, 1);//setting applied current
        for (int i = h_regions_inds_allen[visual_region_idx]; i < h_regions_inds_allen[visual_region_idx + 1]; ++i)
            appcur[h_regions_allen[i]] = 1;// sets in some visual region
        thrust::device_vector<Float> d_appcur(appcur);
        thrust::copy_n(d_appcur.begin(), N, thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::APPCUR]));
    }
    catch (std::runtime_error& e)
    {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    thrust::fill_n(thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::G_AMPA_GREY]), N, 0);
    thrust::fill_n(thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::G_AMPA_WHITE]), N, 1);

    thrust::fill_n(thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::G_GABA]), N, 0);

    for (int n = 0; n < 0 * NumIncsSecond(); ++n)
    {
        if (n % 1000 == 0)
            std::cout << n / float(NumIncsSecond()) << std::endl;

        mouse_cortex.Simulate();
    }

//    int const* connectivity_inds = thrust::raw_pointer_cast(mouse_cortex.white_ampa_connectivity->list.indices.data());
//    int const* offset_into_inds = thrust::raw_pointer_cast(mouse_cortex.white_ampa_connectivity->list.offset_in_indices.data());
//    Graphics::load_in_connectivity(connectivity_inds, offset_into_inds, N, N);

    bool pause = false;
    for(int i = 0; i < NumIncsSecond() * 3; ++i)
    {
        mouse_cortex.Simulate();

        electrode_net.DoElectrodeMeasurements(V_ptr, mouse_cortex.white_ampa_connectivity->is_firing);
        for (int j = 0; j < h_regions_inds.size() - 1; ++j)
            graphics_voltage_handle[j] = avg_V_handle[j];
        Graphics::render(graphics_voltage_handle, nullptr, 0, 0, pause);
    }
}


// Play pre-recorded voltage traces or electrode data through visualization.
// This is useful if you want to run graphics-less on a remote machine then visualize locally on a crappy machine.
void PlayBackData(std::string fl_name)
{
    std::cout << "Loading data from file : " << fl_name << std::endl;

    // First read in three individual axes, then put into positions,
    // finally scale and transform into range [-1, 1].
    std::vector<Float> record;
    int T, N; // Number of timesteps and number of neurons.
    using namespace H5;
    try
    {

        H5File file(fl_name, H5F_ACC_RDONLY);

        DataSet dataset = file.openDataSet("data");
        DataSpace dataspace = dataset.getSpace();

        hsize_t dims_out[2];
        dataspace.getSimpleExtentDims(dims_out, NULL);

        T = dims_out[0]; N = dims_out[1];

        record.resize(T * N);
        dataset.read(record.data(), sizeof(Float) == sizeof(float) ? PredType::NATIVE_FLOAT : PredType::NATIVE_DOUBLE);
    }
    catch (...)
    {
        exit(-1);
    }

    std::cout << "Loaded data:\n  Number of timesteps : " << T << "\n  Number of samples : " << N << std::endl;

    MouseCortex mouse_cortex(N, "HH", true, false);

    // Play through graphics.
    float* graphics_voltage_handle;
    Graphics::setup(graphics_voltage_handle, mouse_cortex.morphology->GetPositionsRaw(), N, N);
    // Hodgkin-Huxley normalized coloring.
    Graphics::set_color_params(15, -60, 0.0);
    
    bool pause = false;
    for (int i = 0; i < T;)
    {
        std::copy(record.begin() + N * i, record.begin() + N * (i + 1), graphics_voltage_handle);
        Graphics::render(graphics_voltage_handle, nullptr, 0, 0, pause);

        if (!pause)
            i += 1; // next frame.
    }
}

void RecordNumFiring(int N, std::string fl_suffix="")
{
    std::ofstream fout("N_FIRING_" + fl_suffix + ".txt");
    MouseCortex mouse_cortex(N, "HH", false, false);
    Float const* V_ptr = mouse_cortex.ephys->SimVarsCpu()[HH::Params::V];

    // Transient.
    for(int i = 0; i < NumIncsSecond() * 3; ++i)
        mouse_cortex.Simulate();

    int const n_incs = NumIncsSecond() * 10;
    std::cout << "\r           ";
    for (int i = 0; i < n_incs; ++i)
    {
        if (i % 1000 == 0)
            std::cout << "\r" << i / (float)n_incs * 100 << "%";
        fout << mouse_cortex.Simulate() << '\n';
    }
    std::cout << std::endl;
    fout.close();
}

// Simulate uncoupled independent neurons to get an idea of how they behave individually and get F-I curve.
void UncoupledNeurons(int N, std::string fl_suffix="")
{
    std::ofstream fout("UNCOUPLED_" + fl_suffix + ".txt");
    MouseCortex mouse_cortex(N, "HH", false, false);
    Float const* V_ptr = mouse_cortex.ephys->SimVarsCpu()[HH::Params::V];
    Float const* app_ptr = mouse_cortex.ephys->SimVarsCpu()[HH::appcur];
    // Turn off all coupling.
    thrust::fill_n(thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::G_GABA]), N, 0);
    thrust::fill_n(thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::G_AMPA_WHITE]), N, 0);
    thrust::fill_n(thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::G_AMPA_GREY]), N, 0);

    int const n_incs = NumIncsSecond() * 1;
    thrust::host_vector<Float> V_host(N);
    for (int i = 0; i < n_incs; ++i)
    {
        thrust::copy_n(thrust::device_ptr<Float const>(V_ptr), N, V_host.begin());
        fout << mouse_cortex.Simulate() << ',';  // Number of neurons firing.
        for (Float& val : V_host)
            fout << val << ',';
        fout << std::endl;
    }
    fout.close();
}

/// n_incs_hebbian_sample is the number of simulation increments to do before we output hebbian weights to a file.
void HebbianLearningExample(int N, int n_incs_hebbian_sample = 1000)
{
    MouseCortex::use_weights_network = true;

    // Note we are using uniform random weights here and no local connections.
    MouseCortex mouse_cortex(N, "SAN", true, false);
    auto& conns{ mouse_cortex.white_ampa_connectivity->list };
    auto& is_firing{ mouse_cortex.white_ampa_connectivity->is_firing };
    auto& firing_inds{ mouse_cortex.white_ampa_connectivity->is_firing };
    Hebbian heb(300, 0.1, conns.indices.size());
    thrust::host_vector<Float> h_weights = conns.weights;

    // Transient.
    for(int i = 0; i < NumIncsSecond() * 3; ++i)
        mouse_cortex.Simulate();

    int const n_incs = NumIncsSecond() * 10;
    for (int i = 0; i < n_incs; ++i)
    {
        if (i % 1000 == 0)
            std::cout << "\r" << i / (float)n_incs * 100 << "%";

        // Simulation!
        mouse_cortex.Simulate();

        // Perform Hebbian update.
        heb.UpdateWeights(conns, is_firing, firing_inds);

        if (i % n_incs_hebbian_sample == 0)
        {
            // Write hebbian weights to a file.
            std::ofstream fout("HEBBIAN_WEIGHTS_" + std::to_string(i) + ".txt");
            h_weights = conns.weights;
            for (auto& weight : h_weights)
                fout << weight << ',';
            fout.close();
        }
    }
}

// NOTE THAT: this benchmark defaults to using uniform random connections and no gray matter. 
// These can both be switched so that the gray and white matter are realistic. 
// For FIGURE 2.
void SweepBenchmark(std::vector<int> const& choices_of_N, bool uniform_random_white=true, bool use_gray=false)
{
    std::ofstream fout("HH_VARIED_N_OUT.csv");
    int n_incs = NumIncsSecond();
    for (int N : choices_of_N)
    {
        {
            std::cout << "N " << N << std::endl;
            MouseCortex mouse_cortex(N, "HH", uniform_random_white, use_gray);

            // Start of simulation.
            for (int i = 0; i < n_incs; ++i)
                mouse_cortex.Simulate();

            // Measure elapsed time and write to row of file in format <N> <TIME>
            fout << N << ',' << time << std::endl;
        }
    }
    fout.close();
}

// Generate some examples exhibiting waves in the 3D visualization. 
// N is number of neurons.
// render_N is number of neurons to render. If not specified, render_N = N is default.
// ElectrodeK constrols the number of nearest neighbors for electorde measurements.
// For FIGURE 3.
void WavesExamples(int const N, int render_N = -1, int const electrodeK = 1)
{
    if (render_N < 0)
        render_N = N;

    MouseCortex mouse_cortex(N);
    std::vector<int> electrode_inds(render_N);
    int const render_stride = N / render_N;

    // Generate electrodes by sampling render_N neurons uniformly from the N and creating electrodes at every such neuron.
    for (int i = 0; i < render_N; ++i)
        electrode_inds[i] = render_stride * i;
    thrust::host_vector<int> h_regions, h_regions_inds;
    mouse_cortex.morphology->GenSphericalRegions(h_regions, h_regions_inds, electrode_inds, electrodeK);
    ElectrodeNet electrode_net(h_regions, h_regions_inds);
    Float const* avg_V_handle = electrode_net.ElectrodeOutputHandle();
    Float const* V_ptr = mouse_cortex.ephys->SimVarsCpu()[HH::Params::V];

    float* graphics_voltage_handle;
    Graphics::setup(graphics_voltage_handle, mouse_cortex.morphology->GetPositionsRaw(), N, render_N);
    // Hodgkin-Huxley normalized coloring.
    Graphics::set_color_params(15, -60, 0.0);

    // Use super-regions at a specific depth in the regions json tree.
    int visual_region_idx;
    thrust::host_vector<int> h_regions_allen, h_regions_inds_allen, associated_ids;
    try
    {
        RegionsTree tree("CUBIC_regionID_lookup_no_comma.json");
        mouse_cortex.morphology->SpecfiyRegionsDepth(6, tree);
        mouse_cortex.morphology->GenRegionsFromIDs(h_regions_allen, h_regions_inds_allen, associated_ids);

        // Find region corresponding to the visual region in h_regions.
        int ID = tree.GetIdOfRegionByName("AI");
        std::cout << "Visual region ID: " << ID << std::endl;
        visual_region_idx = std::find(associated_ids.begin(), associated_ids.end(), ID) - associated_ids.begin();

        // Set applied current to zero everywhere but the visual regions.
        thrust::host_vector<Float> appcur(N, 2);
        for (int i = h_regions_inds_allen[visual_region_idx]; i < h_regions_inds_allen[visual_region_idx + 1]; ++i)
            appcur[h_regions_allen[i]] = 7;
        thrust::device_vector<Float> d_appcur(appcur);
        thrust::copy_n(d_appcur.begin(), N, thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::APPCUR]));
    }
    catch (std::runtime_error& e)
    {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    // Simulate and record ERP.
    std::ofstream fout("N_FIRING_APPCUR.csv");
    int switch_incs = NumIncsSecond();
    for (int window = 0; window < 3; ++window)
    {
        int n_incs_transient = NumIncsSecond() * 1;
        int n_incs_record = NumIncsSecond() * 2;
        float stim = window == 0 ? 0 : (window == 1 ? 5 : 10);
        float base = window == 0 ? 0 : 3;
        bool pause = false;
        for (int i = 0; i < n_incs_transient + n_incs_record; ++i)
        {
            thrust::host_vector<Float> appcur(N, base);
            Float appcur_write = base;
            if (i % switch_incs == 0)

            if((i < n_incs_transient || i % (2 * switch_incs) < switch_incs) && i % (switch_incs / 500) < 2)
            {
//                for (int j = h_regions_inds_allen[visual_region_idx]; j < h_regions_inds_allen[visual_region_idx + 1]; ++j)
//                    appcur[h_regions_allen[j]] = stim;
                for (int j = 0; j < N; ++j)
                    if (mouse_cortex.morphology->GetPositions()[j].y > 0.0)
                        appcur[j] = stim;
//                thrust::fill_n(appcur.begin(), N, stim);
                appcur_write = stim;
            }
            thrust::device_vector<Float> d_appcur(appcur);
            thrust::copy_n(d_appcur.begin(), N, thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::APPCUR]));

            fout << mouse_cortex.Simulate() << ',' << appcur_write << '\n';

            if (i > n_incs_transient)
            {
                electrode_net.DoElectrodeMeasurements(V_ptr, mouse_cortex.white_ampa_connectivity->is_firing);
                for (int j = 0; j < h_regions_inds.size() - 1; ++j)
                    graphics_voltage_handle[j] = avg_V_handle[j];
                Graphics::render(graphics_voltage_handle, nullptr, 0, 0, pause);
            }
        }
    }
    fout.close();
}

// Simulates ERP for three cases: realistic connectivity, realistic short range and uniform long range, and uniform long range. 
// For FIGURE 4.
void SimulatedERP(int const N=1e6)
{
    MouseCortex::excitatory_conns_file_name = "cortex_connectivity_1m_1k_exc.h5";
    MouseCortex::inhibitory_conns_file_name = "cortex_connectivity_1m_1k_inh.h5";

    // Figure out what the indices of the electrodes should be.
    thrust::host_vector<int> h_regions, h_regions_inds, associated_ids;

    // Regions to sample EEG for. This differs from the above because we want an equal number of neurons per region for EEG.
    thrust::host_vector<int> eeg_sample_regions, eeg_sample_regions_inds; 
    {
        MouseCortex mouse_cortex(N, "HH", false, false);
        // Use super-regions at a specific depth in the regions json tree.
        try
        {
            RegionsTree tree("CUBIC_regionID_lookup_no_comma.json");
            mouse_cortex.morphology->SpecfiyRegionsDepth(6, tree);
            mouse_cortex.morphology->GenRegionsFromIDs(h_regions, h_regions_inds, associated_ids);
            
            // Sample the same number of neurons from each subregion to measure EEG for.
            int N_sample = INT_MAX;
            for (int i = 0; i < h_regions_inds.size() - 1; ++i)
            {
                if(h_regions_inds[i+1] - h_regions_inds[i] < N_sample)
                    N_sample = h_regions_inds[i+1] - h_regions_inds[i];
                if(h_regions_inds[i+1] - h_regions_inds[i] < N_sample)
                    N_sample = h_regions_inds[i+1] - h_regions_inds[i];
            }
            std::cout << "Sampling the first " << N_sample << " neurons from each region" << std::endl;

            eeg_sample_regions.resize((h_regions_inds.size() - 1) * N_sample);
            eeg_sample_regions_inds.resize(h_regions_inds.size(), 0);
            for (int i = 0; i < h_regions_inds.size() - 1; ++i)
            {
                eeg_sample_regions_inds[i + 1] = eeg_sample_regions_inds[i] + N_sample;
                for (int j = 0; j < N_sample; ++j)
                    eeg_sample_regions[eeg_sample_regions_inds[i] + j] = h_regions[h_regions_inds[i] + j];
            }
        }
        catch (std::runtime_error& e)
        {
            std::cerr << e.what() << std::endl;
            exit(1);
        }
    }
    int n_incs_transient = NumIncsSecond() * 3;
    int n_incs_record = NumIncsSecond() * 1;

    ElectrodeNet electrode_net(eeg_sample_regions, eeg_sample_regions_inds);
    Float const* avg_V_handle = electrode_net.ElectrodeOutputHandle();

    // Simulate and record the three cases: gray + white, gray + uniform, uniform only.
    bool use_uniform_white[3]{ false, true, true };
    bool use_gray[3]{ true, true, false };
    std::string fout_names[3]{ "ERP_OUT_GRAY_WHITE.csv", "ERP_OUT_GRAY_UNIFORM.csv", "ERP_OUT_UNIFORM.csv" };
    for(int case_idx = 0; case_idx < 3; ++case_idx)
    {
        std::cout << " USE UNIFORM : " << use_uniform_white[case_idx] << ", USE GRAY : " << use_gray[case_idx] << std::endl;
        MouseCortex mouse_cortex(N, "HH", use_uniform_white[case_idx], use_gray[case_idx]);
        std::ofstream fout(fout_names[case_idx]);
        Float const* V_ptr = mouse_cortex.ephys->SimVarsCpu()[HH::Params::V];

        // Set applied current to 6 everywhere.
        thrust::host_vector<Float> appcur(N, 6);
        thrust::device_vector<Float> d_appcur(appcur);
        thrust::copy_n(d_appcur.begin(), N, thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::APPCUR]));

        // Transient.
        for (int i = 0; i < n_incs_transient; ++i)
            mouse_cortex.Simulate();

        // Simulate and record ERP.
        for (int i = 0; i < n_incs_record; ++i)
        {
            electrode_net.DoElectrodeMeasurements(V_ptr, mouse_cortex.white_ampa_connectivity->is_firing);
            mouse_cortex.Simulate();

            for (int j = 0; j < eeg_sample_regions_inds.size() - 1; ++j)
                fout << avg_V_handle[j] << ',';
            fout << '\n';
        }
        fout.close();
    }
}

// Given connectivity and regions, writes a matrix R to a file 
// where R_{ij} is the number of connections from region i to region j.
void EvaluatePhysicalConnectivity(std::string fout_name, 
    thrust::host_vector<int> const& region_per_neuron, thrust::host_vector<int> const& associated_ids,
    AdjacencyList const& conns)
{
    std::cout << "Writing physical connectivity to file: " << fout_name << "..." << std::endl;
    std::ofstream fout(fout_name);
    int N_regions = associated_ids.size();
    std::vector<std::vector<int>> phys_conns(N_regions, std::vector<int>(N_regions, 0));

    // Copy adjacency list to cpu.
    thrust::host_vector<int> indices = conns.indices;
    thrust::host_vector<int> offset_in_indices = conns.offset_in_indices;

    for (int n = 0; n < region_per_neuron.size(); ++n)
    {
        // Get associated region for this neuron.
        int reg = std::find(associated_ids.begin(), associated_ids.end(), region_per_neuron[n]) - associated_ids.begin();

        // Iterate over downstream neurons and incremement phys_conns respectively for each region.
        for (int d = offset_in_indices[n]; d < offset_in_indices[n + 1]; ++d)
        {
            int downstream = indices[d];
            int downstream_reg = std::find(associated_ids.begin(), associated_ids.end(), region_per_neuron[downstream]) - associated_ids.begin();
            phys_conns[reg][downstream_reg] += 1;
        }
    }
    
    // Write physical connectivity to file.
    for (int i = 0; i < N_regions; ++i)
    {
        for (int j = 0; j < N_regions; ++j)
            fout << phys_conns[i][j] << ',';
        fout << '\n';
    }
    fout.close();
}

// Record fMRI in regions for use in functional connectivity.
// A specified region is stimulated.
// For FIGURE 5-6.
void FunctionalData(int const N=1e6, std::string stim_region="VIS")
{
    MouseCortex::excitatory_conns_file_name = "cortex_connectivity_1m_1k_exc.h5";
    MouseCortex::inhibitory_conns_file_name = "cortex_connectivity_1m_1k_inh.h5";

    // Figure out what the indices of the electrodes should be.
    thrust::host_vector<int> h_regions, h_regions_inds, associated_ids;

    // Regions to sample EEG for. This differs from the above because we want an equal number of neurons per region for EEG.
    thrust::host_vector<int> eeg_sample_regions, eeg_sample_regions_inds; 
    int stim_region_idx = -1;
    std::ofstream region_sizes_fout("REGION_SIZES.csv");
    std::ofstream region_names("REGION_NAMES.txt");
    {
        MouseCortex mouse_cortex(N, "HH", false, true);

        // Use super-regions at a specific depth in the regions json tree.
        try
        {
            RegionsTree tree("CUBIC_regionID_lookup_no_comma.json");
            mouse_cortex.morphology->SpecfiyRegionsDepth(6, tree);
            mouse_cortex.morphology->GenRegionsFromIDs(h_regions, h_regions_inds, associated_ids);

            // Print region names to file.
            for (int id : associated_ids)
                region_names << tree.GetNameOfRegionById(id) << '\n';

            // Find region corresponding to the visual region in h_regions.
            int ID = tree.GetIdOfRegionByName(stim_region);
            std::cout << stim_region << " region ID: " << ID << std::endl;
            stim_region_idx = std::find(associated_ids.begin(), associated_ids.end(), ID) - associated_ids.begin();
            
            // Sample the same number of neurons from each subregion to measure EEG for.
            int N_sample = INT_MAX;
            for (int i = 0; i < h_regions_inds.size() - 1; ++i)
            {
                region_sizes_fout << h_regions_inds[i + 1] - h_regions_inds[i] << '\n';
                if(h_regions_inds[i+1] - h_regions_inds[i] < N_sample)
                    N_sample = h_regions_inds[i+1] - h_regions_inds[i];
                if(h_regions_inds[i+1] - h_regions_inds[i] < N_sample)
                    N_sample = h_regions_inds[i+1] - h_regions_inds[i];
            }
            std::cout << "Sampling the first " << N_sample << " neurons from each region" << std::endl;

            eeg_sample_regions.resize((h_regions_inds.size() - 1) * N_sample);
            eeg_sample_regions_inds.resize(h_regions_inds.size(), 0);
            for (int i = 0; i < h_regions_inds.size() - 1; ++i)
            {
                eeg_sample_regions_inds[i + 1] = eeg_sample_regions_inds[i] + N_sample;
                for (int j = 0; j < N_sample; ++j)
                    eeg_sample_regions[eeg_sample_regions_inds[i] + j] = h_regions[h_regions_inds[i] + j];
            }
            EvaluatePhysicalConnectivity("PHYSICAL_CONNECTIVITY.csv", mouse_cortex.morphology->GetRegions(), associated_ids, mouse_cortex.white_ampa_connectivity->list);
        }
        catch (std::runtime_error& e)
        {
            std::cerr << e.what() << std::endl;
            exit(1);
        }
    }
    region_sizes_fout.close();
    region_names.close();
    int n_incs_transient = NumIncsSecond() * 3;
    int n_incs_record = NumIncsSecond() * 300;

    ElectrodeNet electrode_net(eeg_sample_regions, eeg_sample_regions_inds);
    Float const* avg_V_handle = electrode_net.ElectrodeOutputHandle();
    int const* firing_handle = electrode_net.FiringPerRegionHandle();

    // Run simulation
    MouseCortex mouse_cortex(N, "HH", false, true);
    std::ofstream firing_fout("FMRI_" + stim_region + ".csv");
    Float const* V_ptr = mouse_cortex.ephys->SimVarsCpu()[HH::Params::V];

    // Set applied current to zero everywhere but the stim region.
    thrust::host_vector<Float> appcur(N, 6);
    for (int i = h_regions_inds[stim_region_idx]; i < h_regions_inds[stim_region_idx + 1]; ++i)
        appcur[h_regions[i]] = 10;
    thrust::device_vector<Float> d_appcur(appcur);
    thrust::copy_n(d_appcur.begin(), N, thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::APPCUR]));

    // Transient.
    for (int i = 0; i < n_incs_transient; ++i)
        mouse_cortex.Simulate();

    // Simulate and record ERP.
    for (int i = 0; i < n_incs_record; ++i)
    {
        electrode_net.DoElectrodeMeasurements(V_ptr, mouse_cortex.white_ampa_connectivity->is_firing);
        mouse_cortex.Simulate();

        for (int j = 0; j < eeg_sample_regions_inds.size() - 1; ++j)
            firing_fout << firing_handle[j] << ',';
        firing_fout << '\n';
    }
    firing_fout.close();
}


//Below are Paracrine Main Test Functions
thrust::device_vector<Float> paracrine_test() {
    //Start with concentration at neuron location
    //spread to grid 
    //step the diffusion equation
    //interpolate back to neuron
    //return neuron_concentrations

    //Seed random number
    srand((unsigned)time(0));
    
    //Get all variables set so we can create paracrine object
    int grid_size = 32; // total size of grid = (grid_size x grid_size x grid_size) 
    int nnz = 10000; //number of neurons
    Float dx = (Float)2/(Float)grid_size; //dx from Ningyuan's code
    Float dt = 0.04; //seconds     (so 40 milliseconds)
    Float diffusion = 2e-5;  //diffusion constant taken from Ningyuan's code
    Float decay = 3e-4; //difusion constant taken from Ningyuan's code

    //Create empty thrust vectors
    thrust::host_vector <Float> neuron_locations_x(nnz); //x coordinate locations of neurons
    thrust::host_vector <Float> neuron_locations_y(nnz); //y coordinate locations of neurons
    thrust::host_vector <Float> neuron_locations_z(nnz); //z coordinate locations of neurons
    thrust::device_vector <Float> neuron_concentrations(nnz); //concentrations at neuron locations
    thrust::device_vector <Float> grid(grid_size * grid_size * grid_size); //concentrations at grid point locations
    for (int i = 0; i < grid_size * grid_size * grid_size; i++) {
        grid[i] = ((Float)rand() / (Float)RAND_MAX) * 2;
    }


    //Neuron Initialization
    //Note: The location of gridpoints and neurons is on a scale from 0 to 31
    //Neuron at location (0.5, 0.5, 0.5) is at PHYSICAL location (0.5 * dx, 0.5 * dx, 0.5 * dx) 
    //where dx is the distance between gridpoints
    //For this code, it simplifies calculations and programming to have locations in the code be from 0 to 31)
    for (int i = 0; i < nnz; i++) {
        //Create random neuron locations
        neuron_locations_x[i] = ((Float)rand() / (Float)RAND_MAX) * 31;
        neuron_locations_y[i] = ((Float)rand() / (Float)RAND_MAX) * 31;
        neuron_locations_z[i] = ((Float)rand() / (Float)RAND_MAX) * 31;

        //Create some random concentrations at neuron locations
        neuron_concentrations[i] = ((Float)rand() / (Float)RAND_MAX); //values between 0 and 1
    }

    //Create paracrine object
    paracrine p_obj(grid_size, nnz, dx, dt,
     diffusion, decay, neuron_locations_x,
     neuron_locations_y, neuron_locations_z, grid, neuron_concentrations);

    //Initialize object
    p_obj.initialize();

    //Open Document
    std::ofstream output;
    output.open("paracrinetest.txt");

    //How many seconds would you like the simulation to run for?
    Float time = 100;
    //Get number of diffusion steps
    int total_steps = ceil(time / dt);

    for (int timestep = 0; timestep < total_steps; timestep++) {
        if (timestep % (total_steps / 100) == 0)
            std::cout << 100 * (Float(timestep) / Float(total_steps)) << "%" << std::endl;

        //Spread the neurotransmitter concentration created at neuron locations to the grid
        //For now, there is a constant amount of neurotransmitter created every timestep
        //See spread function for more details
        grid = p_obj.spread(0.001, grid, neuron_concentrations);

        //Step the diffusion equation forward
        //This includes Crank-Nicolson and conjugate gradient
        grid = p_obj.update_density(grid);

        //Send data to output to analyze
        output << timestep * dt << "," << grid[grid_size * grid_size * 16 + grid_size * 16 + 16] << std::endl;

        //Interpolate the concentrations at the gridpoints back to the neuron location
        //So that we can determine the new amount of spreading in the next iteration
        neuron_concentrations = p_obj.interpolate(grid);

    }
    output.close();
}


//Create test to spread from neuron to grid, then interpolate from grid to neuron
//we should return back to the same values
void spread_interp_test() {
    std::cout << "Testing spreading, then interpolation" << std::endl;

  //Get all variables set so we can create paracrine object
  int grid_size = 32;
  int nnz = 1024;
  //Float dx = 1.0;
  Float dx = (Float)2/(Float)grid_size; //dx from Ningyuan's code
  Float dt = 1.0;
  Float diffusion = 1.0;
  Float decay = 1.0;
  thrust::host_vector <Float> neuron_locations_x(nnz);
  thrust::host_vector <Float> neuron_locations_y(nnz);
  thrust::host_vector <Float> neuron_locations_z(nnz);
  thrust::device_vector <Float> neuron_IC(nnz);
  thrust::device_vector <Float> grid_IC(grid_size * grid_size * grid_size);

  //initialize thrust vector
  //Set neuron locations very simple first (assume 0.5 in x,y,z directions)
  //Locations: between 0 and grid_size (e.g. 0 and <32), x=0.5 would be halfway between the first and second gridpoint
  //the actual location of (0.5,0.5,0.5) is (0.5*dx, 0.5*dx, 0.5*dx) but for ease in computation, we leave as is and 
  //account for the real distances in the initialization step (paracrine.cu initialization function)
  for (int i = 0; i < nnz; i++) { //get every neuron away from beginning cube
    neuron_locations_x[i] = 20;
    neuron_locations_y[i] = 20;
    neuron_locations_z[i] = 20;
  }

  //set 3 test neuron locations to beginning cube
  //First Neuron
  neuron_locations_x[0] = 0.1;
  neuron_locations_y[0] = 0.1;
  neuron_locations_z[0] = 0.1;

  //Second Neuron
  neuron_locations_x[1] = 0.5;
  neuron_locations_y[1] = 0.5;
  neuron_locations_z[1] = 0.5;

  //Third Neuron
  neuron_locations_x[2] = 0.5;
  neuron_locations_y[2] = 0.5;
  neuron_locations_z[2] = 0.5;

  //declare dummy neuron_concentrations vector (NOT USED RIGHT NOW)
  // this neuron_concentrations vector is added so that when we want to have 
  // the concentration at the neuron locations dictate how much neurotransmitter is released, we can easily do it
  // for now, we are just releasing generation_constant amount of neurotransmitter (a constant)
  thrust::device_vector <Float> neuron_concentrations(nnz);

  //Set initial conditions on grid and make a dummy vector for neuron_concentrations
  for (int i = 0; i < grid_size * grid_size * grid_size; i++) {
    grid_IC[i] = 0;
  }
  for (int i = 0; i < nnz; i++) {
      //neuron_concentrations[i] = 0.0;
//      neuron_concentrations[i] = ((Float)rand() / (Float)RAND_MAX) * 2;
      neuron_concentrations[i] =1.0;

  }
//  neuron_concentrations[1] = 100;

  grid_IC[grid_size * grid_size * 0 + grid_size * 0 + 0] = 0; //grid[0,0,0]
  grid_IC[grid_size * grid_size * 1 + grid_size * 0 + 0] = 0; //grid[1,0,0]
  grid_IC[grid_size * grid_size * 0 + grid_size * 1 + 0] = 0; //grid[0,1,0]
  grid_IC[grid_size * grid_size * 1 + grid_size * 1 + 0] = 0; //grid[1,1,0]
  grid_IC[grid_size * grid_size * 0 + grid_size * 0 + 1] = 0; //grid[0,0,1]
  grid_IC[grid_size * grid_size * 1 + grid_size * 0 + 1] = 0; //grid[1,0,1]
  grid_IC[grid_size * grid_size * 0 + grid_size * 1 + 1] = 0; //grid[0,1,1]
  grid_IC[grid_size * grid_size * 1 + grid_size * 1 + 1] = 0; //grid[1,1,1]

  //initialize object
  paracrine test(grid_size, nnz, dx, dt,
    diffusion, decay, neuron_locations_x,
    neuron_locations_y, neuron_locations_z, grid_IC, neuron_IC);
  test.initialize();

  //Generation constant is how much neurotransmitter is being spread from each neuron
  //It's incorporated as a large vector to have different neurotransmitter creation values
  //per neuron, see paracrine.cu spread function (particularly where P is created)
  Float generation_constant = 0;

  thrust::device_vector<Float> testvec(grid_size * grid_size * grid_size,0);

  std::cout << "Before Spreading:" << std::endl;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        std::cout << "grid[" << i << "," << j << "," << k << "]=" 
		<< grid_IC[grid_size * grid_size * i + grid_size * j + k] << std::endl;
      }
    }
  }
  for (int n = 0; n < 4; n++) {
      //The spreading function takes the amount generated from the neuron location and adds it to the already
      //existing concentration at the gridpoint location
      //so this test starts with some initial neuron concentration
      //spreads it to the grid
      testvec = test.spread(generation_constant, grid_IC, neuron_concentrations);

      thrust::transform(testvec.begin(), testvec.end(), grid_IC.begin(), grid_IC.begin(), thrust::minus <Float> ());
//      for (int f = 0; f < grid_size * grid_size * grid_size; f++)
 //         grid_IC[f] = testvec[f];
      //std::cout << "After Spreading " << generation_constant << " from each of the three neurons" << std::endl;
      std::cout << "After Spreading the neuron concentration from each of the three neurons" << std::endl;
      for (int i = 0; i < 3; i++) {
          std::cout << "Neuron[" << i << "] at location [" << neuron_locations_x[i] << "," << neuron_locations_y[i] <<
              "," << neuron_locations_z[i] << "] ="<<neuron_concentrations[i]<< std::endl;
      }

      Float sum = 0;
      for (int i = 0; i < 2; i++) {
          for (int j = 0; j < 2; j++) {
              for (int k = 0; k < 2; k++) {
                  std::cout << "grid[" << i << "," << j << "," << k << "]="
                      << grid_IC[grid_size * grid_size * i + grid_size * j + k] << std::endl;
                  sum = sum + grid_IC[grid_size * grid_size * i + grid_size * j + k];
              }
          }
      }
      std::cout << "SUM:" << sum << std::endl;

      neuron_concentrations = test.interpolate(grid_IC);

      sum = 0;
      for (int i = 0; i < 3; i++) {
          std::cout << "Neuron " << i << " concentration at location [" << neuron_locations_x[i] <<
              "," << neuron_locations_y[i] << "," << neuron_locations_z[i] << "]=" << neuron_concentrations[i] << std::endl;
          sum = sum + neuron_concentrations[i];
      }
  }
}

//Diffusion Visualization
void update_density_test() {
  std::cout << "Testing Update Density" << std::endl;
  //Get all variables set so we can create paracrine object
  int grid_size = 32;
  int nnz = 1024;
  Float dx = (Float)2/(Float)grid_size; //dx from Ningyuan's code
  Float dt = 0.04;
  Float diffusion = 2e-4;
  //Float diffusion = 2e-5; //default value
  Float decay = 0;
//  Float decay = 3e-4; //default value
//  Float decay = 10;
  thrust::host_vector <Float> neuron_locations_x(nnz);
  thrust::host_vector <Float> neuron_locations_y(nnz);
  thrust::host_vector <Float> neuron_locations_z(nnz);
  thrust::host_vector <Float> neuron_IC(nnz);
  thrust::host_vector <Float> grid(grid_size * grid_size * grid_size);

  //initialize thrust vector
  //Set neuron locations very simple first (assume 0.5 in x,y,z directions)
  for (int i = 0; i < nnz; i++) {
    neuron_locations_x[i] = 0.5;
    neuron_locations_y[i] = 0.5;
    neuron_locations_z[i] = 0.5;
  }

  //Set initial conditions on grid
  for (int i = 0; i < grid_size * grid_size * grid_size; i++) {
      //grid[i] = 1e-4;
      grid[i] = (rand() % 4) / 4.0;
  }
      grid[grid_size * grid_size * 16 + grid_size * 16 + 16] = 1.0;

  //initialize object
  paracrine Diffusiontest(grid_size, nnz, dx, dt,
    diffusion, decay, neuron_locations_x,
    neuron_locations_y, neuron_locations_z, grid, neuron_IC);
  Diffusiontest.initialize();

  std::cout << "Initialization finished" << std::endl;
  //Open Document
  std::ofstream output;
  output.open("tester.txt");

   // How many seconds would you like the simulation to run for?
    Float time = 100;

    // Get number of diffusion steps.
    int total_steps = ceil(time / dt);

    int N = 0; // No points for now.
    float* graphics_voltage_handle;
    Graphics::setup(graphics_voltage_handle, nullptr, N, N);

    bool pause = false;
    thrust::host_vector<float> slice(grid_size * grid_size);
    for (int timestep = 0; timestep < total_steps; timestep++) 
    {
        grid = Diffusiontest.update_density(grid);
        output << timestep * dt << "," << grid[grid_size * grid_size * 16 + grid_size * 16 + 16] << std::endl;
        if (timestep % (total_steps / 100) == 0)
            std::cout << 100 * (Float(timestep) / Float(total_steps)) << "%" << std::endl;

        // Slice density at index specified by Graphics. 
        int sidx = Graphics::get_slice_index();
        std::cout << sidx << std::endl;
        thrust::copy(grid.begin() + grid_size * grid_size * sidx, grid.begin() + grid_size * grid_size * (sidx + 1), slice.begin());
        Graphics::render(graphics_voltage_handle, slice.data(), grid_size, grid_size, pause);
    }
    output.close();
}


//Conjugate Gradient Test
void CGtest() {
  //REMEMBER TO ENABLE THE FIRST GUESS KERNEL IN THE CG FUNCTION WHEN WE ARE DOING LARGE COMPUTATION
  //Get all variables set so we can create paracrine object
  int grid_size = 4;
  int nnz = 1;
  Float dx = (Float)2/(Float)grid_size; //dx from Ningyuan's code
  Float dt = 1.0;
  Float diffusion = 1.0;
  Float decay = 1.0;
  thrust::host_vector <Float> neuron_locations_x(nnz);
  thrust::host_vector <Float> neuron_locations_y(nnz);
  thrust::host_vector <Float> neuron_locations_z(nnz);
  thrust::host_vector <Float> neuron_IC(nnz);
  thrust::host_vector <Float> grid_IC(grid_size * grid_size * grid_size);

  //initialize thrust vector
  //Set neuron locations very simple first (assume 0.5 in x,y,z directions)
  for (int i = 0; i < nnz; i++) {
    neuron_locations_x[i] = 0.5;
    neuron_locations_y[i] = 0.5;
    neuron_locations_z[i] = 0.5;
  }

  //Set initial conditions on grid
  for (int i = 0; i < grid_size * grid_size * grid_size; i++)
    grid_IC[i] = 1.0;

  //initialize object
  paracrine CGtest(grid_size, nnz, dx, dt,
    diffusion, decay, neuron_locations_x,
    neuron_locations_y, neuron_locations_z, grid_IC, neuron_IC);

  //CG solves Ax=b
  //thrust::device_vector <Float> A(27, 1.0);
  thrust::device_vector <Float> A(27,0.0);
  //If main diagonal
  A[3 * 3 * 0 + 3 * 0 + 0] = 3.0; //A[0,0,0]=1
  A[3 * 3 * 1 + 3 * 1 + 1] = 3.0; //A[1,1,1]=2
  A[3 * 3 * 2 + 3 * 2 + 2] = 3.0; //A[2,2,2]=3
  //If a vertex, set equal to 1
  A[3 * 3 * 0 + 3 * 2 + 2] = 1.0; //A[0,2,2]=1
  A[3 * 3 * 2 + 3 * 0 + 2] = 1.0; //A[2,0,2]=1
  A[3 * 3 * 2 + 3 * 2 + 0] = 1.0; //A[2,2,0]=1
  A[3 * 3 * 2 + 3 * 0 + 0] = 1.0; //A[2,0,0]=1
  A[3 * 3 * 0 + 3 * 0 + 2] = 1.0; //A[0,0,2]=1
  A[3 * 3 * 0 + 3 * 2 + 0] = 1.0; //A[0,2,0]=1
  
  //thrust::device_vector <Float> b(grid_size * grid_size * grid_size, 10);
  thrust::device_vector <Float> b(grid_size * grid_size * grid_size);
  for (int i = 0; i < grid_size * grid_size * grid_size; i++) {
      b[i] = i;
  }

  //Now the answer to Ax=b should be x such that convolve(x,A)=b;
  thrust::device_vector <Float> x(grid_size * grid_size * grid_size);

  thrust::device_vector <Float> laplacian_grid = CGtest.convolve(grid_IC, A);

  x = CGtest.CG(A, b);

  //Now we need to test convolve(x,A)=b;
  thrust::device_vector <Float> b_test(grid_size * grid_size * grid_size);
  b_test = CGtest.convolve(x, A);
  for (int i = 0; i < grid_size * grid_size * grid_size; i++) {
    std::cout << "real b=" << b[i] << "   b_test=" << b_test[i] << std::endl;
  }

}

//Mask Multiplication Test
void laplaciantest() {

  //Get all variables set so we can create paracrine object
  int grid_size = 32;
  int nnz = 1;
  Float dx = (Float)2/(Float)grid_size; //dx from Ningyuan's code
  Float dt = 1.0;
  Float diffusion = 1.0;
  Float decay = 1.0;
  thrust::host_vector <Float> neuron_locations_x(nnz);
  thrust::host_vector <Float> neuron_locations_y(nnz);
  thrust::host_vector <Float> neuron_locations_z(nnz);
  thrust::host_vector <Float> neuron_IC(nnz);
  thrust::device_vector <Float> grid_IC(grid_size * grid_size * grid_size);

  //initialize thrust vector
  //Set neuron locations very simple first (assume 0.5 in x,y,z directions)
  for (int i = 0; i < nnz; i++) {
    neuron_locations_x[i] = 0.5;
    neuron_locations_y[i] = 0.5;
    neuron_locations_z[i] = 0.5;
  }

  //Set initial conditions on grid
//  for (int i = 0; i < grid_size * grid_size * grid_size; i++)
 //   grid_IC[i] = rand();

  thrust::device_vector<Float> real_values(grid_size*grid_size*grid_size);

  for (int i = 0; i < grid_size; i++) {
      Float xx = dx * i;
      for (int j = 0; j < grid_size; j++) {
          Float yy = dx * j;
          for (int k = 0; k < grid_size; k++) {
            Float zz = dx * k;
            grid_IC[grid_size * grid_size * i + grid_size * j + k] = xx * xx * xx + yy * yy + zz * zz + xx * yy * zz;
            real_values[grid_size * grid_size * i + grid_size * j + k] = 6*xx+4;

          }
      }
  }



  //initialize object
  paracrine MaskMulttest(grid_size, nnz, dx, dt,
    diffusion, decay, neuron_locations_x,
    neuron_locations_y, neuron_locations_z, grid_IC, neuron_IC);
  MaskMulttest.initialize();

  thrust::device_vector<Float> result(grid_size*grid_size*grid_size);
  thrust::device_vector<Float> laplace_stencil_divided(27);
  laplace_stencil_divided = MaskMulttest.stencil;
  Float dum = 1 / (dx * dx * 30.0);
  thrust::transform(laplace_stencil_divided.begin(), laplace_stencil_divided.end(), laplace_stencil_divided.begin(), dum * _1);
  result = MaskMulttest.convolve(grid_IC, laplace_stencil_divided);

  //For this test, given A (3x3x3) and x (grid_size x grid_size x grid_size), we should be able to compute Ax=b
  //This is done by sliding A across x and calculating the sum of the element wise multiplications
//  thrust::device_vector <Float> A(27, 1.0);
//  thrust::device_vector <Float> x(grid_size * grid_size * grid_size, 1.0);
//  thrust::device_vector <Float> b(grid_size * grid_size * grid_size);
//  b = MaskMulttest.convolve(x, A);

  //The following comments were meant for the test case of a grid of 1's
  //A is a 3x3x3 mask of 1's
  //x is grid_size x grid_size x grid_size of 1's
  //We should see when the mask isn't touching the boundary of x (i.e. when i,j, or k are not 0 or grid_size)
  //we expect the value of b to be 27.
  //If there is 1 side of the mask touching the boundary (i.e. when i,j, or k are equal to 0 or grid_size)
  //we expect the value of b to be 18.
  //If there are 2 sides of the mask touching the boundary (i.e. when i and j, or j and k, or k and i, are 0 or grid_size)
  //we expect the value of b to be 12.
  //If there are 3 sides of the mask touching the boundary
  //we expect the value of b to be 8.
  //-------------------------------------------------------------------

  std::cout << "Testing laplacian of f(x,y,z)=x^2+y^2+z^2+xyz, the result should be laplacian_f = 6 at values not touching the boundary"
      << std::endl;
  
  for (int i = 1; i < 5; i++) {
    for (int j = 1; j < 5; j++) {
      for (int k = 1; k < 5; k++) {
        std::cout << "expected_values[" << i << "," << j << "," << k << "]=" << real_values[grid_size * grid_size * i + grid_size * j + k] << "     ";
        std::cout << "values[" << i << "," << j << "," << k << "]=" << result[grid_size * grid_size * i + grid_size * j + k] << std::endl;
      }
    }
  }
}

//Interpolation Test
void Interpolationtest() {
  //Get all variables set so we can create paracrine object
  int grid_size = 32;
  int nnz = 1024;
  Float dx = (Float)2/(Float)grid_size; //dx from Ningyuan's code
  Float dt = 1.0;
  Float diffusion = 1.0;
  Float decay = 1.0;
  thrust::host_vector <Float> neuron_locations_x(nnz);
  thrust::host_vector <Float> neuron_locations_y(nnz);
  thrust::host_vector <Float> neuron_locations_z(nnz);
  thrust::device_vector <Float> neuron_IC(nnz);
  thrust::device_vector <Float> grid_IC(grid_size * grid_size * grid_size);

  //initialize thrust vector
  //Set neuron locations very simple first (assume 0.5 in x,y,z directions)
  for (int i = 0; i < nnz; i++) {
    neuron_locations_x[i] = 30;
    neuron_locations_y[i] = 30;
    neuron_locations_z[i] = 30;
  }

  //set neuron locations
  //First Neuron
  neuron_locations_x[0] = 0.1;
  neuron_locations_y[0] = 0.1;
  neuron_locations_z[0] = 0.1;

  //Second Neuron
  neuron_locations_x[1] = 0.5;
  neuron_locations_y[1] = 0.5;
  neuron_locations_z[1] = 0.5;

  //Third Neuron
  neuron_locations_x[2] = 0.5;
  neuron_locations_y[2] = 0.5;
  neuron_locations_z[2] = 0.5;

  //Set initial conditions on grid
  for (int i = 0; i < grid_size * grid_size * grid_size; i++)
    grid_IC[i] = 10;

  grid_IC[grid_size * grid_size * 0 + grid_size * 0 + 0] = 0.0; //grid[0,0,0]
  grid_IC[grid_size * grid_size * 1 + grid_size * 0 + 0] = 0.0 ; //grid[1,0,0]
  grid_IC[grid_size * grid_size * 0 + grid_size * 1 + 0] = 0.0; //grid[0,1,0]
  grid_IC[grid_size * grid_size * 1 + grid_size * 1 + 0] = 0.0; //grid[1,1,0]
  grid_IC[grid_size * grid_size * 0 + grid_size * 0 + 1] = 0.0; //grid[0,0,1]
  grid_IC[grid_size * grid_size * 1 + grid_size * 0 + 1] = 0.2; //grid[1,0,1]
  grid_IC[grid_size * grid_size * 0 + grid_size * 1 + 1] = 0.0; //grid[0,1,1]
  grid_IC[grid_size * grid_size * 1 + grid_size * 1 + 1] = 1.0; //grid[1,1,1]

  //initialize object
  paracrine Interpolationtest(grid_size, nnz, dx, dt,
    diffusion, decay, neuron_locations_x,
    neuron_locations_y, neuron_locations_z, grid_IC, neuron_IC);
  Interpolationtest.initialize();

  //Create Concentration at Neuron Locations vector
  thrust::device_vector <Float> neuron_concentrations(nnz);
  neuron_concentrations = Interpolationtest.interpolate( grid_IC);

  for (int i = 0; i < 3; i++) {
    std::cout << "Neuron " << i << " concentration at location [" << neuron_locations_x[i] <<
      "," << neuron_locations_y[i] << "," << neuron_locations_z[i] << "]=" << neuron_concentrations[i] << std::endl;
  }

}

//Spreading Test
void Spreadtest() {
  //Get all variables set so we can create paracrine object
  int grid_size = 32;
  int nnz = 1024;
  Float dx = (Float)2/(Float)grid_size; //dx from Ningyuan's code
  Float dt = 1.0;
  Float diffusion = 1.0;
  Float decay = 1.0;
  thrust::host_vector <Float> neuron_locations_x(nnz);
  thrust::host_vector <Float> neuron_locations_y(nnz);
  thrust::host_vector <Float> neuron_locations_z(nnz);
  thrust::host_vector <Float> neuron_IC(nnz);
  thrust::host_vector <Float> grid_IC(grid_size * grid_size * grid_size);

  //initialize thrust vector
  //Set neuron locations very simple first (assume 0.5 in x,y,z directions)
  //Locations: between 0 and grid_size (e.g. 0 and <32), x=0.5 would be halfway between the first and second gridpoint
  //the actual location of (0.5,0.5,0.5) is (0.5*dx, 0.5*dx, 0.5*dx) but for ease in computation, we leave as is and 
  //account for the real distances in the initialization step (paracrine.cu initialization function)
  for (int i = 0; i < nnz; i++) { //get every neuron away from beginning cube
    neuron_locations_x[i] = 20;
    neuron_locations_y[i] = 20;
    neuron_locations_z[i] = 20;
  }

  //set 3 test neuron locations to beginning cube
  //First Neuron
  neuron_locations_x[0] = 0.25;
  neuron_locations_y[0] = 0.25;
  neuron_locations_z[0] = 0.25;

  //Second Neuron
  neuron_locations_x[1] = 0.25;
  neuron_locations_y[1] = 0.25;
  neuron_locations_z[1] = 0.25;

  //Third Neuron
  neuron_locations_x[2] = 0.25;
  neuron_locations_y[2] = 0.25;
  neuron_locations_z[2] = 0.25;

  //declare dummy neuron_concentrations vector (NOT USED RIGHT NOW)
  // this neuron_concentrations vector is added so that when we want to have 
  // the concentration at the neuron locations dictate how much neurotransmitter is released, we can easily do it
  // for now, we are just releasing generation_constant amount of neurotransmitter (a constant)
  thrust::device_vector <Float> neuron_concentrations(nnz);

  //Set initial conditions on grid and make a dummy vector for neuron_concentrations
  for (int i = 0; i < grid_size * grid_size * grid_size; i++) {
    grid_IC[i] = 0;
  }
  for (int i = 0; i < nnz; i++)
    neuron_concentrations[i] = 10;

//  neuron_concentrations[1] = 100;

  grid_IC[grid_size * grid_size * 0 + grid_size * 0 + 0] = 0; //grid[0,0,0]
  grid_IC[grid_size * grid_size * 1 + grid_size * 0 + 0] = 0; //grid[1,0,0]
  grid_IC[grid_size * grid_size * 0 + grid_size * 1 + 0] = 0; //grid[0,1,0]
  grid_IC[grid_size * grid_size * 1 + grid_size * 1 + 0] = 0; //grid[1,1,0]
  grid_IC[grid_size * grid_size * 0 + grid_size * 0 + 1] = 0; //grid[0,0,1]
  grid_IC[grid_size * grid_size * 1 + grid_size * 0 + 1] = 0; //grid[1,0,1]
  grid_IC[grid_size * grid_size * 0 + grid_size * 1 + 1] = 0; //grid[0,1,1]
  grid_IC[grid_size * grid_size * 1 + grid_size * 1 + 1] = 0; //grid[1,1,1]

  //initialize object
  paracrine Spreadingtest(grid_size, nnz, dx, dt,
    diffusion, decay, neuron_locations_x,
    neuron_locations_y, neuron_locations_z, grid_IC, neuron_IC);
  Spreadingtest.initialize();

  //Generation constant is how much neurotransmitter is being spread from each neuron
  //It's incorporated as a large vector to have different neurotransmitter creation values
  //per neuron, see paracrine.cu spread function (particularly where P is created)
  Float generation_constant = 2;

  std::cout << "Before Spreading:" << std::endl;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        std::cout << "grid[" << i << "," << j << "," << k << "]=" 
		<< grid_IC[grid_size * grid_size * i + grid_size * j + k] << std::endl;
      }
    }
  }
  grid_IC = Spreadingtest.spread(generation_constant, grid_IC, neuron_concentrations);
  std::cout << "After Spreading " << generation_constant << " from each of the three neurons" << std::endl;
  for (int i = 0; i < 3; i++) {
    std::cout << "Neuron[" << i << "] at location [" << neuron_locations_x[i] << "," << neuron_locations_y[i] <<
      "," << neuron_locations_z[i] << "]" << std::endl;
  }

  Float sum = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        std::cout << "grid[" << i << "," << j << "," << k << "]="
		<< grid_IC[grid_size * grid_size * i + grid_size * j + k] << std::endl;
        sum = sum + grid_IC[grid_size * grid_size * i + grid_size * j + k];
      }
    }
  }
  std::cout << "SUM:" << sum << std::endl;
}



void update_paracrine(MouseCortex const& mouse_cortex, thrust::device_vector<Float>& grid, paracrine& par)
{
    Float const* T_ptr = mouse_cortex.ephys->SimVarsCpu()[HH::Params::EOUTPUT];
    thrust::device_vector<Float> concentrations(par.nnz);
    thrust::copy_n(thrust::device_ptr<Float const>(T_ptr), par.nnz, concentrations.begin());
    grid = par.spread(1, grid, concentrations); // THE FIRST PARAMETER IS THE GENERATION CONSTANT
    grid = par.update_density(grid);
    par.interpolate(grid); // NOT USED
}

// Visualization of 3d paracrine density simulation.
void DiffusionTest (int N) {
    //Get all variables set so we can create paracrine object
    int grid_size = 32;
    int nnz = N;
    Float dx = 0.0625; //dx from Ningyuan's code
    Float dt = 0.02;
    Float diffusion = 0.0001;
    Float decay = 0;
    std::ofstream fout("test.txt");
    thrust::host_vector<Float> neuron_locations_x(nnz);
    thrust::host_vector<Float> neuron_locations_y(nnz);
    thrust::host_vector<Float> neuron_locations_z(nnz);
    thrust::host_vector<Float> neuron_IC(nnz);
    thrust::host_vector<Float> grid(grid_size * grid_size * grid_size);
    thrust::device_vector<Float> grid_gpu(grid_size * grid_size * grid_size);

    MouseCortex mouse_cortex(N);
    auto const& positions = mouse_cortex.morphology->GetPositions();

    // Copy neuron locations to paracrine. Need to do some scaling and translating.
    for (int i = 0; i < N; i++) {
        glm::vec3 pos = positions[i];

        // Transform from [-1, 1] to [0, grid_size]
        // This is done by the sequence of transforms: [-1, 1] -> [0, 2] -> [0, 1] -> [0, grid_size].
        // Note we also need to transpose coordinates here because of different (x,y,z) convention in morphology vs paracrine.
        pos = (pos + 1.0f) * 0.5f * float(grid_size);
        neuron_locations_x[i] = pos.x;
        neuron_locations_y[i] = pos.y;
        neuron_locations_z[i] = pos.z;
    }

    // Set initial conditions on grid.
    for (int i = 0; i < grid_size * grid_size * grid_size; i++) {
        grid[i] = 0.001;
    }
    grid[grid_size * grid_size * 16 + grid_size * 16 + 16] = 1;

    // Initialize paracrine object.
    paracrine par(grid_size, nnz, dx, dt,
        diffusion, decay, neuron_locations_x,
        neuron_locations_y, neuron_locations_z, grid, neuron_IC);
    par.initialize();

    std::cout << "Paracrine initialization finished" << std::endl;

    // How many seconds would you like the simulation to run for?
    Float time = 100;

    // Get number of diffusion steps.
    int total_steps = ceil(time / dt);

    float* graphics_voltage_handle;
    Graphics::setup(graphics_voltage_handle, mouse_cortex.morphology->GetPositionsRaw(), N, N);
    Graphics::set_color_params(30, -60, 0.0);

    // Set applied current to 6 everywhere.
    thrust::host_vector<Float> appcur(N, 6);
    thrust::device_vector<Float> d_appcur(appcur);
    thrust::copy_n(d_appcur.begin(), N, thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::APPCUR]));

    bool pause = false;
    thrust::host_vector<float> slice(grid_size * grid_size);
    std::cout <<"Initial:" << grid[grid_size * grid_size * 16 + grid_size * 16 + 16] << std::endl;
    for (int timestep = 0; timestep < total_steps;)
    {
        if (!pause)
        {
            // Simulate mouse cortex and update paracrine.
            mouse_cortex.Simulate();
            update_paracrine(mouse_cortex, grid_gpu, par);
            grid = grid_gpu; // Copy device to host.
            timestep += 1;
        }
        //if (timestep % (total_steps / 100) == 0)
           // std::cout << 100 * (Float(timestep) / Float(total_steps)) << "%" << std::endl;
        std::cout <<"Center Concentration at timestep " << std::to_string(timestep) <<": "<< grid[grid_size * grid_size * 16 + grid_size * 16 + 16] << std::endl;
        for (int k = 0; k < grid_size; ++k)
            fout << grid[grid_size * grid_size * 16 + grid_size * 16 + k] << ',';
        fout << std::endl;
        // Slice density at index specified by Graphics.
        int sidx = Graphics::get_slice_index();
        std::cout << sidx << std::endl;
        auto it = grid.begin();
        thrust::copy(grid.begin() + grid_size * grid_size * sidx, grid.begin() + grid_size * grid_size * (sidx + 1), slice.begin());
        Graphics::render(graphics_voltage_handle, slice.data(), grid_size, grid_size, pause);
    }
    fout.close();
}
// Record mouse_cortex simulation to an HDF file. 
// sample_rate is number of timesteps to simulate before sampling. So a value of 1 is every timestep.
// output is in h5 format.
void SimulateAndRecord(MouseCortex& mouse_cortex, ElectrodeNet& eeg, size_t total_timesteps, size_t sample_rate, std::string h5_out_name)
{
    size_t eeg_N = eeg.N_regions;
    auto eeg_out = eeg.ElectrodeOutput();
    Float const* V_ptr = mouse_cortex.ephys->SimVarsCpu()[HH::Params::V];

    hsize_t T = total_timesteps / sample_rate;
    std::vector<Float> out(eeg_N * T, 0.0);
    for (int i = 0; i < total_timesteps; ++i)
    {
        if (i % 1000 == 0)
            std::cout << " Sim time in seconds : " << i / float(NumIncsSecond()) << std::endl;

        if (i % sample_rate == 0)
        {
            eeg.DoElectrodeMeasurements(V_ptr, mouse_cortex.white_ampa_connectivity->is_firing);
            thrust::copy_n(eeg_out.begin(), eeg_N, out.begin() + eeg_N * (i / sample_rate));
        }
        mouse_cortex.Simulate();
    }
    Utility::WriteDataToHDF(h5_out_name, out.data(), { T, (hsize_t)eeg_N });
}

// Setup uniform electrodes that measure K nearest neighbors.
// Electrodes are chosen uniformly from neurons indices. 
ElectrodeNet SetupElectrodes(MouseCortex const& mouse_cortex, int knn, int eeg_N = -1)
{
    if (eeg_N < 0)
        eeg_N = mouse_cortex.ephys->N;

    std::vector<int> electrode_inds(eeg_N);
    int const stride = mouse_cortex.ephys->N / eeg_N;

    for (int i = 0; i < eeg_N; ++i)
        electrode_inds[i] = stride * i;
    thrust::host_vector<int> h_regions, h_regions_inds;
    mouse_cortex.morphology->GenSphericalRegions(h_regions, h_regions_inds, electrode_inds, knn);
    return ElectrodeNet(h_regions, h_regions_inds);
}

// Case where we simulate neurons with condutances for synapses selected by user.
void SimulateSynapticStrengths(int N, float g_ampa_grey, float g_ampa_white, float g_gaba)
{
    MouseCortex mouse_cortex(N);

    int eeg_N = N / 10;
    if (eeg_N < 0)
        eeg_N = mouse_cortex.ephys->N;

    std::vector<int> electrode_inds(eeg_N);
    int const stride = mouse_cortex.ephys->N / eeg_N;

    for (int i = 0; i < eeg_N; ++i)
        electrode_inds[i] = stride * i;
    thrust::host_vector<int> h_regions, h_regions_inds;
    mouse_cortex.morphology->GenSphericalRegions(h_regions, h_regions_inds, electrode_inds, 10);
    ElectrodeNet electrode_net(h_regions, h_regions_inds);

    thrust::fill_n(thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::G_AMPA_GREY]), N, g_ampa_grey);
    thrust::fill_n(thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::G_AMPA_WHITE]), N, g_ampa_white);
    thrust::fill_n(thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::G_GABA]), N, g_gaba);

    auto fname = "SimSynStrength_GAMPAGREY=" + std::to_string(g_ampa_grey) + "_GAMPAWHITE=" + std::to_string(g_ampa_white) + "_GGABA=" + std::to_string(g_gaba);
    SimulateAndRecord(mouse_cortex, electrode_net, 5 * NumIncsSecond(), 10, fname);
}

int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        std::cout << """No case specified in command line arguments.\n Usage: ./executable <case> [parameters...]"""
		     """where <case> is one of the functions in mains.cu"""
		     """and [parameters...] are the list of parameters to send to this specific function."""
		  << std::endl;
        return -1;
    }
    
    std::string case_str = argv[1];
    if(case_str == "SimulateSynapticStrength")
    {
        if(argc != 6)
	{
	    std::cout << """Incorrect number of parameters for SimulateSynapticStrength. Four parameters required are:"""
		         """<# of neurons> <g_ampa_grey> <g_ampa_white> <g_gaba>"""
                      << std::endl;
	    return -1;
	}
        SimulateSynapticStrengths(atoi(argv[2]), atof(argv[3]), atof(argv[4]), atof(argv[5]));
    }

    // RUN NEURONEPHYS
    VisualizationTest(1e5);
    
    // RUN PARACRINE
    //DiffusionTest(1e5);
}

