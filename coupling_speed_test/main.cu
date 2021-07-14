#include "mouse_cortex.cuh"
#include "graphics.h"
#include <set>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <chrono>
#include "regions_tree.h"

int NumIncsSecond(float dt=HH::leapfrog_dt)
{
    return int(1000 / dt); 
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
            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < n_incs; ++i)
                mouse_cortex.Simulate();
            auto stop = std::chrono::steady_clock::now();

            // Measure elapsed time and write to row of file in format <N> <TIME>
            float time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.0f;
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

    int n_incs_transient = NumIncsSecond() * 3;
    int n_incs_record = NumIncsSecond() * 3;

    float* graphics_voltage_handle;
    Graphics::setup(graphics_voltage_handle, mouse_cortex.morphology->GetPositionsRaw(), N, render_N);
    // Hodgkin-Huxley normalized coloring.
    Graphics::set_color_params(15, -60, 0.0);

    // Transient.
    for (int i = 0; i < n_incs_transient; ++i)
        mouse_cortex.Simulate();

    // Simulate and record ERP.
    bool pause = false;
    for (int i = 0; i < n_incs_record; ++i)
    {
        electrode_net.DoElectrodeMeasurements(V_ptr);
        mouse_cortex.Simulate();
        for (int j = 0; j < h_regions_inds.size() - 1; ++j)
            graphics_voltage_handle[j] = avg_V_handle[j];
        Graphics::render(graphics_voltage_handle, nullptr, 0, 0, pause);
    }
}

// Simulates ERP for three cases: realistic connectivity, realistic short range and uniform long range, and uniform long range. 
// For FIGURE 4.
void SimulatedERP(int const N)
{
    MouseCortex::excitatory_conns_file_name = "cortex_connectivity_1m_1k_exc.h5";
    MouseCortex::inhibitory_conns_file_name = "cortex_connectivity_1m_1k_inh.h5";
    MouseCortex::uniform_random_Ncouple = 1000;

    // Figure out what the indices of the electrodes should be.
    thrust::host_vector<int> h_regions, h_regions_inds;
    {
        MouseCortex mouse_cortex(N);
        mouse_cortex.morphology->SpecfiyRegionsDepth(7, "CUBIC_regionID_lookup_no_comma.json");
        mouse_cortex.morphology->GenRegionsFromIDs(h_regions, h_regions_inds);
    }
    ElectrodeNet electrode_net(h_regions, h_regions_inds);
    Float const* avg_V_handle = electrode_net.ElectrodeOutputHandle();
    int n_incs_transient = NumIncsSecond() * 3;
    int n_incs_record = NumIncsSecond();

    // Simulate and record the three cases: gray + white, gray + uniform, uniform only.
    bool use_uniform_white[3]{ false, true, true };
    bool use_gray[3]{ true, true, false };
    std::string fout_names[3]{ "ERP_OUT_GRAY_WHITE.csv", "ERP_OUT_GRAY_UNIFORM.csv", "ERP_OUT_UNIFORM.csv" };
    for(int case_idx = 0; case_idx < 3; ++case_idx)
    {
        MouseCortex mouse_cortex(N, "HH", use_uniform_white[case_idx], use_gray[case_idx]);
        std::ofstream fout(fout_names[case_idx]);
        Float const* V_ptr = mouse_cortex.ephys->SimVarsCpu()[HH::Params::V];

        // Transient.
        for (int i = 0; i < n_incs_transient; ++i)
            mouse_cortex.Simulate();

        // Simulate and record ERP.
        for (int i = 0; i < n_incs_record; ++i)
        {
            electrode_net.DoElectrodeMeasurements(V_ptr);
            mouse_cortex.Simulate();
            for (int i = 0; i < h_regions_inds.size() - 1; ++i)
                fout << avg_V_handle[i] << ',';
            fout << '\n';
        }
        fout.close();
    }
}

int main()
{
    SimulatedERP(1e5);
    exit(0);
    for(int i = 1; i < 10; ++i)
        RecordNumFiring(i * 1e5, std::to_string(i));
    WavesExamples(1e5, 4e4, 10);
    return 0;
    MouseCortex::excitatory_conns_file_name = "cortex_connectivity_13M_100_exc.h5";
    MouseCortex::inhibitory_conns_file_name = "cortex_connectivity_13M_100_inh.h5";
    MouseCortex::uniform_random_Ncouple = 100;

    // Logarithmic scale choices of N.
    std::vector<int> choices_of_N(30);
    float exp_min = 4;
    float exp_max = 7.12;

    // Vary exponent linearly.
    for (int i = 0; i < choices_of_N.size(); ++i)
    {
        float exp = exp_min + (exp_max - exp_min) * i / (choices_of_N.size() - 1);
        choices_of_N[i] = pow(10, exp);
    }
    SweepBenchmark(choices_of_N);
}
