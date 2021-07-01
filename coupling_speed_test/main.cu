#include "mouse_cortex.cuh"
#include "graphics.h"
#include <set>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <chrono>

// NOTE THAT: this benchmark defaults to using uniform random connections and no gray matter. 
// These can both be switched so that the gray and white matter are realistic. 
void SweepBenchmark(std::vector<int> const& choices_of_N, std::string neuron_type = "HH", bool uniform_random_white=true, bool use_gray=false)
{
    std::ofstream fout("HH_VARIED_N_OUT.csv");
    for (int N : choices_of_N)
    {
        {
            std::cout << "N " << N << std::endl;
            MouseCortex mouse_cortex(N, neuron_type, uniform_random_white, use_gray);
            Float dt = HH::leapfrog_dt;
            int n_incs = 1000 / dt; // Number of increments to simulate a second.

            // Start of simulation.
            auto start = std::chrono::steady_clock::now();

            // Run simulation.
            for (int i = 0; i < n_incs; ++i)
                mouse_cortex.Simulate();

            // End of simulation.
            auto stop = std::chrono::steady_clock::now();

            // Measure elapsed time and write to row of file in format <N> <TIME>
            float time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.0f;
            fout << N << ',' << time << std::endl;
        }
    }
    fout.close();
}

int main()
{
    MouseCortex::excitatory_conns_file_name = "cortex_connectivity_13M_100_exc.h5";
    MouseCortex::inhibitory_conns_file_name = "cortex_connectivity_13M_100_inh.h5";
    MouseCortex::uniform_random_Ncouple = 1000;

    // Logarithmic scale choices of N.
    std::vector<int> choices_of_N(30);
    float exp_min = 4;
    float exp_max = 6.3;

    // Vary exponent linearly.
    for (int i = 0; i < choices_of_N.size(); ++i)
    {
        float exp = exp_min + (exp_max - exp_min) * i / (choices_of_N.size() - 1);
        choices_of_N[i] = pow(10, exp);
    }
    SweepBenchmark(choices_of_N);
}
