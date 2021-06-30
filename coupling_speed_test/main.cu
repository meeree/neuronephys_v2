#include "mouse_cortex.cuh"
#include "graphics.h"
#include <set>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <chrono>

int main()
{
    MouseCortex mouse_cortex(1e5);
    std::ofstream n_firing_out("N_FIRING.txt");
    Float dt = HH::leapfrog_dt;
    int n_incs = 1000 / dt; // Number of increments to simulate a second.

    // Start of simulation.
    auto start = std::chrono::steady_clock::now();

    // Run simulation.
    for (int i = 0; i < n_incs; ++i)
        n_firing_out << mouse_cortex.Simulate() << std::endl;
    n_firing_out.close();

    // End of simulation.
    auto stop = std::chrono::steady_clock::now();
    
    // Measure elapsed time and print.
    std::cout << "Elapsed time in seconds: "
        << std::chrono::duration_cast<std::chrono::seconds>(stop - start).count()
        << " s" << std::endl;
}
