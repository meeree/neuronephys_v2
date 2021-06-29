#include "mouse_cortex.cuh"
#include "graphics.h"
#include <set>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/type_ptr.hpp>

int main()
{
    MouseCortex mouse_cortex(1e5);
    std::ofstream n_firing_out("N_FIRING.txt");
    for (int i = 0; i < 2500; ++i)
        n_firing_out << mouse_cortex.Simulate() << std::endl;
    n_firing_out.close();
}
