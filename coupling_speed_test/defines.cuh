#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <fstream>
#include <stdio.h>
#include <random>
#include <H5Cpp.h>
#include <chrono>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define USE_DOUBLE_PRECISION 0 
#if USE_DOUBLE_PRECISION
typedef double Float;
#else
typedef float Float;
#endif // USE_DOUBLE_PRECISION
//SIM: Hodgkin Huxley
#define SIM_PARAMS_V 0
#define SIM_PARAMS_M 1
#define SIM_PARAMS_N 2
#define SIM_PARAMS_H 3
#define SIM_PARAMS_Y 4
#define SIM_PARAMS_OUTPUT 5
#define SIM_PARAMS_TIN 6//input synaptic neurotransmitter
#define SIM_PARAMS_TCOUNT 7//count of presynaptic neurons
#define SIM_PARAMS_APPCUR 8
#define NUM_SIM_PARAMS 9//Dimension of state space

// TODO : SET NUMBER OF THREADS DYNAMICALLY BASED ON CARD USED
#define NTHREADS 64 
#define NBLOCKS 32
#define NCHUNKS 100
#define N_PER_CHUNK 10000
#define NSAMPLES_PRINT 501

namespace hh_params {
    constexpr static const Float rk4_dt = 0.05;                                                          // Timestep used with RK4 numerical method (ms)
    constexpr static const Float leapfrog_dt = 0.4;                                                      // Timestep used with leapfrog numerical method (ms)
    constexpr static const Float C = 1;
    constexpr static const Float gna = 120;  constexpr static const Float ena = 55;                      // Sodium conductance and potential 
    constexpr static const Float gk = 36; constexpr static const Float ek = -72;                         // Potassium conductance and potential
    constexpr static const Float gl = 0.3; constexpr static const Float el = -50;                        // Leak conductance and potential 
    constexpr static const Float Tmax = 1.0;                                                             // Max neurotransmitter output
    constexpr static const Float Vt = -20.0; constexpr static const Float Kp = 3.0;                      // Neurotransmitter potential and scalar 
    constexpr const static Float Ar = 5.0; constexpr const static Float Ad = 0.18;                       // alpha and beta for GABA gating variable
    constexpr static const Float g_coupling = 0.01; constexpr static const Float e_coupling_gaba = -70.0; // Coupling conductance and potential for GABA
    constexpr static const Float coupling_threshold = -50;                                               // Threshold for GABA coupling
    constexpr static const Float appcur = 8;                                                            // Base current to apply when we only apply a constant current. 
}


//PARAMETERS FOR AN MODEL with INHIBITORY-EXCITATORY NETWORK
//excitatory: AMPA, NMDA. inhibitory: GABA
#define AN_IE_VAR_V 0
#define AN_IE_VAR_NK 1 
#define AN_IE_VAR_CA 2
#define AN_IE_VAR_YE 3
#define AN_IE_VAR_YI 4
#define AN_IE_VAR_EIN 5 //excitatory input synaptic neurotransmitter
#define AN_IE_VAR_ECOUNT 6//count of presynaptic excitatory neurons
#define AN_IE_VAR_EOUTPUT 7//neurotransmitter output
#define AN_IE_VAR_IIN 8 //inhibitory input synaptic neurotransmitter
#define AN_IE_VAR_ICOUNT 9//count of presynaptic inhibitory neurons
#define AN_IE_VAR_IOUTPUT 10//neurotransmitter output
#define NUM_AN_IE_VAR 11 //Dimension of state space
namespace an_ie_params {
    constexpr static const Float rk4_dt = 0.04;
    constexpr static const Float an_Vt = -45; constexpr static const Float an_Kp = 1;
    constexpr static const Float an_coupling_threshold = -60;
    constexpr static const Float Tmax = 1.0;
    constexpr static const Float C = 1;
    constexpr static const Float A = 0.02;
    constexpr static const Float vL = -60.95;
    constexpr static const Float vNa = 55;
    constexpr static const Float vK = -100;
    constexpr static const Float tauhA = 15;
    constexpr static const Float vCa = 120;
    constexpr static const Float kD = 30;
    constexpr static const Float vAMPA = 0;
    constexpr static const Float vNMDA = 0;
    constexpr static const Float vGABA = -70;
    constexpr static const Float alphaCa = 0.5;
    constexpr static const Float tauAMPA = 2;
    constexpr static const Float tausNMDA = 100;
    constexpr static const Float tauxNMDA = 2;
    constexpr static const Float tauGABA = 10;
    constexpr static const Float gL = 0.016307;
    constexpr static const Float gNa = 12.2438;
    constexpr static const Float gK = 19.20436;
    constexpr static const Float gA = 1.79259;
    constexpr static const Float gKS = 0.0350135;
    constexpr static const Float gNaP = 0.63314;
    constexpr static const Float gAR = 0.0166454;
    constexpr static const Float gCa = 0.1624;
    constexpr static const Float gKCa = 0.7506;
    constexpr static const Float gAMPA = 0.03;
    constexpr static const Float gNMDA = 0.0; // 0.0434132;
    constexpr static const Float gGABA = 0.0; //TODO: Change to non-averaged value
    constexpr static const Float tauCa = 739.09;
    constexpr static const Float white_noise_mean = 0.1f;
    constexpr static const Float white_noise_intensity = 0.05f; // Controls stddev of white noise.
};
