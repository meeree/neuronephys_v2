#pragma once
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
//#include <chrono>
#include <vector>

// we need these includes for CUDA's random number stuff
#include <curand.h>
#include <curand_kernel.h>

//! Toggle for using random seed that is fixed. If fixed, code is easier to reproduce.
#define USE_FIXED_RANDOM_SEED 1 

//! Toggle for enabling/disabling double precision.
#define USE_DOUBLE_PRECISION 0 
#if USE_DOUBLE_PRECISION
typedef double Float;
#else
typedef float Float;
#endif

//! Just some utility functions that can be used throughout the codebase.
namespace Utility
{
    //! Get random engine. This is defined to be seeded/unseeded based on macro USE_FIXED_RANDOM_SEED in defines.cuh.
    std::default_random_engine& GetRandomEngine();

    //! Get a timestamp that can be appended to file names so we know when the data was recorded.
    std::string GetTimestampString();

    //! Write a bunch of data to an HDF file. For example, neuron voltages, paracrine density, etc.
    //! \param file_name Name of output file in hdf format.
    //! \param data Data to write as a raw float pointer. To get this pointer from a vector use vector.data(). 
    //! \param dims Dimensions of the data. For examples, for N electrodes over T timesteps, dims could be {N, T}.
    //! \params prepend_timestamp If set to true, we add a timestamp to the file name string. Defaults to true.
    bool WriteDataToHDF(std::string file_name, Float const* data, std::vector<hsize_t> dims, bool prepend_timestamp = true);
};

// TODO : SET NUMBER OF THREADS DYNAMICALLY BASED ON CARD USED
#define NTHREADS 512 
#define NBLOCKS 32

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
