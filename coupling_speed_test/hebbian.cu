#include "hebbian.cuh"

struct is_zero
{
    __host__ __device__
    bool operator()(const int x)
    {
        return x == 0;
    }
};

struct scale
{
    Float scalar;
    scale(Float scalar_) : scalar{ scalar_ } {}
    __host__ __device__
    Float operator()(const Float V)
    {
        return V * scalar;
    }
};

struct set_zero 
{
    __host__ __device__
    int operator()(const int)
    {
        return 0;
    }
};

struct sub_one
{
    __host__ __device__
    int operator()(const int T)
    {
        return T - 1;
    }
};

inline __global__ void set_timers (int const num_firing,
    int const window_duration,
	int* timers,
	int const* connectivity_offsets,
	int const* firing_inds)
{
	for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < num_firing; tid += gridDim.x * blockDim.x) {
		int const pre_idx = firing_inds[tid];
		int* timer = timers + connectivity_offsets[pre_idx];
        for (; timer != timers + connectivity_offsets[pre_idx + 1]; ++timer)
            *timer = window_duration;
	}
}

void Hebbian::UpdateWeights(AdjacencyList const& connectivity, thrust::device_vector<int> const& firing)
{
	// Set Tk to window_duration for all downstream synapses Sk to any neuron that is firing. 
    set_timers << <NBLOCKS, NTHREADS >> > (
        num_firing, window_duration, thrust::raw_pointer_cast(connectivity.timers.data())
        thrust::raw_pointer_cast(connectivity.indices_chunks.data()), thrust::raw_pointer_cast(connectivity.offset_in_indices_chunks.data()),
        thrust::raw_pointer_cast(firing_neurons.data()));
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	// Let R <- {Sk | Tk > 0}. 
	// See second example here: https://thrust.github.io/doc/classthrust_1_1counting__iterator.html
	auto const R_end = thrust::copy_if(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(Nconns),
        connectivity.timers.begin(),
        R.begin(),
        thrust::identity<int>());

    // Let R0 <- {Sk in R | F[dk] = 0}. 
    // I know this is gross, sry :). It's still faster than writing a kernel myself.
	auto const R0_end = thrust::copy_if(
        R.begin(),
        R_end,
        thrust::make_permutation_iterator(firing.begin(), R.begin()),
        R0.begin(), 
        is_zero());

    // Let R1 <- {Sk in R | F[dk] = 1}. 
	auto const R1_end = thrust::copy_if(
        R.begin(),
        R_end,
        thrust::make_permutation_iterator(firing.begin(), R.begin()),
        R1.begin(), 
        thrust::identity<int>());

    // For all Sk in R0, Tk |-> Tk - 1.
    thrust::transform(
        thrust::make_permutation_iterator(connectivity.timers.begin(), R0.begin()),
        thrust::make_permutation_iterator(connectivity.timers.begin(), R0_end),
        thrust::make_permutation_iterator(connectivity.timers.begin(), R0.begin()),
        sub_one());

    // Let R0 = {Sk in R0 | Tk = 0}.
	auto const R0_end = thrust::copy_if(
        R0.begin(),
        R0_end,
        thrust::make_permutation_iterator(connectivity.timers.begin(), R0.begin()),
        R0.begin(), 
        is_zero());

    // HEBBIAN DOWNSCALE: For all Sk in R0, Vk |-> Vk * downscale.
    thrust::transform(
        thrust::make_permutation_iterator(connectivity.vals.begin(), R0.begin()),
        thrust::make_permutation_iterator(connectivity.vals.begin(), R0_end),
        thrust::make_permutation_iterator(connectivity.vals.begin(), R0.begin()),
        scale(downscale));

    // For all Sk in R1, Tk |-> 0.
    thrust::transform(
        thrust::make_permutation_iterator(connectivity.timers.begin(), R1.begin()),
        thrust::make_permutation_iterator(connectivity.timers.begin(), R1_end),
        thrust::make_permutation_iterator(connectivity.timers.begin(), R1.begin()),
        set_zero());

    // HEBBIAN UPSCALE: For all Sk in R1, Vk |-> Vk * upscale.
    thrust::transform(
        thrust::make_permutation_iterator(connectivity.vals.begin(), R1.begin()),
        thrust::make_permutation_iterator(connectivity.vals.begin(), R1_end),
        thrust::make_permutation_iterator(connectivity.vals.begin(), R1.begin()),
        scale(upscale));
}
