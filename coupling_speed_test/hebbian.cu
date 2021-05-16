#include "hebbian.cuh"

struct set_timer
{
    int window_duration;
    set_timer(int window_duration_) : window_duration{ window_duration_ } {}
    __host__ __device__
    int operator()(int timer, int is_firing)
    {
        return is_firing ? window_duration : (timer > 0 ? timer - 1 : 0);
    }
};

inline __global__ void hebbian_rule_kernel(int const Nconns,
	int const* connectivity,
    int const* upstream,
    Float* weights,
    int const* timers,
	int const* is_firing,
    Float heb_inc)
{
	for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < Nconns; tid += gridDim.x * blockDim.x) 
    {
        int const pre_idx = upstream[tid];
        int const post_idx = connectivity[tid];

        // HEBBIAN RULE :)
        Float alpha = expf(-fabsf(weights[tid] - 1)); 
        weights[tid] += (is_firing[post_idx] ? 1 : -1) * heb_inc * alpha;
	}
}

void Hebbian::UpdateWeights(AdjacencyList& connectivity, thrust::device_vector<int> const& is_firing)
{
    // Set timers to window_duration for all neurons that are firing.
    // If neuron is not firing, decrement timer if nonzero.
    thrust::transform(
        timers.begin(),
        timers.end(), 
        is_firing.begin(),
        timers.begin(),
        set_timer(window_duration)
    );

    hebbian_rule_kernel << <NBLOCKS, NTHREADS >> > (connectivity.indices.size(),
        thrust::raw_pointer_cast(connectivity.indices.data()),
        thrust::raw_pointer_cast(connectivity.upstream.data()),
        thrust::raw_pointer_cast(connectivity.weights.data()),
        thrust::raw_pointer_cast(timers.data()),
        thrust::raw_pointer_cast(is_firing.data()),
        heb_inc);
}
