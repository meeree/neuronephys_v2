#pragma once
#include "coupling.cuh"

// TODO: ORGANIZE
namespace NetworkFactory
{
	// Establishes a uniform random network coupling with N neurons.
	// Ncouple is the number of neurons each neuron is connected to.
	void UniformNetwork(thrust::host_vector<int>& h_coupling, thrust::host_vector<int>& h_coupling_inds, int N, int Ncouple);

	// Read hdf file and convert to AdjacenyList
	void BrainConnectivity(thrust::host_vector<int>& h_coupling, thrust::host_vector<int>& h_coupling_inds, int const N, const H5std_string file_name);

	// Convert coupling stored as DOWNSTREAM network to upstream matrix connectivity. 
	CuSparseAdjacencyMatrix DownstreamToUpstreamSpMatrix(cusparseHandle_t& handle, AdjacencyList& list);
};
