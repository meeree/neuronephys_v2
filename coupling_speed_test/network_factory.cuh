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

	// 
	//! Localized small-world style coupling
	/*!
	  \param [out] h_coupling Connectivity column indices for connectivity matrix.
	  \param [out] h_coupling_inds Connectivity row offsets for connectivity matrix.
	  \param [in] N Number of neurons.
	  \param [in] N_conns Number of downstream connections per neuron.
	  \param [in] stddev Standard deviation for small-world coupling. Given as a percentage of world-space region, which is assumed to be [-1,-1,-1] to [1,1,1] 
	  \param [in] search_scalar Controls how many standard deviations we search for neurons to connect to. E.g., if this is 1, we only consider connections 1 stddev away. 
	  \param [in] positions Morphology given as a float pointer. 
	*/
	void LocalConnectivity(thrust::host_vector<int>& h_coupling, thrust::host_vector<int>& h_coupling_inds, int const N, int const N_conns,
		                   float stddev, float const search_scalar, float const* positions);

	// Convert coupling stored as DOWNSTREAM network to upstream matrix connectivity. 
	CuSparseAdjacencyMatrix DownstreamToUpstreamSpMatrix(cusparseHandle_t& handle, AdjacencyList& list);
};
