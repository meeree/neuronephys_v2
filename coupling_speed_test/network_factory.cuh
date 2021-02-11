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
	//! Localized coupling where a neuron is connected to the k-nearest neighbors.
	/*!
	  \param [out] h_coupling Connectivity column indices for connectivity matrix.
	  \param [out] h_coupling_inds Connectivity row offsets for connectivity matrix.
	  \param [in] N Number of neurons.
	  \param [in] k Number of nearest neighbors to couple to.
	  \param [in] positions Morphology given as a float pointer. 
	*/
	void LocalConnectivity(thrust::host_vector<int>& h_coupling, thrust::host_vector<int>& h_coupling_inds, int const N, int k, float const* positions);

	// 
	//! Generate a lookup table of k nearest neighbors per neuron. 
	/*!
	  \param [in] positions Morphology given as a float pointer. 
	  \param [in] N Number of neurons.
	  \param [in] k Number of nearest neighbors to record.
	*/	
	std::vector<std::vector<int>> KNN(float const* positions, int const N, int const k);

	// Convert coupling stored as DOWNSTREAM network to upstream matrix connectivity. 
	CuSparseAdjacencyMatrix DownstreamToUpstreamSpMatrix(cusparseHandle_t& handle, AdjacencyList& list);
};
