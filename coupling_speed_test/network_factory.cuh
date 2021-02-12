#pragma once
#include "coupling.cuh"
#include <ANN/ANN.h>

// Handles things related to creation of connectivities, localized information, regions, etc.
class NetworkFactory 
{
private:
	int N;

	// Stuff used for localized information gathering.
	std::unique_ptr<ANNkd_tree> kd_tree;
	ANNpointArray p_vec; // Positions in ANN format.

public:
	NetworkFactory (int N_, float const* positions)
		: N{N_}, kd_tree{nullptr}, p_vec{ annAllocPts(N, 3) }
	{
		float const* pos_it = positions;
        for (int i = 0; i < N; ++i)
            for (int dim = 0; dim < 3; ++dim)
                p_vec[i][dim] = *pos_it++;
		kd_tree = std::make_unique<ANNkd_tree>(p_vec, N, 3);
	}

	// Establishes a uniform random network coupling with N neurons.
	// Ncouple is the number of neurons each neuron is connected to.
	void UniformNetwork(thrust::host_vector<int>& h_coupling, thrust::host_vector<int>& h_coupling_inds, int Ncouple);

	// Read hdf file and convert to AdjacenyList
	void BrainConnectivity(thrust::host_vector<int>& h_coupling, thrust::host_vector<int>& h_coupling_inds, const H5std_string file_name);

	// 
	//! Localized coupling where a neuron is connected to the k-nearest neighbors.
	/*!
	  \param [out] h_coupling Connectivity column indices for connectivity matrix.
	  \param [out] h_coupling_inds Connectivity row offsets for connectivity matrix.
	  \param [in] k Number of nearest neighbors to couple to.
	*/
	void LocalConnectivity(thrust::host_vector<int>& h_coupling, thrust::host_vector<int>& h_coupling_inds, int k);

	// 
	//! Generate a lookup table of k nearest neighbors per neuron. 
	/*!
	  \param [in] k Number of nearest neighbors to record.
	*/	
	std::vector<std::vector<int>> KNN(int const k);

	// 
	//! Generate a list of spherical regions at specific neurons.
	/*!
		\param [out] h_regions Column indices for region adjacency list. 
		\param [out] h_region_inds Row offsets for region adjacency list.
		\param [in] electrode_inds Indices of neurons at which to place electrodes.
		\param [in] k Number of nearest neighbors for regions.
	*/	
	void GenSphericalRegions(thrust::host_vector<int>& h_regions, thrust::host_vector<int>& h_region_inds, std::vector<int> const& electrode_inds, int const k);

	// Convert coupling stored as DOWNSTREAM network to upstream matrix connectivity. 
	CuSparseAdjacencyMatrix DownstreamToUpstreamSpMatrix(cusparseHandle_t& handle, AdjacencyList& list);

	~NetworkFactory () { annClose(); }
};
