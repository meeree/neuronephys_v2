#pragma once
#include "coupling.cuh"

// Implements a networks of "electrodes" measuring averaged activity in regions. 
// Regions are stored as an adjacency list.
class ElectrodeNet 
{
private:
	AdjacencyList d_regions;
	int N_regions;
	thrust::device_vector<Float> d_avg_voltages;
	thrust::host_vector<Float> h_avg_voltages;
public:
	ElectrodeNet (thrust::host_vector<int> const& regions_inds, 
	              thrust::host_vector<int> const& regions_offsets)
		: d_regions{regions_inds, regions_offsets}, 
		  N_regions{(int)regions_offsets.size() - 1},
		  d_avg_voltages(N_regions, 0),
		  h_avg_voltages(N_regions, 0)
	{
	}

	// Computes the averaged activity in each region and then copies to CPU in h_avg_voltages.
	void DoElectrodeMeasurements (Float const* V_ptr);
	
	Float const* ElectrodeOutputHandle () const
	{
		return h_avg_voltages.data();
	}
};