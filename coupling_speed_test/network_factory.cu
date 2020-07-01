#include "network_factory.cuh"
#include <unordered_set>

namespace NetworkFactory
{
	void UniformNetwork(thrust::host_vector<int>& h_coupling, thrust::host_vector<int>& h_coupling_inds, int N, int Ncouple)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> int_dist(0, N-1);

		h_coupling.resize(N * Ncouple);
		h_coupling_inds.resize(N + 1, 0);
		thrust::host_vector<int> neuron_downstream(Ncouple);
		std::unordered_set<int> set(Ncouple);
		for (int i = 0; i < N; ++i)
		{
			thrust::fill_n(neuron_downstream.begin(), Ncouple, INT_MAX);
			for (int j = 0; j < Ncouple; ++j)
			{
				// This selects random numbers without making duplicates.
				// We use a set for constant time insert and lookup and clear it each time. 
				int downstream;
				do
				{
					downstream = int_dist(gen);
				} while (set.find(downstream) != set.end());
				neuron_downstream[j] = downstream;
				set.insert(downstream);
			}
			thrust::copy_n(neuron_downstream.begin(), Ncouple, h_coupling.begin() + i * Ncouple);
			h_coupling_inds[i] = i * Ncouple;
			set.clear();
		}
		h_coupling_inds[N] = N * Ncouple;
	}

	void BrainConnectivity(thrust::host_vector<int>& h_coupling, thrust::host_vector<int>& h_coupling_inds, int const N, const H5std_string file_name)
	{
		using namespace H5;
		const H5std_string fl_name = "../" + file_name;
		int numconns, numpnts;
		std::vector<int> col_idx, row_ptr;
		try
		{
			H5File file(fl_name, H5F_ACC_RDONLY);
			DataSet col_idx_arr = file.openDataSet("/col_idx");
			numconns = (int)(col_idx_arr.getStorageSize() / sizeof(int));
			col_idx.resize(numconns);
			col_idx_arr.read(col_idx.data(), PredType::NATIVE_INT);

			DataSet row_ptr_arr = file.openDataSet("/row_ptr");
			numpnts = (int)(row_ptr_arr.getStorageSize() / sizeof(int));
			row_ptr.resize(numpnts);
			row_ptr_arr.read(row_ptr.data(), PredType::NATIVE_INT);
		}
		catch (...)
		{
			throw std::runtime_error("ERROR : Failed to read in coupling");
		}
		numpnts -= 1; // row_ptr has one extra element.
		if (N > numpnts)
			throw std::runtime_error("Not enough points in file " + fl_name  + ". Have N = " + std::to_string(N));

		int sparsity{ numpnts / N };
		h_coupling.resize(numconns);
		h_coupling_inds.resize(numpnts + 1, 0);
		auto it = h_coupling.begin();
		for (int from_idx = 0; from_idx < N; ++from_idx)
		{
			int const lookup_idx{ from_idx * sparsity };
			for (int to_idx = row_ptr[lookup_idx]; to_idx < row_ptr[lookup_idx + 1]; ++to_idx)
				if (col_idx[to_idx] % sparsity == 0)
					*it++ = col_idx[to_idx] / sparsity;
			h_coupling_inds[from_idx + 1] = it - h_coupling.begin();
		}
	}

	CuSparseAdjacencyMatrix DownstreamToUpstreamSpMatrix(cusparseHandle_t& handle, AdjacencyList& list)
	{
		// Since CuSparseAdjacencyMatrix is CSC format, and a column of the matrix contains downstream neurons, 
		// we don't need to do any conversion stuff, yay! 
		CuSparseAdjacencyMatrix mat{ handle, list };
		mat.vals.resize(list.indices.size(), 1);
		return mat;
	}
};