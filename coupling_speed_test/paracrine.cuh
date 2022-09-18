#include "defines.cuh"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include "cuda_runtime.h"


//Create Paracrine Class
class paracrine {
public:
	const int grid_size; //size of 1 dimension of the grid (assuming cube grid)
	const int nnz; //number of neurons
	const Float dx;
	const Float dt;
	thrust::device_vector<Float> neuron_x; //locations of neurons in x direction
	thrust::device_vector<Float> neuron_y; //locations of neurons in y direction
	thrust::device_vector<Float> neuron_z; //locations of neurons in z direction
	thrust::host_vector<Float> grid_IC; //IC for each gridpoint (size grid_size**3)
	thrust::host_vector<Float> neuron_IC; //IC for each neuron (size nnz)

	thrust::host_vector<Float> Q; //used in trilinear interpolation
	thrust::host_vector<Float> weighted_spread; //used in trilinear interpolation


	thrust::host_vector<Float> ones; //vector of ones (size nnz) used in spreading

	//Diffusion Variables
	thrust::device_vector<Float> stencil;
	//Initialize mask dimensions for the 3d convolution
	int mask_depth = 3;
	int mask_height = 3;
	int mask_width = 3;

	thrust::device_vector<Float> eye;
	thrust::device_vector<Float> A;
	thrust::device_vector<Float> B;
	thrust::device_vector<Float> b;
	thrust::device_vector<Float> laplacian_grid;
	const Float diffusion;
	const Float decay;
	const Float growth_rate;


	//Paracrine Constructor
	paracrine(const int grid_size, const int nnz, const Float dx, const Float dt, const Float diffusion, const Float decay, const Float growth_rate, const thrust::host_vector<Float> neuron_x, const thrust::host_vector<Float> neuron_y, \
		const thrust::host_vector<Float> neuron_z, const thrust::host_vector<Float> grid_IC, const thrust::host_vector<Float> neuron_IC)\
		: grid_size(grid_size), nnz(nnz), dx(dx), dt(dt), diffusion(diffusion), decay(decay), growth_rate(growth_rate), neuron_x(neuron_x), neuron_y(neuron_y), neuron_z(neuron_z), grid_IC(grid_IC), neuron_IC(neuron_IC)
	{//gather parameters by this constructor
		std::cout << "Paracrine Parameters Saved" << std::endl;
	};

	//Member functions
	void initialize();

	Float inner_product(thrust::device_vector<Float> A, thrust::device_vector<Float> B);
	thrust::device_vector<Float> convolve(thrust::device_vector<Float> grid, thrust::device_vector<Float> mask);
	thrust::device_vector<Float> CG(thrust::device_vector<Float> A, thrust::device_vector<Float> b, const int max_iter, const Float error_tolerance);

	thrust::device_vector<Float> interpolate(thrust::device_vector<Float> grid);
	thrust::device_vector<Float> spread(thrust::device_vector<Float> grid, thrust::device_vector<Float> neuron_concentrations);
	thrust::device_vector<Float> update_diffusion(thrust::device_vector<Float> grid);
	//thrust::device_vector <Float> diffusion_stepper(thrust::device_vector<Float> grid);
	//thrust::device_vector<Float> initial_guess(thrust::device_vector<Float> grid, thrust::device_vector<Float> laplacian_grid);

};

