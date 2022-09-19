#include "paracrine.cuh"
#include <cmath>        // std::abs
using namespace thrust::placeholders;

//Paracrine Initialization Function
void paracrine::initialize() {
	std::cout << "Beginning Paracrine Initialization" << std::endl;

	//Initialize Diffusion Part
	//Initialize Stencil, eye, A, B (used in diffusion)
	for (int i = 0; i < 27; i++) {
		stencil.push_back(0.0);
		eye.push_back(0.0);
		A.push_back(0.0);
		B.push_back(0.0);
	}
	int stencil_height = 3;
	int stencil_depth = 3;

	//The stencil is flattened using the scheme stencil[x,y,z] = stencil[stencil_height * stencil_depth * x + stencil_depth * y + z];
	//x=0 plane
	stencil[stencil_height * stencil_depth * 0 + stencil_depth * 0 + 0] = 1;// / (30 * dx * dx); //stencil[0,0,0]
	stencil[stencil_height * stencil_depth * 0 + stencil_depth * 1 + 0] = 3;// / (30 * dx * dx); //stencil[0,1,0]
	stencil[stencil_height * stencil_depth * 0 + stencil_depth * 2 + 0] = 1;// / (30 * dx * dx); //stencil[0,2,0]

	stencil[stencil_height * stencil_depth * 0 + stencil_depth * 0 + 1] = 3;// / (30 * dx * dx); //stencil[0,0,1]
	stencil[stencil_height * stencil_depth * 0 + stencil_depth * 1 + 1] = 14;// / (30 * dx * dx); //stencil[0,1,1]
	stencil[stencil_height * stencil_depth * 0 + stencil_depth * 2 + 1] = 3;// / (30 * dx * dx); //stencil[0,2,1]

	stencil[stencil_height * stencil_depth * 0 + stencil_depth * 0 + 2] = 1;// / (30 * dx * dx); //stencil[0,0,2]
	stencil[stencil_height * stencil_depth * 0 + stencil_depth * 1 + 2] = 3;// / (30 * dx * dx); //stencil[0,1,2]
	stencil[stencil_height * stencil_depth * 0 + stencil_depth * 2 + 2] = 1;// / (30 * dx * dx); //stencil[0,2,2]

	//x=1 plane
	stencil[stencil_height * stencil_depth * 1 + stencil_depth * 0 + 0] = 3;// / (30 * dx * dx); //stencil[1,0,0]
	stencil[stencil_height * stencil_depth * 1 + stencil_depth * 1 + 0] = 14;// / (30 * dx * dx); //stencil[1,1,0]
	stencil[stencil_height * stencil_depth * 1 + stencil_depth * 2 + 0] = 3;// / (30 * dx * dx); //stencil[1,2,0]

	stencil[stencil_height * stencil_depth * 1 + stencil_depth * 0 + 1] = 14;// / (30 * dx * dx); //stencil[1,0,1]
	stencil[stencil_height * stencil_depth * 1 + stencil_depth * 1 + 1] = -128;// / (30 * dx * dx); //stencil[1,1,1]
	stencil[stencil_height * stencil_depth * 1 + stencil_depth * 2 + 1] = 14;// / (30 * dx * dx); //stencil[1,2,1]

	stencil[stencil_height * stencil_depth * 1 + stencil_depth * 0 + 2] = 3;// / (30 * dx * dx); //stencil[1,0,2]
	stencil[stencil_height * stencil_depth * 1 + stencil_depth * 1 + 2] = 14;// / (30 * dx * dx); //stencil[1,1,2]
	stencil[stencil_height * stencil_depth * 1 + stencil_depth * 2 + 2] = 3;// / (30 * dx * dx); //stencil[1,2,2]

	//x=2 plane
	stencil[stencil_height * stencil_depth * 2 + stencil_depth * 0 + 0] = 1;// / (30 * dx * dx); //stencil[2,0,0]
	stencil[stencil_height * stencil_depth * 2 + stencil_depth * 1 + 0] = 3;// / (30 * dx * dx); //stencil[2,1,0]
	stencil[stencil_height * stencil_depth * 2 + stencil_depth * 2 + 0] = 1;// / (30 * dx * dx); //stencil[2,2,0]

	stencil[stencil_height * stencil_depth * 2 + stencil_depth * 0 + 1] = 3;// / (30 * dx * dx); //stencil[2,0,1]
	stencil[stencil_height * stencil_depth * 2 + stencil_depth * 1 + 1] = 14;// / (30 * dx * dx); //stencil[2,1,1]
	stencil[stencil_height * stencil_depth * 2 + stencil_depth * 2 + 1] = 3;// / (30 * dx * dx); //stencil[2,2,1]

	stencil[stencil_height * stencil_depth * 2 + stencil_depth * 0 + 2] = 1;// / (30 * dx * dx); //stencil[2,0,2]
	stencil[stencil_height * stencil_depth * 2 + stencil_depth * 1 + 2] = 3;// / (30 * dx * dx); //stencil[2,1,2]
	stencil[stencil_height * stencil_depth * 2 + stencil_depth * 2 + 2] = 1;// / (30 * dx * dx); //stencil[2,2,2]

	//Create 3D identity matrix "eye"
	//x=0 plane
	eye[stencil_height * stencil_depth * 0 + stencil_depth * 0 + 0] = 1;
	eye[stencil_height * stencil_depth * 0 + stencil_depth * 1 + 1] = 1;
	eye[stencil_height * stencil_depth * 0 + stencil_depth * 2 + 2] = 1;
	//x=1 plane
	eye[stencil_height * stencil_depth * 1 + stencil_depth * 0 + 0] = 1;
	eye[stencil_height * stencil_depth * 1 + stencil_depth * 1 + 1] = 1;
	eye[stencil_height * stencil_depth * 1 + stencil_depth * 2 + 2] = 1;
	//x=2 plane
	eye[stencil_height * stencil_depth * 2 + stencil_depth * 0 + 0] = 1;
	eye[stencil_height * stencil_depth * 2 + stencil_depth * 1 + 1] = 1;
	eye[stencil_height * stencil_depth * 2 + stencil_depth * 2 + 2] = 1;

	Float dum = 1 / (30 * dx * dx);
	thrust::transform(stencil.begin(), stencil.end(), stencil.begin(), dum * _1);
}

__global__ void gpu_interpolate(int nnz, int grid_size, Float dx, Float* grid, Float* neuron_x, Float* neuron_y, Float* neuron_z, Float* neuron_concentrations)
	//https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
{
	//Each neuron gets a thread 
	//int grid_height = grid_size;
	//int grid_depth = grid_size;
	//int rowsize = 8;
	
	for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < nnz; n += gridDim.x * blockDim.x) //neuron loop
	{
		//calculater the adjacent lower indices, the upper indices will just be +1
		//even we are using kernel with larger support, calculating one index still suffices
		int x_low_idx = floorf(neuron_x[n]/dx); 
		int y_low_idx = floorf(neuron_y[n]/dx);
		int z_low_idx = floorf(neuron_z[n]/dx);
		Float del_x = neuron_x[n] / dx - x_low_idx ;
		Float del_y = neuron_y[n] / dx - y_low_idx;
		Float del_z = neuron_z[n] / dx - z_low_idx;
		//iterate through the three faces

		for (int x_idx = 0; x_idx < 2; x_idx++) {
			for (int y_idx = 0; y_idx < 2; y_idx++) {
				for (int z_idx = 0; z_idx < 2; z_idx++) {
					//neuron_concentrations[n] += grid[grid_size * grid_size * (x_low_idx + x_idx) + grid_size * (y_low_idx + y_idx) + (z_low_idx + z_idx)]
						//*abs(x_idx+del_x-1)* abs(y_idx + del_y - 1)* abs(z_idx + del_z - 1);
					atomicAdd(&neuron_concentrations[n], grid[grid_size * grid_size * (x_low_idx + x_idx) + grid_size * (y_low_idx + y_idx) + (z_low_idx + z_idx)]
						* fabsf(x_idx + del_x - 1) * fabsf(y_idx + del_y - 1) * fabsf(z_idx + del_z - 1));
				}
			}
		}

	}

}

thrust::device_vector <Float> paracrine::interpolate(thrust::device_vector <Float> grid) { //return  thrust array of concentrations at neuron locations

	thrust::device_vector <Float> d_neuron_concentrations(nnz);
	//thrust::device_vector <Float> d_Q = Q;

	//Calculate blocksize and gridsize
	int block_size = 1024; //schedule a block full of number of neurons
	int cuda_grid_size = nnz / block_size + 1;
	//printf("test concentration is: %g ", (Float)grid[grid_size * grid_size * grid_size / 2 + grid_size * grid_size / 2 + grid_size / 2]);

	gpu_interpolate << < cuda_grid_size, block_size >> > (nnz, grid_size, dx, thrust::raw_pointer_cast(grid.data()), thrust::raw_pointer_cast(neuron_x.data()), thrust::raw_pointer_cast(neuron_y.data()),
		thrust::raw_pointer_cast(neuron_z.data()),thrust::raw_pointer_cast(d_neuron_concentrations.data()));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	return d_neuron_concentrations;
}

__global__ void gpu_spread(int nnz, int grid_size, Float dx, Float* grid, Float* neuron_x, Float* neuron_y, Float* neuron_z, Float* growth){
	//spread from neuron to grid
	
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nnz; i += gridDim.x * blockDim.x) { //neuron loop
        int x_low_idx = floorf(neuron_x[i] / dx);
        int y_low_idx = floorf(neuron_y[i] / dx);
        int z_low_idx = floorf(neuron_z[i] / dx);
        Float del_x = neuron_x[i] / dx - x_low_idx;
        Float del_y = neuron_y[i] / dx - y_low_idx;
        Float del_z = neuron_z[i] / dx - z_low_idx;
        //iterate through the three faces
        for (int x_idx = 0; x_idx < 2; x_idx++) {
            for (int y_idx = 0; y_idx < 2; y_idx++) {
                for (int z_idx = 0; z_idx < 2; z_idx++) {
                    atomicAdd(&grid[grid_size * grid_size * (x_low_idx + x_idx) + grid_size * (y_low_idx + y_idx) + (z_low_idx + z_idx)], 
                        growth[i] / (dx*dx*dx) * fabsf(x_idx + del_x - 1) * fabsf(y_idx + del_y - 1) * fabsf(z_idx + del_z - 1));	//factored by dx^3 because we are spreading the concentration		
                }
            }
        }
	}
	
}

//spread function
thrust::device_vector <Float> paracrine::spread(thrust::device_vector <Float> grid, thrust::device_vector <Float> neuron_growth) {


	  //WE ONLY HAVE TO SPREAD THE NEWLY GENERATED NEUROTRANSMITTER to the grid, i.e growth and add those to the grid

	//Calculate blocksize and gridsize
	int block_size = 1024; //schedule a block full of threads
	int cuda_grid_size = nnz / block_size + 1;

//	std::cout << 1 / dx << std::endl;
	gpu_spread << < cuda_grid_size, block_size >> > (nnz, grid_size, dx, thrust::raw_pointer_cast(grid.data()), thrust::raw_pointer_cast(neuron_x.data()), thrust::raw_pointer_cast(neuron_y.data()),
		thrust::raw_pointer_cast(neuron_z.data()), thrust::raw_pointer_cast(neuron_growth.data()));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	return grid;
}


__global__ void gpu_pad(Float* grid, Float* image, int grid_size, int image_size)
{
	//image is size ((grid_size+2) x (grid_size+2) x (grid_size+2))
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < grid_size; i += gridDim.x * blockDim.x) {
		for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < grid_size; j += gridDim.y * blockDim.y) {
			for (int k = blockIdx.z * blockDim.z + threadIdx.z; k < grid_size; k += gridDim.z * blockDim.z) {
				image[image_size * image_size * (i + 1) + image_size * (j + 1) + (k + 1)] = grid[grid_size * grid_size * i + grid_size * j + k];
			}
		}
	}
}

//3d Convolution kernel
__global__ void gpu_convolve(Float* image, Float* stencil, Float* result, int image_size, int mask_size, int grid_size) {

	//image is size ((grid_size+2) x (grid_size+2) x (grid_size+2))
	//stencil is size (3x3x3)
	//result will be (grid_size x grid_size x grid_size) thanks to the zero padding we did earlier

	//remember the flattening scheme
	//grid[grid_height * grid_depth * x + grid_depth * y + z] = grid[x,y,z]

	//Use the 3d architecture of the GPU to our advantage
	//nested loops to give result(i,j,k)
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < grid_size; i += gridDim.x * blockDim.x) {
		for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < grid_size; j += gridDim.y * blockDim.y) {
			for (int k = blockIdx.z * blockDim.z + threadIdx.z; k < grid_size; k += gridDim.z * blockDim.z) {

				//Now multiply element wise and sum to get actual values
				//Float sum = 0.0;
				for (int ii = 0; ii < mask_size; ii++) {
					for (int jj = 0; jj < mask_size; jj++) {
						for (int kk = 0; kk < mask_size; kk++) {
							//sum += (image[image_size * image_size * (i+ii) + image_size * (j+jj) + (k+kk)])
							//	    * (stencil[mask_size * mask_size * ii + mask_size*jj + kk]);
							atomicAdd(&result[grid_size * grid_size * i + grid_size * j + k], (image[image_size * image_size * (i + ii) + image_size * (j + jj) + (k + kk)]) *
								(stencil[mask_size * mask_size * ii + mask_size * jj + kk]));
						}
					}
				}
				//result[grid_size * grid_size * i + grid_size * j + k] = sum;

			}
		}
	}

}

thrust::device_vector <Float> paracrine::convolve(thrust::device_vector <Float> grid, thrust::device_vector <Float> mask) {

	int mask_size = 3;

	//Mask is 3x3x3 (in our case, 27-point discrete laplacian stencil)
	//Mask dimensions are initialized in paracrine class

	//Pad the image with 0's so that we retain the same size after convolution
	int image_size = grid_size + 2;
	//initialize as 0's first
	thrust::device_vector <Float> image(image_size * image_size * image_size, 0.0); //same as grid but with 0's on boundaries in 3d

	//We now have image. Image is just grid (from argument to this function) with 0s on the boundaries.
	//Now when we convolve, we won't lose any dimensions (our stencil is 3x3x3 so we only needed 1 extra space in each dimension)

	//Create new 3d thrust vector that will eventually be output
	thrust::device_vector <Float> result(grid_size * grid_size * grid_size, 0.0);

	//Call gpu_convolve function to do 3d convolution with mask and image

	//We want a thread for each element in result i.e. grid_size x grid_size x grid_size number of threads

	int threadsPerBlock_dimension = 8; //CANT HAVE MORE THAN 1024 THREADS PER BLOCK
	//allow for dynamically sized grid
	int imageWidth = ceil(Float(image_size) / Float(threadsPerBlock_dimension));
	int imageHeight = ceil(Float(image_size) / Float(threadsPerBlock_dimension));
	int imageDepth = ceil(Float(image_size) / Float(threadsPerBlock_dimension));
	int gridWidth = ceil(Float(grid_size) / Float(threadsPerBlock_dimension));
	int gridHeight = ceil(Float(grid_size) / Float(threadsPerBlock_dimension));
	int gridDepth = ceil(Float(grid_size) / Float(threadsPerBlock_dimension));

	dim3 gridDim1(gridWidth, gridHeight, gridDepth);
	dim3 gridDim2(imageWidth, imageHeight, imageDepth);
	dim3 blockDim(threadsPerBlock_dimension, threadsPerBlock_dimension, threadsPerBlock_dimension);

	gpu_pad << <gridDim2, blockDim >> > (thrust::raw_pointer_cast(grid.data()), thrust::raw_pointer_cast(image.data()), grid_size, image_size);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpu_convolve << < gridDim1, blockDim >> > (thrust::raw_pointer_cast(image.data()), thrust::raw_pointer_cast(mask.data()),
		thrust::raw_pointer_cast(result.data()), image_size, mask_size, grid_size);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	return result;
}

//Inner Product Function
//Just doing this so we can have an easier time reading math
Float paracrine::inner_product(thrust::device_vector <Float> A, thrust::device_vector <Float> B) {

	Float result = thrust::inner_product(A.begin(), A.end(), B.begin(), 0.0f);

	return result;
}



thrust::device_vector <Float> paracrine::update_diffusion(thrust::device_vector<Float> grid) {
	//Computes concentration at gridpoint locations at the next timestep
	//This function includes the conjugate gradient solver
	//Returns a thrust device vector


	//Apply laplace stencil to grid, will have size grid_size x grid_size x grid_size
	laplacian_grid = convolve(grid, stencil);

	//Want to find the following
	//grid_next_step = (1 - dt * decay) * grid + dt * diffusion * laplacian_grid;
	//We do this by first calculating (1-dt*decay), then multiplying it into the grid, 
	//then calculating dt*diffusion, then multiplying it into laplacian_grid,
	//then adding those vectors
	thrust::device_vector<Float> grid_next_step(grid_size * grid_size * grid_size, 0.0);

	//Calculate (1 - timestep * decay_coefficient) * grid 
	//Float dum = 1 - dt * decay;
	thrust::device_vector<Float> dum_array1(grid_size * grid_size * grid_size);
	//thrust::transform(grid.begin(), grid.end(), dum_array1.begin(), dum * _1);

	//Calculate timestep * diffusion_coefficient * laplacian_grid
	//dum = dt * diffusion;
	thrust::device_vector<Float> dum_array2(grid_size * grid_size * grid_size);
	//thrust::transform(laplacian_grid.begin(), laplacian_grid.end(), dum_array1.begin(), dum * _1);

	//Now create A matrix
	Float lambda = diffusion * dt * 0.5;
	//dum = -0.5 * diffusion * dt;
	thrust::transform(stencil.begin(), stencil.end(), A.begin(), -lambda * _1);
	A[3 * 3 * 1 + 3 * 1 + 1] = A[3 * 3 * 1 + 3 * 1 + 1] + 1 + 0.5 * dt * decay; //A(1, 1, 1) = A(1, 1, 1) + 1 + 0.5 * dt * decay;

	//Now create b matrix 
	//b = (1 - dt * decay / 2) * grid + dt * diffusion / 2 * laplacian_grid;
	// b = dum_array1 + dum_array2
	Float dum = 1 - dt * decay / 2;
	thrust::transform(grid.begin(), grid.end(), dum_array1.begin(), dum * _1);

	//dum = dt * diffusion / 2;
	thrust::transform(laplacian_grid.begin(), laplacian_grid.end(), dum_array2.begin(), lambda * _1);

	//Add them together to get b
	thrust::device_vector<Float> b(grid_size * grid_size * grid_size);
	thrust::transform(dum_array1.begin(), dum_array1.end(), dum_array2.begin(), b.begin(), thrust::plus <Float>());

	//NOW EXCECUTES CONJUGATE GRADIENT
	//Determine max iterations and error tolerance for conjugate gradient
	const int max_iterations = 20;
	const Float error_tolerance = 1e-20;
	grid_next_step = CG(A, b, max_iterations, error_tolerance);

	grid = grid_next_step;
	
	return grid;

}


//Conjugate Gradient Function
//TODO: Create CG function in a kernel, not just thrust
//This is currently the slowest part of the code and it's because I didn't create a nice kernel for it yet
	//NOW STARTS CONJUGATE GRADIENT
thrust::device_vector<Float> paracrine::CG(thrust::device_vector<Float> A, thrust::device_vector<Float> b, const int max_iter, const Float error_tolerance) {

	thrust::device_vector<Float> grid_next_step(grid_size * grid_size * grid_size, 0.0);

	//Determine residual
	//r=b-(A * grid_next_step)
	thrust::device_vector<Float> A_grid_next_step(grid_size * grid_size * grid_size);
	thrust::device_vector<Float> r(grid_size * grid_size * grid_size);
	A_grid_next_step = convolve(grid_next_step, A);
	thrust::transform(b.begin(), b.end(), A_grid_next_step.begin(), r.begin(), thrust::minus <Float>());
	//printf("Values in r at a plane are: ");
	//for (int k = 0; k < grid_size; ++k)
	//	printf( "%g\n", (Float)r[grid_size * grid_size * grid_size / 2 + grid_size * k + grid_size / 2]);
	//Set p = r 
	thrust::device_vector<Float> p = r;

	//Find norm of r
//    thrust::device_vector<Float> r_norm(1);
	//Float r_norm = std::sqrt(inner_product(r, r));
	Float r_norm = inner_product(r, r);

	//Check to see if r_norm < error
	if (r_norm < error_tolerance) {
		return grid_next_step;
	}

	thrust::device_vector<Float> Ap(grid_size * grid_size * grid_size,0.0);
	thrust::device_vector<Float> dum_array1(grid_size * grid_size * grid_size,0.0);
	Float pAp;
	Float alpha;
	Float dum;
	
	for (int i = 0; i < max_iter; i++) {

		//Ap
		Ap = convolve(p, A);

		//Determine p*Ap
		//thrust::transform(Ap.begin(), Ap.end(), p.begin(), dum_array1.begin(), thrust::multiplies <Float>());
		//Determine sum(pAp)
		//pAp = thrust::reduce(dum_array1.begin(), dum_array1.end());
		pAp = inner_product(p, Ap);
		//Determine alpha
		alpha = r_norm / pAp;

		//Determine next timestep of grid values: grid_next_step = grid_next_step + alpha * p;
		//Create dum_array1 = alpha*p
		thrust::transform(p.begin(), p.end(), dum_array1.begin(), alpha * _1);
		//grid_next_step=grid_next_step and alpha*p
		thrust::transform(grid_next_step.begin(), grid_next_step.end(), dum_array1.begin(), grid_next_step.begin(), thrust::plus <Float>());

		//Now r = r - alpha * Ap
		//First calculate alpha*Ap
		thrust::transform(Ap.begin(), Ap.end(), dum_array1.begin(), alpha * _1);
		//Now r-alpha*Ap
		thrust::transform(r.begin(), r.end(), dum_array1.begin(), r.begin(), thrust::minus <Float>());

		//Calculate new r_norm
		Float r_norm_new = inner_product(r, r);

		//Check to see if r_norm < error
		if (r_norm_new < error_tolerance) {
			//			std::cout << "Iteration " << i <<"The residual is" << r_norm_new << std::endl;
			return grid_next_step;
		}
		
		//Calculate p = r + (r_norm_new / r_norm) * p;
		dum = r_norm_new / r_norm;
		thrust::transform(p.begin(), p.end(), dum_array1.begin(), dum * _1);
		thrust::transform(r.begin(), r.end(), dum_array1.begin(), p.begin(), thrust::plus <Float>());

		//Reset r_norm
		r_norm = r_norm_new;

	}
	std::cout << "Used maximum iterations for conjugate gradient" << std::endl;
	return grid_next_step;

}



/*
//Diffusion Stepper Function
thrust::device_vector <Float> paracrine::diffusion_stepper(thrust::device_vector <Float> grid) {
	//std::cout << "Diffusion Stepper Started" << std::endl;
	//Returns Grid

	//get b so that we can have Ax=b
	b = convolve(grid, B); //multiplies grid and B. Grid has size 32x32x32 and mask has size 3x3x3
	//this is element wise multiplication and summing as we slide the mask across B

	//We now have Ax=b and we can solve it with whatever linear solver
	//Choose conjugate gradient

	//Calculate Laplacian Grid
	laplacian_grid = convolve(grid, stencil);

	//start conjugate gradient
	int max_iterations = 20000;
	float error_tol = 0.000001;
	//A is calculated in initialization()
	grid = CG(A, b,max_iterations,error_tol);

	//  std::cout << "Diffusion Step DONE" << std::endl;

	return grid;
}


__global__ void gpu_initial_guess(int grid_size, Float diffusion, Float decay, Float dt, Float* grid, Float* laplacian_grid, Float* x) {

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < grid_size * grid_size * grid_size; i += gridDim.x * blockDim.x) {
		x[i] = (1 - dt * decay) * grid[i] + dt * diffusion * laplacian_grid[i];
	}
}

thrust::device_vector <Float> paracrine::initial_guess(thrust::device_vector <Float> grid, thrust::device_vector <Float> laplacian_grid) {
	thrust::device_vector <Float> x(grid_size * grid_size * grid_size);
	int block_size = 512; //schedule a block full of threads
	int cuda_grid_size = ceil((grid_size * grid_size * grid_size) / block_size);

	gpu_initial_guess << < cuda_grid_size, block_size >> > (grid_size, diffusion, decay, dt,
		thrust::raw_pointer_cast(grid.data()), thrust::raw_pointer_cast(laplacian_grid.data()), thrust::raw_pointer_cast(x.data()));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	return x;

}
*/