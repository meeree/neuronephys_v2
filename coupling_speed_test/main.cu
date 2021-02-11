#include "network_factory.cuh"
#include "graphics.h"
#include "morphology.h"

// we need these includes for CUDA's random number stuff
#include <curand.h>
#include <curand_kernel.h>

__global__ void init_random(unsigned int seed, int const N, curandState_t* states) 
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void get_random_test(curandState_t* states, int const N, int* numbers)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        numbers[idx] = curand(&states[idx]) % 100;
    }
}

void UniformTest()
{
    int N = 10;
    int Ncouple = 3;
    thrust::host_vector<int> h_coupling, h_coupling_inds;
    NetworkFactory::UniformNetwork(h_coupling, h_coupling_inds, N, Ncouple);
    AdjacencyList list{ h_coupling, h_coupling_inds };

    thrust::host_vector<int> indices = list.indices;
    thrust::host_vector<int> offsets_in_indices = list.offset_in_indices;

    for (int i = 0; i < offsets_in_indices.size() - 1; ++i)
    {
        std::cout << offsets_in_indices[i + 1] - offsets_in_indices[i] << ": ";
        for (int j = offsets_in_indices[i]; j < offsets_in_indices[i + 1]; ++j)
            std::cout << indices[j] << ',';
        std::cout << std::endl;
    }
}

__global__ void leapfrog(Float** sim_vars, const int N, const Float timestep, curandState_t* random_states)
{
    using namespace hh_params;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        Float* V_ptr = sim_vars[SIM_PARAMS_V] + idx;
        Float* M_ptr = sim_vars[SIM_PARAMS_M] + idx;
        Float* N_ptr = sim_vars[SIM_PARAMS_N] + idx;
        Float* H_ptr = sim_vars[SIM_PARAMS_H] + idx;
        Float* local_Y_ptr = sim_vars[SIM_PARAMS_LOCAL_Y] + idx;
        Float* IY_ptr = sim_vars[SIM_PARAMS_IY] + idx;
        Float* EY_ptr = sim_vars[SIM_PARAMS_EY] + idx;

        const Float local_in = sim_vars[SIM_PARAMS_LOCAL_IN][idx];
        const Float local_num_in = sim_vars[SIM_PARAMS_LOCAL_COUNT][idx];
        const Float Iin = sim_vars[SIM_PARAMS_IIN][idx];
        const Float Inum_in = sim_vars[SIM_PARAMS_ICOUNT][idx];
        const Float Ein = sim_vars[SIM_PARAMS_EIN][idx];
        const Float Enum_in = sim_vars[SIM_PARAMS_ECOUNT][idx];
        const Float appcur = sim_vars[SIM_PARAMS_APPCUR][idx];

        const Float V = *V_ptr;
        const Float M = *M_ptr;
        const Float N = *N_ptr;
        const Float H = *H_ptr;
        const Float local_Y = *local_Y_ptr;
        const Float IY = *IY_ptr;
        const Float EY = *EY_ptr;

        const Float Am = (3.5 + 0.1 * V) / (1 - exp(-3.5 - 0.1*V));
        const Float An = (-0.5 - 0.01 * V) / (exp(-5 - 0.1 * V) - 1);
        const Float Ah = 0.07 * exp(-V / 20 - 3);

        const Float Bm = 4 * exp(-(V + 60) / 18);
        const Float Bn = 0.125*exp(-(V + 60) / 80);
        const Float Bh = 1 / (exp(-3 - 0.1*V) + 1);

        const Float new_M = (Am * timestep + (1 - timestep / 2 * (Am + Bm)) * M) / (timestep / 2 * (Am + Bm) + 1);
        const Float new_N = (An * timestep + (1 - timestep / 2 * (An + Bn)) * N) / (timestep / 2 * (An + Bn) + 1);
        const Float new_H = (Ah * timestep + (1 - timestep / 2 * (Ah + Bh)) * H) / (timestep / 2 * (Ah + Bh) + 1);

        const Float local_Tin_avg = local_num_in > 0 ? local_in / local_num_in : 0;
        const Float new_local_Y = (Ar * local_Tin_avg * timestep + (1 - timestep / 2 * (Ar * local_Tin_avg + Ad)) * local_Y) / (timestep / 2 * (Ar * local_Tin_avg + Ad) + 1);

        const Float Iin_avg = Inum_in > 0 ? Iin / Inum_in : 0;
        const Float new_IY = (Ar * Iin_avg * timestep + (1 - timestep / 2 * (Ar * Iin_avg + Ad)) * IY) / (timestep / 2 * (Ar * Iin_avg + Ad) + 1);

        const Float Ein_avg = Enum_in > 0 ? Ein / Enum_in : 0;
        const Float new_EY = (Ar * Ein_avg * timestep + (1 - timestep / 2 * (Ar * Ein_avg + Ad)) * EY) / (timestep / 2 * (Ar * Ein_avg + Ad) + 1);

        const Float G = gna * new_M*new_M*new_M*new_H + gk * (powf(new_N, 4)) + gl + g_gaba * new_local_Y/* + g_gaba * new_IY + g_ampa * new_EY*/;
        const Float E = gna * new_M*new_M*new_M*new_H * ena + gk * (powf(new_N, 4)) * ek + gl * el + g_gaba * new_local_Y * e_gaba/* + g_gaba * new_IY * e_gaba + g_ampa * new_EY * e_ampa*/;

        // Copy back to sim_vars
        *V_ptr = (appcur * timestep + E * timestep + (1 - timestep / 2 * G)*V) / (1 + timestep / 2 * G);
        *M_ptr = new_M;
        *N_ptr = new_N;
        *H_ptr = new_H;
        *local_Y_ptr = new_local_Y;
        *IY_ptr = new_IY;
        *EY_ptr = new_EY;

        // Calculate amount of neurotransmitter outputted AFTER update
        sim_vars[SIM_PARAMS_LOCAL_OUTPUT][idx] = sim_vars[SIM_PARAMS_IOUTPUT][idx] = sim_vars[SIM_PARAMS_EOUTPUT][idx] = Tmax / (1.0 + exp(-(*V_ptr - Vt) / Kp));
    }
}

#if 0
__device__ void dxdt_hh(const int idx,
    Float const V, Float const M, Float const N, Float const H, Float const Y, Float const Tinavg,
    Float* dvdt, Float* dMdt, Float* dNdt, Float* dHdt, Float* dYdt)
{
    using namespace hh_params;
    // Calculate intermediate quantities
    const Float Am = (3.5 + 0.1 * V) / (1 - exp(-3.5 - 0.1*V));
    const Float An = (-0.5 - 0.01 * V) / (exp(-5 - 0.1 * V) - 1);
    const Float Ah = 0.07 * exp(-V / 20 - 3);

    const Float Bm = 4 * exp(-(V + 60) / 18);
    const Float Bn = 0.125*exp(-(V + 60) / 80);
    const Float Bh = 1 / (exp(-3 - 0.1*V) + 1);

    // Calculate gating variable derivatives
    *dMdt = Am * (1 - M) - Bm * M;
    *dNdt = An * (1 - N) - Bn * N;
    *dHdt = Ah * (1 - H) - Bh * H;
    *dYdt = Ar * Tinavg * (1 - Y) - Ad * Y;

    // Calculate voltage derivative
    *dvdt = appcur
        - gna * M * M * M * H * (V - ena)
        - gk * powf(N, 4) * (V - ek)
        - gl * (V - el)
        - g_coupling * Y * (V - e_coupling_gaba);

    *dvdt /= C;
}

__global__ void RK4_hh(Float** sim_vars, const int N, const float timestep, curandState_t* random_states)
{
    using namespace hh_params;
    Float dvdt1, dMdt1, dNdt1, dHdt1, dYdt1;
    Float dvdt2, dMdt2, dNdt2, dHdt2, dYdt2;
    Float dvdt3, dMdt3, dNdt3, dHdt3, dYdt3;
    Float dvdt4, dMdt4, dNdt4, dHdt4, dYdt4;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        Float* V_ptr = sim_vars[SIM_PARAMS_V] + idx;
        Float* M_ptr = sim_vars[SIM_PARAMS_M] + idx;
        Float* N_ptr = sim_vars[SIM_PARAMS_N] + idx;
        Float* H_ptr = sim_vars[SIM_PARAMS_H] + idx;
        Float* Y_ptr = sim_vars[SIM_PARAMS_Y] + idx;

        Float const V = *V_ptr;
        Float const M = *M_ptr;
        Float const N = *N_ptr;
        Float const H = *H_ptr;
        Float const Y = *Y_ptr;

        // Calculate average neurotransmitter input
        Float Tinavg = sim_vars[SIM_PARAMS_TCOUNT][idx];
        Tinavg = Tinavg == 0 ? 0 : sim_vars[SIM_PARAMS_TIN][idx] / Tinavg;

        // k1 = f(X)
        dxdt_hh(idx, V, M, N, H, Y, Tinavg, &dvdt1, &dMdt1, &dNdt1, &dHdt1, &dYdt1);

        // k2 = f(X + dt/2 * k1)
        dxdt_hh(idx,
            V + dvdt1 * timestep / 2, M + dMdt1 * timestep / 2, N + dNdt1 * timestep / 2, H + dHdt1 * timestep / 2, Y + dYdt1 * timestep / 2, Tinavg,
            &dvdt2, &dMdt2, &dNdt2, &dHdt2, &dYdt2);

        // k3 = f(X + dt/2 * k2)
        dxdt_hh(idx,
            V + dvdt2 * timestep / 2, M + dMdt2 * timestep / 2, N + dNdt2 * timestep / 2, H + dHdt2 * timestep / 2, Y + dYdt2 * timestep / 2, Tinavg,
            &dvdt3, &dMdt3, &dNdt3, &dHdt3, &dYdt3);

        // k4 = f(X + dt * k3)
        dxdt_hh(idx,
            V + dvdt3 * timestep, M + dMdt3 * timestep, N + dNdt3 * timestep, H + dHdt3 * timestep, Y + dYdt3 * timestep, Tinavg,
            &dvdt4, &dMdt4, &dNdt4, &dHdt4, &dYdt4);

        // X(t + dt) = X(t) + dt * (k1 + 2k2 + 2k3 + k4)/6
        *V_ptr = V + timestep * (dvdt1 + dvdt2 * 2 + dvdt3 * 2 + dvdt4) / 6;
        *M_ptr = M + timestep * (dMdt1 + dMdt2 * 2 + dMdt3 * 2 + dMdt4) / 6;
        *N_ptr = N + timestep * (dNdt1 + dNdt2 * 2 + dNdt3 * 2 + dNdt4) / 6;
        *H_ptr = H + timestep * (dHdt1 + dHdt2 * 2 + dHdt3 * 2 + dHdt4) / 6;
        *Y_ptr = Y + timestep * (dYdt1 + dYdt2 * 2 + dYdt3 * 2 + dYdt4) / 6;

        // Calculate amount of neurotransmitter outputted AFTER update
        sim_vars[SIM_PARAMS_OUTPUT][idx] = Tmax / (1.0 + exp(-(*V_ptr - Vt) / Kp));
    }
}
#endif

__device__ void dxdt_an_ie_model(const int idx,
    Float const V, Float const nK, Float const Ca, Float const Ye, Float const Yi,
    Float const e_Tinavg, Float const i_Tinavg,//e/i_Tinavg: excitatory/inhibitory neurotransmitter input
    Float const white_noise,
    Float* dvdt, Float* dnKdt, Float* dCadt, Float* dYedt, Float* dYidt)
{
    using namespace an_ie_params;
    // Calculate intermediate quantities
    Float mCa_inf = 1 / (1 + exp(-(V + 20) / 9));
    Float mKCa_inf = 1 / (1 + powf(kD / Ca, 3.5));
    Float mNaP_inf = 1 / (1 + exp(-(V + 55.7) / 7.7));

    Float alpha_n = V == -34 ? 0.1 : 0.01 * (V + 34) / (1 - exp(-(V + 34) / 10));
    Float beta_n = 0.125 * exp(-(V + 44) / 25);

    // Calculate gating variable derivatives
    *dnKdt = 4 * (alpha_n * (1 - nK) - beta_n * nK);
    *dYedt = 3.48 * e_Tinavg - Ye / tauAMPA;
    *dYidt = i_Tinavg - Yi / tauGABA;

    // Calculate Ca2+ derivative 
    Float ICa = gCa * mCa_inf * mCa_inf * (V - vCa);
    *dCadt = -alphaCa * (10 * A * ICa) - Ca / tauCa;

    // Calculate voltage derivative
    *dvdt = 
        white_noise
        -gK * powf(nK, 4) * (V - vK)
        - ICa
        - gKCa * mKCa_inf * (V - vK)
        - gNaP * mNaP_inf * mNaP_inf * mNaP_inf * (V - vNa)
        - gL * (V - vL)
        - an_ie_params::gAMPA * Ye * (V - an_ie_params::vAMPA)//TODO: Check value
        - an_ie_params::gGABA * Yi * (V - an_ie_params::vGABA);//TODO: Set right value

    *dvdt /= C;
}

__global__ void RK4_an_ie_model(Float** sim_vars, const int N, const float timestep, curandState_t* random_states)
{
    using namespace an_ie_params;
    Float dvdt1, dnKdt1, dCadt1, dYedt1, dYidt1;
    Float dvdt2, dnKdt2, dCadt2, dYedt2, dYidt2;
    Float dvdt3, dnKdt3, dCadt3, dYedt3, dYidt3;
    Float dvdt4, dnKdt4, dCadt4, dYedt4, dYidt4;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        Float* V_ptr = sim_vars[AN_IE_VAR_V] + idx;
        Float* nK_ptr = sim_vars[AN_IE_VAR_NK] + idx;
        Float* Ca_ptr = sim_vars[AN_IE_VAR_CA] + idx;
        Float* Ye_ptr = sim_vars[AN_IE_VAR_YE] + idx;
        Float* Yi_ptr = sim_vars[AN_IE_VAR_YI] + idx;

        // Calculate average neurotransmitter input
        Float e_Tinavg = sim_vars[AN_IE_VAR_ECOUNT][idx];
        e_Tinavg = e_Tinavg == 0 ? 0 : sim_vars[AN_IE_VAR_EIN][idx] / e_Tinavg;
        Float i_Tinavg = sim_vars[AN_IE_VAR_ICOUNT][idx];
        i_Tinavg = i_Tinavg == 0 ? 0 : sim_vars[AN_IE_VAR_IIN][idx] / i_Tinavg;

        Float const V = *V_ptr;
        Float const nK = *nK_ptr;
        Float const Ca = *Ca_ptr;
        Float const Ye = *Ye_ptr;
        Float const Yi = *Yi_ptr;

        // Add random white noise to V.
        Float white_noise = white_noise_intensity * curand_normal(random_states + idx) + white_noise_mean;

        // k1 = f(X)
        dxdt_an_ie_model(idx, V, nK, Ca, Ye, Yi, e_Tinavg, i_Tinavg, white_noise, &dvdt1, &dnKdt1, &dCadt1, &dYedt1, &dYidt1);

        // k2 = f(X + dt/2 * k1)
        dxdt_an_ie_model(idx,
            V + dvdt1 * timestep / 2, nK + dnKdt1 * timestep / 2, Ca + dCadt1 * timestep / 2, Ye + dYedt1 * timestep / 2, Yi + dYidt1 * timestep / 2, white_noise, 
            e_Tinavg, i_Tinavg, &dvdt2, &dnKdt2, &dCadt2, &dYedt2, &dYidt2);

        // k3 = f(X + dt/2 * k2)
        dxdt_an_ie_model(idx,
            V + dvdt2 * timestep / 2, nK + dnKdt2 * timestep / 2, Ca + dCadt2 * timestep / 2, Ye + dYedt2 * timestep / 2, Yi + dYidt2 * timestep / 2,  white_noise, 
            e_Tinavg, i_Tinavg, &dvdt3, &dnKdt3, &dCadt3, &dYedt3, &dYidt3);

        // k4 = f(X + dt * k3)
        dxdt_an_ie_model(idx,
            V + dvdt3 * timestep, nK + dnKdt3 * timestep, Ca + dCadt3 * timestep, Ye + dYedt3 * timestep, Yi + dYidt3 * timestep / 2, white_noise, 
            e_Tinavg, i_Tinavg, &dvdt4, &dnKdt4, &dCadt4, &dYedt4, &dYidt4);

        // X(t + dt) = X(t) + dt * (k1 + 2k2 + 2k3 + k4)/6
        *V_ptr = V + timestep * (dvdt1 + dvdt2 * 2 + dvdt3 * 2 + dvdt4) / 6;
        *nK_ptr = nK + timestep * (dnKdt1 + dnKdt2 * 2 + dnKdt3 * 2 + dnKdt4) / 6;
        *Ca_ptr = Ca + timestep * (dCadt1 + dCadt2 * 2 + dCadt3 * 2 + dCadt4) / 6;
        *Ye_ptr = Ye + timestep * (dYedt1 + dYedt2 * 2 + dYedt3 * 2 + dYedt4) / 6;
        *Yi_ptr = Yi + timestep * (dYidt1 + dYidt2 * 2 + dYidt3 * 2 + dYidt4) / 6;

        // Calculate amount of neurotransmitter outputted AFTER update
        sim_vars[AN_IE_VAR_IOUTPUT][idx] = sim_vars[AN_IE_VAR_EOUTPUT][idx] = Tmax / (1.0 + exp(-(*V_ptr - an_Vt) / an_Kp));
    }
}

__global__ void Euler_an_ie_model(Float** sim_vars, const int N, const float timestep, curandState_t* random_states)
{
    using namespace an_ie_params;
    Float dvdt, dnKdt, dCadt, dYedt, dYidt;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        Float* V_ptr = sim_vars[AN_IE_VAR_V] + idx;
        Float* nK_ptr = sim_vars[AN_IE_VAR_NK] + idx;
        Float* Ca_ptr = sim_vars[AN_IE_VAR_CA] + idx;
        Float* Ye_ptr = sim_vars[AN_IE_VAR_YE] + idx;
        Float* Yi_ptr = sim_vars[AN_IE_VAR_YI] + idx;

        // Calculate average neurotransmitter input
        Float e_Tinavg = sim_vars[AN_IE_VAR_ECOUNT][idx];
        e_Tinavg = e_Tinavg == 0 ? 0 : sim_vars[AN_IE_VAR_EIN][idx] / e_Tinavg;
        Float i_Tinavg = sim_vars[AN_IE_VAR_ICOUNT][idx];
        i_Tinavg = i_Tinavg == 0 ? 0 : sim_vars[AN_IE_VAR_IIN][idx] / i_Tinavg;

        Float const V = *V_ptr;
        Float const nK = *nK_ptr;
        Float const Ca = *Ca_ptr;
        Float const Ye = *Ye_ptr;
        Float const Yi = *Yi_ptr;

        // Add random white noise to V.
        Float white_noise = white_noise_intensity * curand_normal(random_states + idx) + white_noise_mean;

        dxdt_an_ie_model(idx, V, nK, Ca, Ye, Yi, e_Tinavg, i_Tinavg, white_noise, &dvdt, &dnKdt, &dCadt, &dYedt, &dYidt);
        
        // X(t + dt) = X(t) + dt * dxdt
        *V_ptr = V + timestep * dvdt;
        *nK_ptr = nK + timestep * dnKdt;
        *Ca_ptr = Ca + timestep * dCadt;
        *Ye_ptr = Ye + timestep * dYedt;
        *Yi_ptr = Yi + timestep * dYidt;

        // Calculate amount of neurotransmitter outputted AFTER update
        sim_vars[AN_IE_VAR_IOUTPUT][idx] = sim_vars[AN_IE_VAR_EOUTPUT][idx] = Tmax / (1.0 + exp(-(*V_ptr - an_Vt) / an_Kp));
    }
}

// Internal function. Sets up network connectivity.
AdjacencyList __GenerateConnectivity(int const N, Float*& Tcount, const std::string connectivity_filename, bool local=false)
{
    // Generate coupling
    AdjacencyList list;
    {
        thrust::host_vector<int> h_coupling, h_coupling_inds;
        if(local)
            NetworkFactory::LocalConnectivity(h_coupling, h_coupling_inds, N, 50, Morphology::get_positions());
        else
            NetworkFactory::BrainConnectivity(h_coupling, h_coupling_inds, N, connectivity_filename);
        list.indices = h_coupling;
        list.offset_in_indices = h_coupling_inds;
    }

    // Calculate numer of upstream neurons for each neuron based on synaptic coupling
    thrust::host_vector<Float> Tcount_cpu(N, 0);
    {
        thrust::host_vector<int> indices_cpu(list.indices);
        for (auto ind : indices_cpu)
        {
            if (ind >= N)
                throw std::runtime_error("ERROR in __GenerateConnectivity: Index " + std::to_string(ind) + " out of range.");
            ++Tcount_cpu[ind];
        }
        cudaMemcpy(Tcount, Tcount_cpu.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
    }
    std::cout << "Setup input connectivity" << std::endl;
    return list;
}

ChunkedAdjacencyList __GenerateConnectivityChunked(int const N, Float*& Tcount, const std::string connectivity_filename, bool local=false)
{
    // Generate coupling
    ChunkedAdjacencyList list;
    thrust::host_vector<Float> Tcount_cpu(N, 0);
    {
        thrust::host_vector<int> h_coupling, h_coupling_inds;
        if(local)
            NetworkFactory::LocalConnectivity(h_coupling, h_coupling_inds, N, 50, Morphology::get_positions());
        else
            NetworkFactory::BrainConnectivity(h_coupling, h_coupling_inds, N, connectivity_filename);

        // Calculate numer of upstream neurons for each neuron based on synaptic coupling
        thrust::host_vector<int> indices_cpu(h_coupling);
        for (auto ind : indices_cpu)
        {
            if (ind >= N)
                throw std::runtime_error("ERROR in __GenerateConnectivity: Index " + std::to_string(ind) + " out of range.");
            ++Tcount_cpu[ind];
        }
        cudaMemcpy(Tcount, Tcount_cpu.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        thrust::host_vector<thrust::host_vector<int>> h_coupling_chunked(NCHUNKS), h_coupling_inds_chunked(NCHUNKS, h_coupling_inds);
        for (int i = 1; i < N+1; ++i)
        {
            for(auto& inds_chunk: h_coupling_inds_chunked)
                inds_chunk[i] = inds_chunk[i - 1];

            for (int j = h_coupling_inds[i - 1]; j < h_coupling_inds[i]; ++j)
            {
                int const ds_idx = h_coupling[j];
                int const ds_chunk = ds_idx / N_PER_CHUNK;
                h_coupling_chunked[ds_chunk].push_back(ds_idx % N_PER_CHUNK);
                h_coupling_inds_chunked[ds_chunk][i] += 1;
            }
        }

        thrust::host_vector<int*> raw_h_coupling_chunked(NCHUNKS), raw_h_coupling_inds_chunked(NCHUNKS);
        for (int i = 0; i < NCHUNKS; ++i)
        {
            int sz = h_coupling_chunked[i].size() * sizeof(int);
            cudaMalloc(&(raw_h_coupling_chunked[i]), sz);
            cudaMemcpy(raw_h_coupling_chunked[i], h_coupling_chunked[i].data(), sz, cudaMemcpyHostToDevice);

            sz = h_coupling_inds_chunked[i].size() * sizeof(int);
            cudaMalloc(&(raw_h_coupling_inds_chunked[i]), sz);
            cudaMemcpy(raw_h_coupling_inds_chunked[i], h_coupling_inds_chunked[i].data(), sz, cudaMemcpyHostToDevice);
        }
        list.indices_chunks = raw_h_coupling_chunked;
        list.offset_in_indices_chunks = raw_h_coupling_inds_chunked;
    }
    std::cout << "Setup input connectivity" << std::endl;
    return list;
}

// This is simply a struct that contains all the relevant stuff for coupling. 
// It can be called which will perform coupling and store it. 
// Description:
// - list Adjacency list containing connectivity.
// - coupling Coupling class that can be called to do coupling with a variety of connectivity structs.
// - downstream_inputs Input to downstream neurons after coupling.
// - neuron_outputs Output of neurons.
struct CouplingStruct 
{
    ChunkedAdjacencyList& list;
    AtomicCoupling& coupling;
    Float* downstream_inputs;
    Float* neuron_outputs;
    
    // Performs coupling and stores results in downstream_inputs. 
    // Returns number of neurons firing.
    int operator() () 
    {
        return coupling(list, downstream_inputs, neuron_outputs);
    }
};

#if 0
std::pair<float, float> run_timing(int const N, int const Ncouple)
{
    // Set up simulation variables. Stored in a struct of arrays.
    std::ofstream coupling_timing_fl("COUPLING_TIMING_" + std::to_string(N) + "_" + std::to_string(Ncouple) + ".txt");
    std::ofstream num_firing_fl("NUM_FIRING_" + std::to_string(N) + "_" + std::to_string(Ncouple) + ".txt");
    std::ofstream sample_fl("SAMPLE_FILE.txt");
    thrust::host_vector<Float*> sim_vars_cpu(NUM_SIM_PARAMS);
    thrust::device_vector<Float*> sim_vars;
    {
        thrust::host_vector<Float> zeroed_arr_cpu(N, 0);
        for (Float*& arr : sim_vars_cpu)
        {
            cudaMalloc(&arr, N * sizeof(Float));
            cudaMemcpy(arr, zeroed_arr_cpu.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        }

        // Set H gating var to 1 since it is a deactive var
        thrust::fill_n(zeroed_arr_cpu.begin(), N, 1);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_H], zeroed_arr_cpu.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        // Set applied currents to 9
        thrust::fill_n(zeroed_arr_cpu.begin(), N, 9);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_APPCUR], zeroed_arr_cpu.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
    }
    // Copy pointers onto device
    sim_vars = sim_vars_cpu;

    // Generate coupling
    AtomicCoupling coupling(N);
    AdjacencyList list{ __GenerateConnectivity(N, sim_vars_cpu[SIM_PARAMS_TCOUNT], "cortex_connectivity_1M_1k.h5") };

    std::cout << "Setup input connectivity" << std::endl;

    Float* V_ptr = sim_vars[SIM_PARAMS_V];
    Float* neuron_outputs = sim_vars[SIM_PARAMS_OUTPUT];
    Float* spikes = sim_vars[SIM_PARAMS_TIN];
    Float* spike_counts = sim_vars[SIM_PARAMS_TCOUNT];

#if 0
    std::cout << "Running transient simulation setup " << std::endl;
    for (int i = 0; i < 0; ++i)
    {
        if (i % 100 == 0)
            std::cout << i << std::endl;
        // Update ODE 
        leapfrog << <NBLOCKS, NTHREADS >> > (thrust::raw_pointer_cast(sim_vars.data()), N, 0.4);

        // Perform coupling
        coupling(connectivity, V_ptr, spikes, neuron_outputs);
    }
#endif 

    float time1 = 0, time2 = 0;
    float time_inc;

    // Use Gaussian white noise with variance proportional to dt.
    Float const sqrt_dt = sqrt(0.4);
    Float const noise_intensity = 0;
    Float const ambient_noise = 8;
    thrust::host_vector<Float> appcur(N, 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> appcur_dist(ambient_noise, sqrt_dt * noise_intensity);
    for (int i = 0; i < 2500; ++i)
    {
#if 1  // Output some sample values. Useful to check the simulation is correct.
        int smpl0 = 0, smpl1 = 1025, smpl2 = N - 1;
        Float V0, V1, V2;
        cudaMemcpy(&V0, V_ptr + smpl0, sizeof(Float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&V1, V_ptr + smpl1, sizeof(Float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&V2, V_ptr + smpl2, sizeof(Float), cudaMemcpyDeviceToHost);
        sample_fl << V0 << ',' << V1 << ',' << V2 << std::endl;
#endif 

        for (auto& val : appcur)
            val = appcur_dist(gen);
        cudaMemcpy(sim_vars[SIM_PARAMS_APPCUR], appcur.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        // Update ODE 
        leapfrog << <NBLOCKS, NTHREADS >> > (thrust::raw_pointer_cast(sim_vars.data()), N, 0.4);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_inc, start, stop);
        time1 += time_inc;

        // Perform coupling
        cudaEvent_t start2, stop2;
        cudaEventCreate(&start2);
        cudaEventCreate(&stop2);
        cudaEventRecord(start2, 0);
        int num_firing = coupling(list, spikes, neuron_outputs);
        cudaDeviceSynchronize();
        cudaEventRecord(stop2, 0);
        cudaEventSynchronize(stop2);
        cudaEventElapsedTime(&time_inc, start2, stop2);
        time2 += time_inc;

        // Write to output files
        coupling_timing_fl << time_inc << '\n';
        num_firing_fl << num_firing << '\n';
    }
    coupling_timing_fl.close();
    num_firing_fl.close();
    sample_fl.close();
    return { time1 / 1000, time2 / 1000 };
}

void run_benchmark()
{
    std::vector<int> sample_Ncouple{ 200 };
    std::vector<int> sample_N{ 1000000 };
    std::ofstream out("OUTPUT.txt");
    for (int Ncouple : sample_Ncouple)
    {
        for (int N : sample_N)
        {
            if (Ncouple > N)
                Ncouple = N; // For small N, use 100% coupling

            // 2 billion synapses is the max since this results in ~8gb mem usage. 
            if (N * Ncouple >= 2e9)
                continue; // skip

            try
            {
                std::pair<float, float> time = run_timing(N, Ncouple);
                out << N << ',' << Ncouple << ',' << time.first << ',' << time.second << ',' << time.first + time.second << std::endl;
                std::cout << N << ',' << Ncouple << ',' << time.first << ',' << time.second << ',' << time.first + time.second << std::endl;
            }
            catch (std::exception& e)
            {
                out << N << ',' << Ncouple << ",NaN,NaN,NaN" << std::endl;
                std::cout << N << ',' << Ncouple << ",NaN,NaN,NaN" << std::endl;
            }
        }
    }
    out.close();
}
#endif

std::string SampleValsToStr(int const smpl, std::vector<Float*> const& ptrs)
{
    Float val;
    std::string out;
    for (Float* ptr : ptrs)
    {
        cudaMemcpy(&val, ptr + smpl, sizeof(Float), cudaMemcpyDeviceToHost);
        out += std::to_string(val) + ",";
    }
    out += "   ";
    return out;
}

template<typename SimFunc>
void _RunAndPrintSimulation(int const samp_freq, int const samp_len, std::ostream& fout,
    int const N, thrust::device_vector<Float*>& sim_vars,
    SimFunc sim_func, 
    std::vector<CouplingStruct>& coupling_structs,
    Float* V_ptr,
    thrust::device_vector<curandState_t>& random_states,
    std::vector<Float*> const& sample_ptrs,
    float* graphics_voltages_handle = nullptr, int const render_N = -1)
{
    // Note, we multiply by 1000 to convert from sec to ms.
    int const nincs = (1000 * samp_len / an_ie_params::rk4_dt);
    int const print_incs = (int)((1000.0 / samp_freq) / an_ie_params::rk4_dt + 0.5);

    // If graphics handle is non-null, allocate CPU mem to copy voltages to.
    Float* cpu_voltages;
    uint8_t* frame_data;
    FILE* replay_out;
    if (graphics_voltages_handle)
    {
        cpu_voltages = (Float*)malloc(sizeof(Float) * N);
        frame_data = (uint8_t*)malloc(3 * sizeof(uint8_t) * Graphics::get_width() * Graphics::get_height());
        replay_out = fopen("REPLAY.raw", "wb");
    }

    for (int i = 0; i < nincs; ++i)
    {
        if (i % (nincs / 100) == 0)
            std::cout << 100 * (i / (float)nincs) << "%" << std::endl;

        if (i % print_incs == 0)
        {
//            fout << SampleValsToStr(0, sample_ptrs) << SampleValsToStr(1, sample_ptrs) << SampleValsToStr(2, sample_ptrs) << std::endl;
            cudaMemcpy(cpu_voltages, V_ptr, N * sizeof(Float), cudaMemcpyDeviceToHost);
            int knn[NSAMPLES_PRINT]{0,967185,961357,975142,526318,60760,82355,889193,282324,132016,967680,140256,655715,699914,571249,239471,901947,293158,983985,755743,961660,449918,437360,627379,29632,680144,428995,215054,920443,497422,589815,705522,987485,877772,415620,673572,642659,207738,469096,944894,346830,431706,829327,661825,856795,84950,344529,196000,637175,653581,278265,692725,384015,294834,607171,570985,798159,984694,517696,853142,801988,173291,946596,67240,433000,47727,139214,764181,869719,747938,659871,461688,626386,323483,978204,170081,33011,297149,161269,49778,833242,947781,477872,816586,459457,302955,976979,752175,281266,482258,468056,252779,481624,306572,69381,512634,878731,770683,210172,754446,245179,775737,505566,273374,631691,257039,866962,943383,781515,302851,274716,654170,774270,186777,579090,961701,248403,391131,798679,920072,836068,725939,139961,725711,68872,208176,554174,451865,850095,928513,71331,184185,206851,963520,573202,422621,276440,431509,656213,574629,348116,340017,401641,178121,472683,431773,988701,316288,448838,324110,50348,252161,900363,371728,202627,420531,707788,33432,689220,507782,377508,107303,533828,244536,708091,434632,242843,186242,802645,888756,422591,375054,708090,290710,473553,278169,742863,551584,562769,650165,154231,263174,651342,187576,328880,215595,826343,935886,183387,526888,991855,225328,323132,608489,887376,28236,269977,609839,933220,465649,601520,314942,512062,943806,715931,492248,783693,798671,215384,780301,908531,588077,863234,681511,397050,582488,533481,666441,45380,743624,43468,161719,666957,533942,709263,662386,550030,809869,91080,392005,910547,117060,673043,720240,293677,564055,589958,90427,486179,293069,426522,378106,351327,305623,885385,630214,870897,835287,422143,940460,144863,890594,751034,650683,726834,321292,559376,548866,383831,429849,979435,864547,256001,271901,80066,303600,753288,939501,496765,274702,109476,324612,446988,944277,92655,813987,745160,639754,568133,316620,921455,852744,761448,538212,889081,181322,774844,438699,882910,825952,892970,561245,303096,49105,431972,269526,570834,920537,83053,357971,548059,76702,546377,639199,48515,537343,227433,994022,256743,708243,831805,800631,186153,328407,475406,825812,483786,605903,772703,9133,627279,765917,570353,854218,969682,85903,454470,810733,798181,585285,54784,904765,5545,771511,569866,553573,194211,818803,923613,818750,931593,77719,418810,44761,877443,784262,665775,501328,638257,501456,973938,216234,367553,157029,345303,718477,918721,112831,695753,108911,259227,350743,551548,489462,523224,773820,305604,312870,818408,621835,221604,804503,983347,638845,290967,951122,187620,435227,221674,127157,532711,131785,769775,961772,423529,597319,986476,148765,251611,68233,341725,941188,397586,915420,100636,895764,953846,328385,509381,998085,448631,29592,12218,34619,702110,637220,597893,100955,834199,126684,561530,79852,394475,190459,733710,450115,465709,496841,720446,35182,469826,653679,299793,542261,434817,840478,232963,590503,49355,147137,838593,661034,417028,177005,539474,136283,590339,912605,218710,26582,511780,724118,34142,907726,927594,54342,35350,300308,172848,336307,428467,568229,535989,531744,810954,258186,727726,370797,586138,773356,600920,849801,970885,312443,836098,53396,337923,863733,657272,965654,12523,934265,630611,21664,519325,397551,416701,171088,577614,505270,442476,340786,457422,302390,957560,748441,255992,454773,800911,437071,51977,260443,69196,155072,777072,510722,273234,976306,391292,563340,789556,};
            for(int j = 0; j < NSAMPLES_PRINT; ++j)
                fout << cpu_voltages[j] << ',';
            fout << std::endl;
            // Update graphics if the graphics handle is non-null
            if (graphics_voltages_handle)
            {
                // TODO: PUT SCALING IN SHADER IN GRAPHICS CODE
                for (int k = 0; k < render_N; ++k)
                    graphics_voltages_handle[k] = 1 / (1 + exp(-(cpu_voltages[k * (N / render_N)] - hh_params::Vt) / hh_params::Kp));

                bool pause{ false };
                Graphics::render(graphics_voltages_handle, nullptr, 0, 0, pause);
                Graphics::write_frame(replay_out, frame_data);
            }
        }

        // Perform coupling. The number of neurons firing is returned by each of these calls. 
        for(auto& coup_struct: coupling_structs)
            coup_struct();

        // Update ODE 
        sim_func << <32, 64>> > (thrust::raw_pointer_cast(sim_vars.data()), N, an_ie_params::rk4_dt, thrust::raw_pointer_cast(random_states.data()));
    }
    fout << SampleValsToStr(0, sample_ptrs) << SampleValsToStr(1, sample_ptrs) << SampleValsToStr(2, sample_ptrs) << std::endl;

    if (graphics_voltages_handle)
    {
        free(cpu_voltages);
        free(frame_data);
        fclose(replay_out);
    }
}

// N : number of neurons to simulate
// samp_freq : sampling frequency (Hz)
// samp_len : How long you record the activity of the model neuron (sec)
void sanmodel_inhexc_benchmark(int const N = 1, int const samp_freq = 1000, int const samp_len = 10, int render_N = -1)
{
    // Set up graphics and morphology
    Morphology::initialize_positions(MorphologyType::eBrain2, N, "cortex_connectivity_1m_1k_inh.h5");

    float* graphics_voltage_handle;
    Graphics::setup(graphics_voltage_handle, Morphology::get_positions(), N, render_N);

    // Set up simulation variables. Stored in a struct of arrays.
    thrust::host_vector<Float*> sim_vars_cpu(NUM_AN_IE_VAR);
    thrust::device_vector<Float*> sim_vars;
    std::ofstream fout("OUT_AN.txt");
    {
        thrust::host_vector<Float> cpu_arr(N, 0);

        for (Float*& arr : sim_vars_cpu)
            cudaMalloc(&arr, N * sizeof(Float));

        // Set voltages randomly
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> voltage_dist(-80, 20);
        for (auto& val : cpu_arr)
            val = voltage_dist(gen);
        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_V], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        // Set Ca 
        std::uniform_real_distribution<> Ca_dist(0.1, 0.5);
        for (auto& val : cpu_arr)
            val = Ca_dist(gen);
        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_CA], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        // Set potasium gating variable 
        std::uniform_real_distribution<> nK_dist(0.0, 0.5);
        for (auto& val : cpu_arr)
            val = nK_dist(gen);
        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_NK], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        thrust::fill_n(cpu_arr.begin(), N, 0.0);
        // AMPA/NMDA gating variable
        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_YE], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        // GABA gating variable

        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_YI], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        // No neurotransmitter input/output. Connection counts are set to zero for now and edited below. 
        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_EOUTPUT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_EIN], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_ECOUNT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_IOUTPUT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_IIN], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_ICOUNT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
    }
    // Copy pointers from CPU to GPU 
    sim_vars = sim_vars_cpu;

    // Random seeds setup. We get a "state" per neuron which we you can think of as the random seed.
    thrust::device_vector<curandState_t> random_states(N);
    init_random << <NBLOCKS, NTHREADS >> > (0, N, thrust::raw_pointer_cast(random_states.data()));

    // Generate coupling //TODO: Add additional coupling
    AtomicCoupling e_coupling(N);
    AtomicCoupling i_coupling(N);
    ChunkedAdjacencyList e_list{ __GenerateConnectivityChunked(N, sim_vars_cpu[AN_IE_VAR_ECOUNT],"cortex_connectivity_1m_1k_exc.h5") };
    ChunkedAdjacencyList i_list{ __GenerateConnectivityChunked(N, sim_vars_cpu[AN_IE_VAR_ICOUNT],"cortex_connectivity_1m_1k_inh.h5") };
    std::vector<CouplingStruct> coupling_structs
    {
        {e_list, e_coupling, sim_vars_cpu[AN_IE_VAR_EIN], sim_vars_cpu[AN_IE_VAR_EOUTPUT]},
        {i_list, i_coupling, sim_vars_cpu[AN_IE_VAR_IIN], sim_vars_cpu[AN_IE_VAR_IOUTPUT]},
    };


    // Get pointers to data we want to write output for
    std::vector<Float*> sample_ptrs{
        sim_vars_cpu[AN_IE_VAR_V],
        sim_vars_cpu[AN_IE_VAR_CA],
        sim_vars_cpu[AN_IE_VAR_NK],
        sim_vars_cpu[AN_IE_VAR_EOUTPUT],
        sim_vars_cpu[AN_IE_VAR_EIN],
        sim_vars_cpu[AN_IE_VAR_ECOUNT],
        sim_vars_cpu[AN_IE_VAR_IOUTPUT],
        sim_vars_cpu[AN_IE_VAR_IIN],
        sim_vars_cpu[AN_IE_VAR_ICOUNT],
    };
    Float* V_ptr{ sim_vars_cpu[AN_IE_VAR_V] };
    _RunAndPrintSimulation(samp_freq, samp_len, fout,
        N, sim_vars,
        Euler_an_ie_model, 
        coupling_structs,
        sim_vars_cpu[AN_IE_VAR_V],
        random_states,
        sample_ptrs,
        graphics_voltage_handle,
        render_N);

    for (Float*& arr : sim_vars_cpu)
        cudaFree(arr);
    fout.close();
}

// N : number of neurons to simulate
// samp_freq : sampling frequency (Hz)
// samp_len : How long you record the activity of the model neuron (sec)
void hh_benchmark(int const N = 1, int const samp_freq = 1000, int const samp_len = 10, int render_N = -1)
{
    using namespace hh_params;

    // Set up graphics and morphology
    Morphology::initialize_positions(MorphologyType::eBrain2, N, "cortex_connectivity_1m_1k_inh.h5");

    float* graphics_voltage_handle;
    Graphics::setup(graphics_voltage_handle, Morphology::get_positions(), N, render_N);

    // Set up simulation variables. Stored in a struct of arrays.
    thrust::host_vector<Float*> sim_vars_cpu(NUM_SIM_PARAMS);
    thrust::device_vector<Float*> sim_vars;
    std::ofstream fout("OUT_HH.txt");
    {
        thrust::host_vector<Float> cpu_arr(N, 0);

        for (Float*& arr : sim_vars_cpu)
            cudaMalloc(&arr, N * sizeof(Float));

        // Set voltages randomly
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> voltage_dist(-65, 40);
        for (auto& val : cpu_arr)
            val = voltage_dist(gen);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_V], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        // Set gating variables  
        std::normal_distribution<> gating_var_dist(0.5, 0.5);
        for (auto& val : cpu_arr)
            val = gating_var_dist(gen);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_M], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        for (auto& val : cpu_arr)
            val = gating_var_dist(gen);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_N], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        for (auto& val : cpu_arr)
            val = gating_var_dist(gen);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_H], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        thrust::fill_n(cpu_arr.begin(), N, 0.0);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_LOCAL_Y], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_IY], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_EY], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        // No neurotransmitter input/output or connections
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_LOCAL_OUTPUT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_LOCAL_IN], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_LOCAL_COUNT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_IOUTPUT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_IIN], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_ICOUNT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_EOUTPUT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_EIN], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_ECOUNT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        // Set applied current
        thrust::fill_n(cpu_arr.begin(), N, appcur);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_APPCUR], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
    }
    // Copy pointers from CPU to GPU 
    sim_vars = sim_vars_cpu;

    // Generate coupling
    AtomicCoupling loc_coupling(N), e_coupling(N), i_coupling(N);
    ChunkedAdjacencyList loc_list{ __GenerateConnectivityChunked(N, sim_vars_cpu[SIM_PARAMS_LOCAL_COUNT], "", true) };
    ChunkedAdjacencyList e_list{ __GenerateConnectivityChunked(N, sim_vars_cpu[SIM_PARAMS_ECOUNT],"cortex_connectivity_1m_1k_exc.h5") };
    ChunkedAdjacencyList i_list{ __GenerateConnectivityChunked(N, sim_vars_cpu[SIM_PARAMS_ICOUNT],"cortex_connectivity_1m_1k_inh.h5") };

    std::vector<CouplingStruct> coupling_structs
    {
        {loc_list, loc_coupling, sim_vars_cpu[SIM_PARAMS_LOCAL_IN], sim_vars_cpu[SIM_PARAMS_LOCAL_OUTPUT]},
        {e_list, e_coupling, sim_vars_cpu[SIM_PARAMS_EIN], sim_vars_cpu[SIM_PARAMS_EOUTPUT]},
        {i_list, i_coupling, sim_vars_cpu[SIM_PARAMS_IIN], sim_vars_cpu[SIM_PARAMS_IOUTPUT]}
    };
    
    //TODO : ADD RANDOM WHITE NOISE TO HH
    // Random seeds setup. We get a "state" per neuron which we you can think of as the random seed.
    thrust::device_vector<curandState_t> random_states(N);
    init_random << <NBLOCKS, NTHREADS >> > (0, N, thrust::raw_pointer_cast(random_states.data()));


    // Get pointers to data we want to write output for
    std::vector<Float*> sample_ptrs{
        sim_vars_cpu[SIM_PARAMS_V],
    };
    _RunAndPrintSimulation(samp_freq, samp_len, fout,
        N, sim_vars,
        leapfrog, 
        coupling_structs,
        sim_vars_cpu[SIM_PARAMS_V],
        random_states,
        sample_ptrs,
        graphics_voltage_handle,
        render_N);

    for (Float*& arr : sim_vars_cpu)
        cudaFree(arr);
    fout.close();
}

int main()
{
    hh_benchmark(1e6, 20000, 10, 1e5);
    Graphics::terminate_graphics();
}
