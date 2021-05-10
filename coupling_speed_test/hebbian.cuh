#include "coupling.cuh"

class Hebbian
{
private:
    int Nconns;
    thrust::device_vector<int> R, R0, R1, D;
public:
    int const window_duration;
    Float const upscale;
    Float const downscale;

    //! Construct Hebbian learner.
    //! \param [in] window_duration_ The window size for Hebbian learning rule in dt timestep units.
    //! \param [in] upscale_ Constant to multiply synaptic weights by for increasing weight (1 <= upscale). 
    //! \param [in] downscale_ Constant to multiply synaptic weights by for decreasing weight (0 <= downscale <= 1).
    //! \param [in] Nconns_ Number of synapses in the whole network.
    Hebbian (int window_duration_, Float upscale_, Float downscale_, int Nconns_)
        : Nconns{ Nconns_ }, R(Nconns), R0(Nconns), R1(Nconns), D(Nconns),
        window_duration{ window_duration_ }, upscale{ upscale_ }, downscale{ downscale_ }
    {}

    //! Do Hebbian learning at a certain instant in time.
    //! \param [in] connectivity The synaptic connectivity stored as an adjacency list of downstream connections.
    //! \param [in] firing Vector where element i is 1/0 if neuron i is/isn't firing.
    void UpdateWeights(AdjacencyList const& connectivity, thrust::device_vector<int> const& firing);
};