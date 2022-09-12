#include "defines.cuh"
#include <ctime>

std::default_random_engine& RandomEngineFixedSeed(int seed)
{
    // This random engine is seeded with the same value every time.
    // This means that randomness will be THE SAME between runs. 
    static std::default_random_engine gen(seed);
    return gen;
}

std::default_random_engine& RandomEngineDynamicSeed()
{
    // This random engine is seeded with the first time the function is called.
    // This means that randomness will be DIFFERENT between runs. 
    static std::default_random_engine gen(time(NULL));
    return gen;
}

namespace Utility
{
    std::default_random_engine& GetRandomEngine()
    {
#if USE_FIXED_RANDOM_SEED
        return RandomEngineFixedSeed(1);
#else
        return RandomEngineDynamicSeed();
#endif
    }
};
