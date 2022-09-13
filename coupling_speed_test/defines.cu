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

    std::string GetTimestampString()
    {
        time_t rawtime;
        struct tm* timeinfo;
        char buffer[80];
        time(&rawtime);
        timeinfo = localtime(&rawtime);

        // Day_month_year_hour_minute_second format.
        strftime(buffer, sizeof(buffer), "%d_%m_%Y_%H_%M_%S", timeinfo);
        return std::string(buffer);
    }

    bool WriteDataToHDF(std::string file_name, Float const* data, std::vector<hsize_t> dims, bool prepend_timestamp)
    {
        using namespace H5;
        if (prepend_timestamp)
            file_name = GetTimestampString() + "_" + file_name;

        try
        {
            H5File file(file_name, H5F_ACC_TRUNC);
            DataSpace dataspace(dims.size(), dims.data());
            
            auto pred_type = sizeof(Float) == sizeof(float) ? PredType::NATIVE_FLOAT : PredType::NATIVE_DOUBLE;
            FloatType datatype(pred_type);

            DataSet dataset = file.createDataSet("data", datatype, dataspace);
            dataset.write(data, pred_type);
        } 
        catch (...)
        {
            return false;
        }
    }
};
