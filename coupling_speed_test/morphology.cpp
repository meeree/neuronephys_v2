#define _USE_MATH_DEFINES
#include "morphology.h"
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include "H5Cpp.h"
using namespace H5;
using namespace std;

struct Pt
{
    float x, y, z;
};

struct MorphologyContainer
{
    vector<Pt> positions;
    Pt wmin, wmax;
} g_morph;

namespace Morphology
{

std::pair<bool,std::array<float,3>> gen_sphere(int const N)
{
	std::array<float, 3> scaling_rate = { 2.0f,2.0f,2.0f };
    unsigned const seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine gen(seed);
    uniform_real_distribution<float> dist(0, 2 * M_PI);
    for (int i = 0; i < N; ++i)
    {
        float const theta{ dist(gen) }, phi{ dist(gen) };
        g_morph.positions[i] = { cos(theta)*sin(phi), sin(theta)*sin(phi), cos(phi) };
    }
    return std::make_pair(true,scaling_rate);
}


std::pair<bool, std::array<float, 3>> load_cubic_brain_2(int const N, const std::string coupling_file_name)
{
    vector<float> coords[3];
	std::array<float, 3> scaling_rate = { 2.0f,2.0f,2.0f };
    static const char* coord_ids[3]{ "/x", "/y", "/z" };
    static H5std_string const fl_name = "../" + coupling_file_name;
    int numpnts;
    try
    {
        H5File file(fl_name, H5F_ACC_RDONLY);
        for (int i = 0; i < 3; ++i)
        {
            DataSet arr = file.openDataSet(coord_ids[i]);
            numpnts = (int)(arr.getStorageSize() / sizeof(float));
            coords[i].resize(numpnts);
            arr.read(coords[i].data(), PredType::NATIVE_FLOAT);
        }
    }
    catch (...)
    {
        return std::make_pair(false,scaling_rate);
    }
    float wmin[3]{ coords[0][0], coords[1][0], coords[2][0] };
    float wmax[3]{ coords[0][0], coords[1][0], coords[2][0] };
    for (int i = 0; i < 3; ++i)
    {
        for (int c = 0; c < numpnts; ++c)
        {
            if (coords[i][c] < wmin[i])
                wmin[i] = coords[i][c];
            if (coords[i][c] > wmax[i])
                wmax[i] = coords[i][c];
        }
		scaling_rate[i] = 2.0f / (wmax[i] - wmin[i]);
    }
    if (N > numpnts)
    {
        cerr << "Not enough points in file " << fl_name << ". Have N = " << N << endl;
		return std::make_pair(false, scaling_rate);
    }
    int sparsity{ numpnts / N };
    for (int i = 0; i < N; ++i)
    {
        Pt& p{ g_morph.positions[i] };
        int const j{ sparsity * i };
        p = Pt{ coords[0][j], coords[1][j], coords[2][j] };
        p.x = 2.0f * (p.x - wmin[0]) / (wmax[0] - wmin[0]) - 1.0f;
        p.y = 2.0f * (p.y - wmin[1]) / (wmax[1] - wmin[1]) - 1.0f;
        p.z = 2.0f * (p.z - wmin[2]) / (wmax[2] - wmin[2]) - 1.0f;
    }
	return std::make_pair(true, scaling_rate);
}

std::pair<bool,std::array<float,3>> load_cubic_scn(int const N)
{
    vector<float> coords[3];
	std::array<float, 3> scaling_rate = { 2.0f,2.0f,2.0f };
    static const char* coord_ids[3]{ "/x", "/y", "/z" };
    static H5std_string const fl_name = "../scn_atlas.h5";
    int numpnts;
    try
    {
        H5File file(fl_name, H5F_ACC_RDONLY);
        for (int i = 0; i < 3; ++i)
        {
            DataSet arr = file.openDataSet(coord_ids[i]);
            numpnts = (int)(arr.getStorageSize() / sizeof(float));
            coords[i].resize(numpnts);
            arr.read(coords[i].data(), PredType::NATIVE_FLOAT);
        }
    }
    catch (...)
    {
        return std::make_pair(false,scaling_rate);
    }
    float wmin[3]{ coords[0][0], coords[1][0], coords[2][0] };
    float wmax[3]{ coords[0][0], coords[1][0], coords[2][0] };
    for (int i = 0; i < 3; ++i)
    {
        for (int c = 0; c < numpnts; ++c)
        {
            if (coords[i][c] < wmin[i])
                wmin[i] = coords[i][c];
            if (coords[i][c] > wmax[i])
                wmax[i] = coords[i][c];
        }
		scaling_rate[i] = 2.0f / (wmax[i] - wmin[i]);
    }
    if (N > numpnts)
    {
        cerr << "Not enough points in file " << fl_name << ". Have N = " << N << endl;
        return std::make_pair(false,scaling_rate);
    }
    int sparsity{ numpnts / N };
    for (int i = 0; i < N; ++i)
    {
        Pt& p{ g_morph.positions[i] };
        int const j{ sparsity * i };
        p = Pt{ coords[0][j], coords[1][j], coords[2][j] };
        p.x = 2.0f * (p.x - wmin[0]) / (wmax[0] - wmin[0]) - 1.0f;
        p.y = 2.0f * (p.y - wmin[1]) / (wmax[1] - wmin[1]) - 1.0f;
        p.z = 2.0f * (p.z - wmin[2]) / (wmax[2] - wmin[2]) - 1.0f;
    }
    return std::make_pair(true,scaling_rate);
}

std::pair<bool, std::array<float,3>>initialize_positions(MorphologyType const type, int const N, const std::string coupling_filename)
{
    // All morphologies are assumed to be within the range (-1, -1, -1) to (1, 1, 1)
    g_morph.wmin = { -1, -1, -1 };
    g_morph.wmax = { 1, 1, 1 };
    g_morph.positions.resize(N);
	std::array<float, 3> scaling_rate = { 2.0f,2.0f,2.0f };
    switch (type)
    {
    case eBrain2: return load_cubic_brain_2(N,coupling_filename);
    case eSphere: return gen_sphere(N); 
    case eSCN: return load_cubic_scn(N);
	default: cerr << "Unknown MorphologyType: " << type << endl; return std::make_pair(false,scaling_rate);
    }
}

// Accessors
float const* get_positions() { return (float*)g_morph.positions.data(); }
float const* get_world_min() { return (float*)&g_morph.wmin; }
float const* get_world_max() { return (float*)&g_morph.wmax; }
};
