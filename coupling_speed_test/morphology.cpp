#define _USE_MATH_DEFINES
#include "morphology.h"
using namespace H5;

void Morphology::Union(Morphology* rhs)
{
	size_t cur_size{ positions.size() };
	positions.resize(cur_size + rhs->positions.size());
	std::copy(rhs->positions.begin(), rhs->positions.end(), positions.begin() + cur_size);
}

void Morphology::Translate(glm::vec3 const& translate)
{
    for (glm::vec3& pt : positions)
        pt += translate;
}

void Morphology::Scale(glm::vec3 const& scale)
{
    for (glm::vec3& pt : positions)
        pt *= scale;
}

bool HollowSphere::Generate(int const N)
{
    static auto gen{ std::default_random_engine() }; // No seed used. This means the randomness is the same for each run.
    std::uniform_real_distribution<float> dist(0, 2 * M_PI);
    positions.resize(N);
    for (int i = 0; i < N; ++i)
    {
        float const theta{ dist(gen) }, phi{ dist(gen) };
        positions[i] = { cos(theta)*sin(phi), sin(theta)*sin(phi), cos(phi) };
    }
    return true;
}

bool UnitSquare::Generate(int const N)
{
    static auto gen{ std::default_random_engine() }; // No seed used. This means the randomness is the same for each run.
    std::uniform_real_distribution<float> dist(-1, 1);
    positions.resize(N);
    for (int i = 0; i < N; ++i)
    {
        float const x{ dist(gen) }, y{ dist(gen) };
        positions[i] = { x, y, 0 };
    }
    return true;
}

bool UnitCube::Generate(int const N)
{
    static auto gen{ std::default_random_engine() }; // No seed used. This means the randomness is the same for each run.
    std::uniform_real_distribution<float> dist(-1, 1);
    positions.resize(N);
    for (int i = 0; i < N; ++i)
    {
        float const x{ dist(gen) }, y{ dist(gen) }, z{ dist(gen) };
        positions[i] = { x, y, z };
    }
    return true;
}

FileMorphology::FileMorphology(std::string const& fl_name_)
    : fl_name{ "../" + fl_name_ }, format{ fl_name_.substr(fl_name_.find_last_of('.') + 1) }
{
}

bool FileMorphology::Generate(int const N)
{
    if (format != "hdf" && format != "HDF")
        return false; // Unsupported format.
    return GenerateFromHDF(N);
}

bool FileMorphology::GenerateFromHDF(int const N)
{
    // First read in three individual axes, then put into positions,
    // finally scale and transform into range [-1, 1].
    std::vector<float> axes[3];
    static const char* coord_ids[3]{ "/x", "/y", "/z" };
    int numpnts;
    try
    {
        H5File file(fl_name, H5F_ACC_RDONLY);
        for (int i = 0; i < 3; ++i)
        {
            DataSet axis = file.openDataSet(coord_ids[i]);
            numpnts = (int)(axis.getStorageSize() / sizeof(float));
            axes[i].resize(numpnts);
            axis.read(axes[i].data(), PredType::NATIVE_FLOAT);
        }
    }
    catch (...)
    {
        return false;
    }

    // Calculate bounding box.
    glm::vec3 wmin, wmax;
    for (int i = 0; i < 3; ++i)
    {
        wmin[i] = *std::min_element(axes[i].begin(), axes[i].end());
        wmax[i] = *std::max_element(axes[i].begin(), axes[i].end());
    }

    // Check if we have enough points.
    if (N > numpnts)
    {
        std::cerr << "Not enough points in file " << fl_name 
                  << ". Have N = " << N << " and " << numpnts 
                  << " points available." << std::endl;
        return false;
    }

    // Sample points uniformly so that we have exactly N, not numpnts. 
    int stride{ numpnts / N };
    for (int i = 0; i < N; ++i)
    {
        int const j{ stride * i };
        positions[i] = { axes[0][j], axes[1][j], axes[2][j] };
    }

    // Translate so that min is zero.
    Translate(-wmin);

    // Scale so that everything is in the range [0, 2].
    Scale(glm::vec3(2.0f) / (wmax - wmin));

    // Finally, translate so that min is -1 and everything is in range [-1, 1].
    Translate(glm::vec3(-1));

	return true;
}
