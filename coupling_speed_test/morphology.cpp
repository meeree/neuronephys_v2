#define _USE_MATH_DEFINES
#include "morphology.h"
#include <glm/ext.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/type_ptr.hpp>
using namespace H5;

void Morphology::Union(Morphology* rhs)
{
    std::cout << "Computing union of two morphologies..." << std::endl;
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

void Morphology::KNN(int const n, int const k, std::vector<int>& knn, std::vector<ANNdist>& knn_dists)
{
    if (!kd_tree)
        SetupANN();
    ANNpoint pt{ glm::value_ptr(positions[n]) };
	kd_tree->annkSearch(pt, k, knn.data(), knn_dists.data());
}

void Morphology::GenSphericalRegions(
    thrust::host_vector<int>& h_regions, thrust::host_vector<int>& h_region_inds, 
    std::vector<int> const& electrode_inds, int const k)
{
    if (!kd_tree)
        SetupANN();

    std::cout << "Generating spherical regions..." << std::endl;
    std::vector<int> knn(k+1); 
    std::vector<ANNdist> knn_dists(k+1);

	int const N_regions = electrode_inds.size();
    h_regions.resize(N_regions * k);
    h_region_inds.resize(N_regions + 1, 0);
    for (int n = 0; n < N_regions; ++n)
    {
		// Initialize query point. 
		int const idx = electrode_inds[n];
	
        KNN(n, k, knn, knn_dists);
		std::copy(knn.begin(), knn.end(), h_regions.begin() + k * n);
        h_region_inds[n + 1] = h_region_inds[n] + k;
    }
}

void Morphology::SetupANN ()
{
    std::cout << "Setting up ANN..." << std::endl;
    int const N = positions.size();
    ANNpointArray p_vec{ annAllocPts(N, 3) };
    for (int i = 0; i < N; ++i)
        p_vec[i] = glm::value_ptr(positions[i]);
	kd_tree = std::make_unique<ANNkd_tree>(p_vec, N, 3);
}

bool HollowSphere::Generate(int const N)
{
    std::cout << "Generating hollow sphere morphology..." << std::endl;
    auto& gen{ Utility::UnseededRandomEngine() };
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
    std::cout << "Generating unit square morphology..." << std::endl;
    auto& gen{ Utility::UnseededRandomEngine() };
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
    std::cout << "Generating unit cube morphology..." << std::endl;
    auto& gen{ Utility::UnseededRandomEngine() };
    std::uniform_real_distribution<float> dist(-1, 1);
    positions.resize(N);
    for (int i = 0; i < N; ++i)
    {
        float const x{ dist(gen) }, y{ dist(gen) }, z{ dist(gen) };
        positions[i] = { x, y, z };
    }
    return true;
}

bool ImplicitSolid::Generate(int const N)
{
    // Generate points uniformly in unit cube and only keep those for which F(p) <= 0.
    std::cout << "Generating implicit solid morphology..." << std::endl;
    auto& gen{ Utility::UnseededRandomEngine() };
    std::uniform_real_distribution<float> dist(-1, 1);
    positions.resize(N);
    int pos_size = 0;
    while (pos_size < N)
    {
        float const x{ dist(gen) }, y{ dist(gen) }, z{ dist(gen) };
        glm::vec3 p{ x, y, z };
        if (F(p) <= 0)
        {
            positions[pos_size] = p;
            pos_size += 1;
        }
    }
    return true;
}

Float SolidSphere::F(glm::vec3 const& p)
{
    return glm::length2(p) - 1;
}

Float SimpleWrinkledSolidSphere::F(glm::vec3 const& p)
{
    // Convert cartesian to spherical coordinates.
    Float r = glm::length(p);
    Float theta = atan2(p.y, p.x);
    Float phi = atan2(sqrt(p.x * p.x + p.y * p.y), p.z);
    return r - cos(phi) * (0.15 * cos(20 * theta) + 0.85);
}

FileMorphology::FileMorphology(std::string const& fl_name_)
    : fl_name{ "../" + fl_name_ }, format{ fl_name_.substr(fl_name_.find_last_of('.') + 1) }
{
}

bool FileMorphology::Generate(int const N)
{
    std::cout << "Generating morphology from file '" + fl_name + "'..." << std::endl;
    if (format != "h5" && format != "H5")
    {
        std::cerr << "Unsupported format " << format << std::endl;
        return false;
    }
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
