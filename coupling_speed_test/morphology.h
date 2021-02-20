// morphology.h : Handles generation and storage of neuron positions. 
#pragma once
#include "defines.cuh"
#include <ANN/ANN.h>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#pragma once

//! Morphology is an abstract class that supports morphology generation 
//! and operations such as union, nearest-neighbor, etc.
class Morphology
{
protected:
	std::unique_ptr<ANNkd_tree> kd_tree; //! Tree used by ANN for spatial search.
	std::vector<glm::vec3> positions; //! Positions in GLM format.

	//! Creates kd_tree using positions.
	void SetupANN();

public:
	Morphology() : kd_tree{ nullptr } {}

	//! Generates a morphology. Pure virtual function.
	/*!
	  \param [in] N Number of points.
      \return If the function is successful for not.
	*/
    virtual bool Generate (int const N) = 0;

	//! Computes the union of this morphology plus another morphology and stores it in updated positions.
	/*!
	  \param [in] rhs The morphology to union with.
	*/
    void Union (Morphology* rhs); 

	//! Translates the positions and overrites the old positions.
	/*!
	  \param [in] translation Translation vector.
	*/
    void Translate(glm::vec3 const& transform);

	//! Scale the positions and overrites the old positions.
	/*!
	  \param [in] scale Scale vector.
	*/
    void Scale(glm::vec3 const& scale);

	//! Finds the k nearest neighbros to neuron n.
	/*!
	  \param [in] n Index of neuron.
	  \param [in] k Number of nearest neighbors to find, including n.
	  \param [out] knn Indices of nearest neighbors.
	  \param [out] knn_dists Distances to each respective neighbor.
	*/
	void KNN(int const n, int const k, std::vector<int>& knn, std::vector<ANNdist>& knn_dists);

	//! Generate a list of KNN regions at specific neurons.
	/*!
		\param [out] h_regions Column indices for region adjacency list. 
		\param [out] h_region_inds Row offsets for region adjacency list.
		\param [in] electrode_inds Indices of neurons at which to place electrodes.
		\param [in] k Number of nearest neighbors for regions.
	*/	
	void GenSphericalRegions(thrust::host_vector<int>& h_regions, thrust::host_vector<int>& h_region_inds, \
		                     std::vector<int> const& electrode_inds, int const k);

	//! Getter for positions. 
	std::vector<glm::vec3> const& GetPositions() { return positions; }
	float* GetPositionsRaw() { return (float*)positions.data(); }

	virtual ~Morphology() { annClose(); }
};

//! Hollow Sphere shell. 2D Manifold.
class HollowSphere : public Morphology
{
public:
    virtual bool Generate(int const N) override;
};

//! 2D Square [-1,1] X [-1,1]. 2D Manifold.
class UnitSquare : public Morphology
{
public:
    virtual bool Generate(int const N) override;
};

//! 3D Square [-1,1] X [-1,1] X [-1,1]. 
class UnitCube : public Morphology
{
public:
	virtual bool Generate(int const N) override;
};

//! Volume given by the solution of the equations F(x,y,z) <= 0.
//! NOTE THAT WE ASSUME THE SOLID IS WITHIN THE UNIT CUBE!
class ImplicitSolid : public Morphology
{
protected:
	//! Defines the inside, outside and boundary by F(x,y,z) < 0, > 0 and = 0, respectively.
	virtual Float F(glm::vec3 const& p) = 0;
public:
	virtual bool Generate(int const N) override;
};

//! Solid unit sphere implemented as an implicit solid.
class SolidSphere : public ImplicitSolid
{
protected:
	//! Returns L2 norm squared minus 1.
	virtual Float F(glm::vec3 const& p) override;
};

//! Sphere with "wrinkles" that are polar oscillations in the radius.
class SimpleWrinkledSolidSphere : public ImplicitSolid
{
protected:
	//! Returns L2 norm squared minus 1 plus some variation for the "wrinkles.
	virtual Float F(glm::vec3 const& p) override;
};

//! Morphology read in from a file with format inferred in constructor.
class FileMorphology : public Morphology
{
protected:	
	std::string fl_name;	
	std::string format;

public:
	/*!
	  \param [in] fl_name_ Name of file to read from.
	*/
	FileMorphology(std::string const& fl_name_);
	bool GenerateFromHDF(int const N);
	virtual bool Generate(int const N) override;
};
