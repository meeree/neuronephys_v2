// morphology.h : Handles generation and storage of neuron positions. 
#include "defines.cuh"
#include <ANN/ANN.h>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#pragma once

// Morphology is an abstract class that supports morphology generation and operations such as union.
class Morphology
{
protected:
	std::vector<glm::vec3> positions; // Positions in GLM format.

public:
	Morphology() = default;

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

	//! Getter for positions. 
	std::vector<glm::vec3> const& GetPositions() { return positions; }
	float* GetPositionsRaw() { return (float*)positions.data(); }
};

// Hollow Sphere shell. 2D Manifold.
class HollowSphere : public Morphology
{
public:
    virtual bool Generate(int const N) override;
};

// 2D Square [-1,1] X [-1,1]. 2D Manifold.
class UnitSquare : public Morphology
{
public:
    virtual bool Generate(int const N) override;
};

// 3D Square [-1,1] X [-1,1] X [-1,1]. 3D Manifold.
class UnitCube : public Morphology
{
public:
	virtual bool Generate(int const N) override;
};

// Morphology read in from a file with format inferred in constructor.
class FileMorphology : public Morphology
{
private:	
	std::string fl_name;	
	std::string format;

public:
	FileMorphology(std::string const& fl_name_);
	bool GenerateFromHDF(int const N);
	virtual bool Generate(int const N) override;
};
