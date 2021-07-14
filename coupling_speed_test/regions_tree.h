#pragma once
#include <nlohmann/json.hpp> // Region info is stored in a json format.
using json = nlohmann::json;

//! Stuff for reading in Allen atlas regions which are broken up into a heirarchical tree of subregions. 
//! Also for working with this tree, e.g., to figure out which region is the parent of a subregion.
class RegionsTree
{
private:
    json regions_json;

public:
    //! Reads in the JSON regions. Throws if there is an error.
    //! \param[in] json_path Path to JSON storing region information.
    RegionsTree(std::string json_path);

    //! Retrieves the exact region at a specified depth in the region tree so that region_idx is an ancestor (i.e. subregion).
    int GetParentRegionAtDepth(int region_idx, int depth);
};
