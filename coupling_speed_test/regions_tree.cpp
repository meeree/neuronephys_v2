#include "regions_tree.h"
#include <fstream>
#include <iostream>

RegionsTree::RegionsTree(std::string json_path)
{
    std::ifstream json_fl("../" + json_path);
    if (!json_fl)
        throw std::runtime_error("Failed to read in JSON region information from path: " + json_path);

    try
    {
        regions_json = json::parse(json_fl);
    }
    catch (json::parse_error& e)
    {
        std::cerr << e.what() << std::endl;
        throw std::runtime_error("JSON parse error was thrown");
    }
    json_fl.close();
}

int RegionsTree::GetParentRegionAtDepth(int region_idx, int depth)
{
    // Get index of entry corresponding to region_idx in JSON.
    int idx_in_json = -1;
    for (int i = 0; i < regions_json.size(); ++i)
    {
        if (regions_json[i]["id"] == region_idx)
        {
            idx_in_json = i;
            break;
        }
    }
    if (idx_in_json == -1)
    {
        std::cout << "SHIT" << std::endl;
        throw std::runtime_error("Coulding find region with index " + std::to_string(region_idx) + " in JSON.");
    }
    return  regions_json[idx_in_json]["structure_id_path"][depth];
}
