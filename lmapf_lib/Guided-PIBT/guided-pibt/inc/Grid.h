#pragma once

#include "common.h"

class Grid{
public:
    Grid(string fname);
    Grid(const std::string & map_name, const std::vector<int> & map, int h, int w);

    int rows = 0;
    int cols = 0;
    std::vector<int> map;
    string map_name;

};
