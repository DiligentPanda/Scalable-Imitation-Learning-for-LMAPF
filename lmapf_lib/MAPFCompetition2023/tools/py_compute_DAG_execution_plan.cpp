#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

namespace py = pybind11;

std::vector<int> compute_DAG_execution_plan(int n, std::vector<std::vector<int> > & edges) {
    std::vector<int> in_degrees(n, 0);
    for (auto & edge: edges) {
        int out_idx=edge[0];
        int in_idx=edge[1];
        for 
    }
}