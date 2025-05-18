#include <iostream>
#include "util/HeuristicTable.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <tuple>
#include <string>
#include <queue>
#include <utility>
#include "util/Timer.h"

namespace py = pybind11;

std::tuple<int, py::array_t<int>, py::array_t<float> >  compute_heuristics(
        // std::string & map_path, 
        // std::string & map_weights_path
        int rows,
        int cols,
        std::vector<int> & map,
        std::string & map_weights_path
    ) {
    g_timer.record_p("compute_heuristic_total_s");

    // Grid *grid= new Grid(map_path);
    
    // TODO: map weights are not supported now.
    // if (_map_weights.size()!=rows*cols*5) {
    //     std::cerr<<"the size of map weights is wrong!"<<std::endl;
    //     exit(1);
    // }

    // just a legacy wrapper
    SharedEnvironment* env=new SharedEnvironment();
    env->map=map;
    env->rows=rows;
    env->cols=cols;

    // std::shared_ptr<std::vector<float> > map_weights=std::make_shared<std::vector<float> >(_map_weights);

    // use default weight matrix for now
    std::shared_ptr<std::vector<float> > map_weights=std::make_shared<std::vector<float> >(env->rows*env->cols*5, 1.0);

    if (map_weights_path!=""){
        std::ifstream f(map_weights_path);
        try
        {
            nlohmann::json _weights = nlohmann::json::parse(f);
            if (_weights.size()!=map_weights->size()) {
                std::cerr<<"map weights size mismatch"<<std::endl;
                exit(-1);
            }

            for (int i=0;i<map_weights->size();++i){
                (*map_weights)[i]=_weights[i].get<float>();
            }
            
        }
        catch (nlohmann::json::parse_error error)
        {
            std::cerr << "Failed to load " << map_weights_path << std::endl;
            std::cerr << "Message: " << error.what() << std::endl;
            exit(1);
        }
    }

    // we set consider_rotation to false
    bool consider_rotation=false;
    HeuristicTable* heuristic_table=new HeuristicTable(env, map_weights ,consider_rotation);

    g_timer.record_p("search_s");
    heuristic_table->compute_weighted_heuristics();
    g_timer.record_d("search_s","search");

    g_timer.record_p("copy_s");
    size_t loc_size= heuristic_table->loc_size;
    auto np_empty_locs = py::array_t<int>(loc_size);
    std::memcpy(np_empty_locs.mutable_data(), heuristic_table->empty_locs, loc_size*sizeof(int));
    auto np_main_heuristics = py::array_t<float>(loc_size*loc_size);
    std::memcpy(np_main_heuristics.mutable_data(), heuristic_table->main_heuristics, loc_size*loc_size*sizeof(float));
    g_timer.record_d("copy_s","copy");

    // TODO implement it efficiently
    

    // we also return the BFS neighbors here, include itself
    // auto BFS_neighbors = std::make_shared<std::vector<int> >(loc_size*max_BFS_neighbors);
    
    // for (int i=0;i<loc_size;++i) {
    //     std::priority_queue<std::pair<float,int>, std::greater<int> > q;
    //     for (int j=0;j<loc_size;++j) {
    //         q.emplace(heuristic_table->main_heuristics[i*loc_size+j],j);
    //     }
    //     for (int k=0;k<max_BFS_neighbors;++k) {
    //         auto & p=q.top();
    //         BFS_neighbors[i*max_BFS_neighbors+k]=p.second;
    //     }
    // }

    delete env;
    delete heuristic_table;

    // cout<<"Computation of heuristics completed!"<<endl; 

    g_timer.record_d("compute_heuristic_total_s","compute_heuristic_total");
    g_timer.print_all_d();

    return std::make_tuple(loc_size, np_empty_locs, np_main_heuristics);
}

std::string playground(){
	return "hello, test!";
}


PYBIND11_MODULE(py_compute_heuristics, m) {
	// optional module docstring
    // m.doc() = ;

    m.def("playground", &playground, "Playground function to test everything");
    // m.def("add", &add, py::arg("i")=0, py::arg("j")=1);
    m.def("compute_heuristics", &compute_heuristics, "Function to compute heuristics");
}