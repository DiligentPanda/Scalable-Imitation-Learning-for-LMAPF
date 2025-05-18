#pragma once
#include "RHCR/main/BasicGraph.h"
#include "Grid.h"
#include "SharedEnv.h"

namespace RHCR {

class CompetitionGraph: public BasicGraph {
// for simplicity, just make everything public. but attributes or functions start with _ are supposed to use privately in general cases.
public:    
    CompetitionGraph(const SharedEnvironment & env);
    // a dummy function.
    bool load_map(string fname);
    // preprocessing the map, e.g., computing heuristics for later planning.
    void preprocessing(bool consider_rotation,const string & file_storage_path);
    // NOTE: in this competition, we only need to deal with grid-like graphs, so we just use BFS to compute shortest paths to all other locations. 
    void compute_heuristics(
        int root_location, 
        double * lengths, 
        bool * visited,
        State * queue,
        vector<double> & result,
        const int n_directions       
    );

    void save_heuristics_table(std::string fname);
    bool load_heuristics_table(std::ifstream& myfile);
};
}