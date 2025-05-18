#include <iostream>
#include "util/HeuristicTable.h"

int main(int argc, char** argv){

    if (argc!=3) {
        cout<<"Usage: ./compute_heuristics <map_path> <output_path>"<<endl;
        exit(-1);
    }

    string map_path=argv[1];
    string output_path=argv[2];


    Grid *grid= new Grid(map_path);

    // just a legacy wrapper
    SharedEnvironment* env=new SharedEnvironment();
    env->map=grid->map;
    env->rows=grid->rows;
    env->cols=grid->cols;

    // use default weight matrix for now
    std::shared_ptr<std::vector<float> > map_weights=std::make_shared<std::vector<float> >(env->rows*env->cols*5, 1.0);

    // we set consider_rotation to false
    bool consider_rotation=false;
    HeuristicTable* heuristic_table=new HeuristicTable(env, map_weights ,consider_rotation);

    heuristic_table->compute_weighted_heuristics();
    heuristic_table->save(output_path);


    delete grid;
    delete env;
    delete heuristic_table;

    // cout<<"Computation of heuristics completed!"<<endl; 
    return 0;
}