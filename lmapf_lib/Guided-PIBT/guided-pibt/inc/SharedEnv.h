#pragma once
#include "States.h"
#include "nlohmann/json.hpp"


class SharedEnvironment {
public:
    int num_of_agents;

    int rows;
    int cols;
    std::string map_name;
    std::vector<int> map;

    int max_h=0;

    std::string file_storage_path;

    // goal locations for each agent
    // each task is a pair of <goal_loc, reveal_time>
    vector< vector<pair<int, int> > > goal_locations;
	std::vector<std::vector<int>> neighbors;

    int curr_timestep = 0;
    vector<State> curr_states;

    SharedEnvironment(){}

    void init_neighbor(){
        neighbors.resize(rows * cols);
        for (int row=0; row<rows; row++){
            for (int col=0; col<cols; col++){
                int loc = row*cols+col;
                if (map[loc]==0){
                    if (row>0 && map[loc-cols]==0){
                        neighbors[loc].push_back(loc-cols);
                    }
                    if (row<rows-1 && map[loc+cols]==0){
                        neighbors[loc].push_back(loc+cols);
                    }
                    if (col>0 && map[loc-1]==0){
                        neighbors[loc].push_back(loc-1);
                    }
                    if (col<cols-1 && map[loc+1]==0){
                        neighbors[loc].push_back(loc+1);
                    }
                }
            }
        }
    };

};
