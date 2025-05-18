#pragma once
#include "SharedEnv.h"
#include "util/MyLogger.h"
#include <vector>
#include "util/HeuristicTable.h"
#include "LaCAM2/SUO2/CostMap.h"
#include "LaCAM2/SUO2/SpatialState.h"
#include "boost/heap/pairing_heap.hpp"
#include "boost/unordered_set.hpp"

namespace SUO2 {

namespace Spatial {

class AStar {

public:

    const SharedEnvironment & env;
    const std::vector<float> & weights;
    std::shared_ptr<HeuristicTable> HT;
    CostMap cost_map;

    int planning_window; // TODO
    std::vector<State*> successors;

    boost::heap::pairing_heap<State*, boost::heap::compare<State::StateCompare> > open_list;
    boost::unordered_set<State*, State::StateHash, State::StateEqual> all_states;

    const int n_dirs=5; // right,down,left,up,stay
    const int n_orients=4;

    AStar(
        const SharedEnvironment & env, const std::vector<float> & weights, std::shared_ptr<HeuristicTable> & HT, int planning_window
    ): env(env), weights(weights),cost_map(env.cols,env.rows), HT(HT), planning_window(planning_window) {
        
    };


    void reset_plan() {
        open_list.clear();
        for (auto & state: all_states) {
            delete state;
        }
        all_states.clear();
    }

    void clear_cost_map() {
        this->cost_map.clear();
    }

    void update_cost_map(const std::vector<std::pair<int,float> > & deltas) {
        // make a copy
        this->cost_map.update(deltas);
    }

    void remove_path_cost(const std::vector<State> & path, float vertex_collision_cost) {
        for (const auto & state: path) {
            this->cost_map[state.pos]-=vertex_collision_cost;
        }
    }

    void add_path_cost(const std::vector<State> & path, float vertex_collision_cost) {
        for (const auto & state: path) {
            this->cost_map[state.pos]+=vertex_collision_cost;
        }
    }

    ~AStar() {
        reset_plan();
    };

    void get_successors(State * curr, int goal_pos) {

#ifndef NO_ROT
        successors.clear();
        int pos=curr->pos;
        int x=pos%(env.cols);
        int y=pos/(env.cols);
        int orient=curr->orient;

        // FW
        int next_pos;
        int next_orient;
        int weight_idx;
        next_orient=orient;
        if (orient==0) {
            // east
            if (x+1<env.cols){
                next_pos=pos+1;
                weight_idx=pos*n_dirs;
                if (env.map[next_pos]==0) {
                    successors.push_back(new State(
                        next_pos, 
                        next_orient, 
                        curr->g+weights[weight_idx]+cost_map[next_pos], 
                        HT->get(next_pos, next_orient, goal_pos),
                        curr
                    ));
                }
            }
        } else if (orient==1) {
            // south
            if (y+1<env.rows) {
                next_pos=pos+env.cols;
                weight_idx=pos*n_dirs+1;
                if (env.map[next_pos]==0) {
                    successors.push_back(new State(
                        next_pos, 
                        next_orient, 
                        curr->g+weights[weight_idx]+cost_map[next_pos], 
                        HT->get(next_pos, next_orient, goal_pos),
                        curr
                    ));
                }
            }
        } else if (orient==2) {
            // west
            if (x-1>=0) { 
                next_pos=pos-1;
                weight_idx=pos*n_dirs+2;
                if (env.map[next_pos]==0) {
                    successors.push_back(new State(
                        next_pos, 
                        next_orient, 
                        curr->g+weights[weight_idx]+cost_map[next_pos], 
                        HT->get(next_pos, next_orient, goal_pos),
                        curr
                    ));
                }
            }
        } else if (orient==3) {
            // north
            if (y-1>=0) {
                next_pos=pos-env.cols;
                weight_idx=pos*n_dirs+3;
                if (env.map[next_pos]==0) {
                    successors.push_back(new State(
                        next_pos, 
                        next_orient, 
                        curr->g+weights[weight_idx]+cost_map[next_pos], 
                        HT->get(next_pos, next_orient, goal_pos),
                        curr
                    ));
                }
            }
        } else {
            std::cerr<<"spatial search in heuristics: invalid orient: "<<orient<<endl;
            exit(-1);
        }        


        next_pos=pos;
        weight_idx=pos*n_dirs+4;
        // CR
        next_orient=(orient+1+n_orients)%n_orients;
        successors.push_back(new State(
            next_pos, 
            next_orient, 
            curr->g+weights[weight_idx]+cost_map[next_pos], 
            HT->get(next_pos, next_orient, goal_pos),
            curr
        ));

        // CCR
        next_orient=(orient-1+n_orients)%n_orients;
        successors.push_back(new State(
            next_pos, 
            next_orient, 
            curr->g+weights[weight_idx]+cost_map[next_pos], 
            HT->get(next_pos, next_orient, goal_pos),
            curr
        ));

        // no wait action because we don't use time in the state
        // // W
        // next_orient=orient;
        // successors.push_back(new State(
        //     next_pos, 
        //     next_orient, 
        //     curr->g+weights[weight_idx], 
        //     HT->get(next_pos, next_orient, goal_pos),
        //     curr
        // ));
#else
        std::cout<<"not support no rotation"<<std::endl;
#endif
        
    }

    State * search(int start_pos, int start_orient, int goal_pos) {
        State * start=new State(
            start_pos, 
            start_orient, 
            0, 
            HT->get(start_pos, start_orient, goal_pos),
            nullptr
        );
        all_states.insert(start);
        start->closed=false;
        start->open_list_handle=open_list.push(start);

        while (!open_list.empty()) {
            State * curr=open_list.top();
            open_list.pop();
            curr->closed=true;

            // goal checking
            if (curr->pos==goal_pos) {
                return curr;
            }
            
            get_successors(curr,goal_pos);

            for (auto & next: successors) {

                auto iter=all_states.find(next);
                if (iter==all_states.end()) {
                    // new state
                    auto & new_state=next;
                    all_states.insert(new_state);
                    new_state->closed=false;
                    new_state->open_list_handle=open_list.push(new_state);
                } else {
                    // old state
                    State * old_state=*iter;
                    if (next->g<old_state->g) {
                        // we need to update the state
                        old_state->copy(next);
                        if (old_state->closed) {
                            // re-open
                            old_state->closed=false;
                            open_list.push(old_state);
                        } else {
                            open_list.increase(old_state->open_list_handle);
                        }
                    }
                    delete next;
                }
            }
        }

        return nullptr;
    }



};










}

}