#pragma once
#include "States.h"
#include "common.h"
#include "SharedEnv.h"

namespace LaCAM2 {

class Executor {
public:

    int rows;
    int cols;
    // loc->agent
    unordered_map<int,int> reservation_table;
    vector<bool> executed;
    const vector<State> * curr_states;
    const vector<State> * planned_next_states;
    vector<State> * next_states;

    Executor(const SharedEnvironment * _env):rows(_env->rows), cols(_env->cols){};
    Executor(int _rows, int _cols): rows(_rows), cols(_cols) {};   

    /*
     * NOTE(hj): currently we only consider one step
     * planned_next_states has no orientation    
     */ 
    void execute(const vector<State> * _curr_states, const vector<State> * _planned_next_states, vector<State> * _next_states) {
        assert(_curr_states->size()==_planned_next_states->size());
        assert(_curr_states->size()==_next_states->size());

        curr_states=_curr_states;
        planned_next_states=_planned_next_states;
        next_states=_next_states;

        executed.clear();
        executed.resize(curr_states->size(),false);
        reservation_table.clear();
        for (int i=0;i<curr_states->size();++i) {
            reservation_table[(*curr_states)[i].location]=i;
        }

        for (int i=0;i<curr_states->size();++i) {
            if (!executed[i]) {
                execute_agent(i,i);
            }
        }
    }

    void execute_agent(int agent_idx, int root_agent_idx) {
        // std::cerr<<"execute agent "<<agent_idx<<endl;
        assert(!executed[agent_idx]);
        // put this line here, we can also deal with the loop case

        const auto & curr_state=(*curr_states)[agent_idx];
        const auto & planned_next_state=(*planned_next_states)[agent_idx];
        auto & next_state=(*next_states)[agent_idx];

        if (curr_state.location==planned_next_state.location) {
            // just need to wait
            next_state.location=curr_state.location;
            if (planned_next_state.orientation!=-1) {
                next_state.orientation=planned_next_state.orientation;
            } else {
                next_state.orientation=curr_state.orientation;
            }
            next_state.timestep=curr_state.timestep+1;
        } else {
            // need to possibly rotate, then foward
            int curr_orient=curr_state.orientation;
            int expected_orient=get_neighbor_orientation(curr_state.location,planned_next_state.location);
            if (curr_orient==expected_orient) {
                // it is ready to forward but we need to make sure the next location is empty
                auto iter=reservation_table.find(planned_next_state.location);
                if (iter!=reservation_table.end()) {
                    int parent_agent_idx=iter->second;
                    if (parent_agent_idx!=root_agent_idx && !executed[parent_agent_idx]) {
                        execute_agent(parent_agent_idx,root_agent_idx);
                    }
                    // parent anget must be excuted now.
                    if (parent_agent_idx==root_agent_idx || (*next_states)[parent_agent_idx].location!=(*curr_states)[parent_agent_idx].location) {
                        // if we have a loop or parent agent also forwards, then we can foward as well
                        next_state.location=planned_next_state.location;
                        next_state.orientation=curr_state.orientation;
                        next_state.timestep=curr_state.timestep+1;
                    } else {
                        // if parent agent still stays, then we need to wait
                        next_state.location=curr_state.location;
                        if (planned_next_state.orientation!=-1) {
                            next_state.orientation=planned_next_state.orientation;
                        } else {
                            next_state.orientation=curr_state.orientation;
                        }
                        next_state.timestep=curr_state.timestep+1;
                    }
                } else {
                    // if empty, then readily to forward
                    next_state.location=planned_next_state.location;
                    next_state.orientation=curr_state.orientation;
                    next_state.timestep=curr_state.timestep+1;
                }
            } else {
                // we need to rotate
                int d1=(curr_orient+4-expected_orient)%4;
                int d2=(expected_orient+4-curr_orient)%4;

                int next_orient=-1;
                if (d1<d2) {
                    next_orient=(curr_orient-1+4)%4;
                } else {
                    next_orient=(curr_orient+1+4)%4;
                }
                
                next_state.location=curr_state.location;
                if (planned_next_state.orientation!=-1) {
                    next_state.orientation=planned_next_state.orientation;
                } else {
                    next_state.orientation=next_orient;
                }
                next_state.timestep=curr_state.timestep+1;
            }
        }

        executed[agent_idx]=true;
    }

    int get_neighbor_orientation(int loc1,int loc2) {

        // 0:east, 1:south, 2:west, 3:north

        if (loc1+1==loc2) {
            return 0;
        }

        if (loc1+cols==loc2) {
            return 1;
        }

        if (loc1-1==loc2) {
            return 2;
        }

        if (loc1-cols==loc2) {
            return 3;
        }

        std::cerr<<"executor: loc1 and loc2 are not neighbors: "<<loc1<<", "<<loc2<<endl;
        exit(-1);

    }

};

} // namespace LaCAM2
