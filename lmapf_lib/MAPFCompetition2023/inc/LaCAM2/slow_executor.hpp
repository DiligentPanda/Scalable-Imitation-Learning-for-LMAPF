#pragma once
#include "States.h"
#include "common.h"
#include "SharedEnv.h"

namespace LaCAM2 {

class SlowExecutor {
public:

    const SharedEnvironment * env;
    const vector<State> * curr_states;
    const vector<State> * planned_next_states;
    vector<State> * next_states;

    SlowExecutor(const SharedEnvironment * _env):env(_env){};

    /*
     * we need to check if everyone is oriented, if so, all agents forward togather.
     */ 
    void execute(const vector<State> * _curr_states, const vector<State> * _planned_next_states, vector<State> * _next_states) {
        assert(_curr_states->size()==_planned_next_states->size());
        assert(_curr_states->size()==_next_states->size());

        curr_states=_curr_states;
        planned_next_states=_planned_next_states;
        next_states=_next_states;
        int n_agents=curr_states->size();

        bool all_oriented=true;
        for (int i=0;i<n_agents;++i) {
            if ((*curr_states)[i].location!=(*planned_next_states)[i].location) {
                int expected_orient=get_neighbor_orientation((*curr_states)[i].location,(*planned_next_states)[i].location);
                if ((*curr_states)[i].orientation!=expected_orient) {
                    // cerr<<"agent "<<i<<" is not oriented: "<<(*curr_states)[i].orientation<<","<<expected_orient<<endl;
                    all_oriented=false;
                    break;
                }
            }
        }

        for (int i=0;i<n_agents;++i) {
            if ((*curr_states)[i].location==(*planned_next_states)[i].location) {
                //TODO(rivers): we could include next step in the plan to rotate earlier
                // currently, we just wait
                (*next_states)[i].location=(*curr_states)[i].location;
                (*next_states)[i].orientation=(*curr_states)[i].orientation;
                (*next_states)[i].timestep=(*curr_states)[i].timestep+1;
            } else {
                int expected_orient=get_neighbor_orientation((*curr_states)[i].location,(*planned_next_states)[i].location);
                if (all_oriented) {
                    // all agents forward togather
                    (*next_states)[i].location=(*planned_next_states)[i].location;
                    (*next_states)[i].orientation=expected_orient;
                    (*next_states)[i].timestep=(*curr_states)[i].timestep+1;
                } else {
                    // consider rotation first
                    int next_orient=get_next_orientation((*curr_states)[i].orientation,expected_orient);
                    (*next_states)[i].location=(*curr_states)[i].location;
                    (*next_states)[i].orientation=next_orient;
                    (*next_states)[i].timestep=(*curr_states)[i].timestep+1;
                }
            }
        // cerr<<"agent "<<i<<": "<<(*next_states)[i]<<endl;
        }
    }

    int get_next_orientation(int curr_orient, int target_orient) {
        if (curr_orient==target_orient){
            return curr_orient;
        }

        int next_orient1=(curr_orient+1+4)%4;
        int next_orient2=(curr_orient-1+4)%4;

        int d1=(next_orient1-target_orient+4)%4;
        int d2=(next_orient2-target_orient+4)%4;
        if (d1<d2) {
            return next_orient1;
        } else {
            return next_orient2;
        }
    }

    int get_neighbor_orientation(int loc1,int loc2) {

        // 0:east, 1:south, 2:west, 3:north

        if (loc1+1==loc2) {
            return 0;
        }

        if (loc1+env->cols==loc2) {
            return 1;
        }

        if (loc1-1==loc2) {
            return 2;
        }

        if (loc1-env->cols==loc2) {
            return 3;
        }

        std::cerr<<"executor: loc1 and loc2 are not neighbors: "<<loc1<<", "<<loc2<<endl;
        exit(-1);

    }

};

} // namespace LaCAM2
