#include "LNS/Parallel/TimeSpaceAStarPlanner.h"

namespace LNS {

namespace Parallel {

using State=TimeSpaceAStarState;

TimeSpaceAStarPlanner::TimeSpaceAStarPlanner(Instance & instance, std::shared_ptr<HeuristicTable> HT, std::shared_ptr<vector<float> > weights, int execution_window): instance(instance), HT(HT), weights(weights), execution_window(execution_window) {};

void TimeSpaceAStarPlanner::findPath(int start_pos, int start_orient, int goal_pos, ConstraintTable & constraint_table, const TimeLimiter & time_limiter) {
    clear();

    // at the beginning, we alway assume the agent havn't arrived its goal, even its start location are the same as the goal location. because we need at least length 2 path.
#ifndef NO_ROT    
    State * start_state = new State(start_pos, start_orient, 0, 0, HT->get(start_pos, start_orient, goal_pos), 0, false, nullptr);
#else
    State * start_state = new State(start_pos, start_orient, 0, 0, HT->get(start_pos, goal_pos), 0, false, nullptr);
#endif
    start_state->closed=false;
    start_state->open_list_handle=open_list.push(start_state);
    all_states.insert(start_state);

    // assert(constraint_table.length_min==0); // the length_min should be at least 1 (otherwise the agent can't reach its goal location
    // auto holding_time = constraint_table.getHoldingTime(goal_pos, constraint_table.length_min); // the earliest timestep that the agent can hold its goal location. The length_min is considered here.

    while (!open_list.empty() && !time_limiter.timeout()) {

        State * curr=open_list.top();
        open_list.pop();
        curr->closed=true;
        ++n_expanded;

        if (execution_window==1 && curr->pos==goal_pos && curr->t>=1 && !constraint_table.constrained(curr->pos, curr->t)) {
            buildPath(curr,goal_pos);
            return;
        }

        if (curr->t>=constraint_table.window_size_for_PATH) {
            // std::cerr<<"collision: "<<curr->num_of_conflicts<<std::endl;
            buildPath(curr,goal_pos);
            return;
        }

        getSuccessors(curr, goal_pos, constraint_table);
        for (auto & next_state: successors) {
            ++n_generated;
            auto iter = all_states.find(next_state);
            if (iter==all_states.end()) {
                // new state
                all_states.insert(next_state);
                next_state->closed=false;
                next_state->open_list_handle=open_list.push(next_state);
            } else {
                // old state
                auto old_state=(*iter);
                if (
                    next_state->num_of_conflicts<old_state->num_of_conflicts || (
                        next_state->num_of_conflicts==old_state->num_of_conflicts
                        && next_state->f<old_state->f
                    )
                ) {
                    // we need to update the state
                    old_state->copy(next_state);
                    if (old_state->closed) {
                        // if (!old_state->arrived){
                        //     std::cerr<<"reopen"<<std::endl;
                        //     exit(-1);
                        // }
                        // reopen closed state
                        old_state->closed=false;
                        old_state->open_list_handle=open_list.push(old_state);
                    } else {
                        // update open state
                        open_list.increase(old_state->open_list_handle);
                    }
                }
                delete next_state;
            }
        }
    }
}

void TimeSpaceAStarPlanner::clear() {
    open_list.clear();
    for (auto s: all_states) {
        delete s;
    }
    all_states.clear();
    path.clear();
    n_expanded = 0;
    n_generated = 0;
}

void TimeSpaceAStarPlanner::buildPath(State * curr, int goal_pos) {
    path.clear();
    path.path_cost = curr->f;

    auto s=curr;
    path.nodes.emplace_back(s->pos,s->orient);
    while (s->prev!=nullptr) {
        s=s->prev;
        path.nodes.emplace_back(s->pos,s->orient);
    }
    std::reverse(path.nodes.begin(), path.nodes.end());
}

void TimeSpaceAStarPlanner::getSuccessors(State * curr, int goal_pos, ConstraintTable & constraint_table) {
    successors.clear();

    int & cols=instance.num_of_cols;
    int & rows=instance.num_of_rows;
    auto & map=instance.my_map;
    auto & weights=*(this->weights);
    int pos=curr->pos;
    int x=pos%(cols);
    int y=pos/(cols);
    int orient=curr->orient;  


    int next_pos;
    int weight_idx;
    int next_timestep=curr->t+1;
    int next_orient=orient;
    float next_g;
    float next_h;
    int next_num_of_conflicts;
    bool next_arrived;

#ifndef NO_ROT
    // FW
    if (orient==0) {
        // east
        if (x+1<cols){
            next_pos=pos+1;
            weight_idx=pos*n_dirs;
            if (map[next_pos]==0 && !constraint_table.path_table_for_CT->constrained(pos,next_pos,next_timestep)) {
                next_g=curr->g+weights[weight_idx];
                next_h=curr->arrived?0:HT->get(next_pos, next_orient, goal_pos);
                next_num_of_conflicts=curr->num_of_conflicts+constraint_table.getNumOfConflictsForStep(curr->pos, next_pos, next_timestep);
                next_arrived=curr->arrived | (next_pos==goal_pos);
                successors.push_back(new State(next_pos, next_orient, next_timestep, next_g, next_h, next_num_of_conflicts, next_arrived, curr));
            }
        }
    } else if (orient==1) {
        // south
        if (y+1<rows) {
            next_pos=pos+cols;
            weight_idx=pos*n_dirs+1;
            if (map[next_pos]==0 && !constraint_table.path_table_for_CT->constrained(pos,next_pos,next_timestep)) {
                next_g=curr->g+weights[weight_idx];
                next_h=curr->arrived?0:HT->get(next_pos, next_orient, goal_pos);
                next_num_of_conflicts=curr->num_of_conflicts+constraint_table.getNumOfConflictsForStep(curr->pos, next_pos, next_timestep);
                next_arrived=curr->arrived | (next_pos==goal_pos);
                successors.push_back(new State(next_pos, next_orient, next_timestep, next_g, next_h, next_num_of_conflicts, next_arrived, curr));
            }
        }
    } else if (orient==2) {
        // west
        if (x-1>=0) {
            next_pos=pos-1;
            weight_idx=pos*n_dirs+2;
            if (map[next_pos]==0 && !constraint_table.path_table_for_CT->constrained(pos,next_pos,next_timestep)) {
                next_g=curr->g+weights[weight_idx];
                next_h=curr->arrived?0:HT->get(next_pos, next_orient, goal_pos);
                next_num_of_conflicts=curr->num_of_conflicts+constraint_table.getNumOfConflictsForStep(curr->pos, next_pos, next_timestep);
                next_arrived=curr->arrived | (next_pos==goal_pos);
                successors.push_back(new State(next_pos, next_orient, next_timestep, next_g, next_h, next_num_of_conflicts, next_arrived, curr));
            }
        }
    } else if (orient==3) {
        // north
        if (y-1>=0) {
            next_pos=pos-cols;
            weight_idx=pos*n_dirs+3;
            if (map[next_pos]==0 && !constraint_table.path_table_for_CT->constrained(pos,next_pos,next_timestep)) {
                next_g=curr->g+weights[weight_idx];
                next_h=curr->arrived?0:HT->get(next_pos, next_orient, goal_pos);
                next_num_of_conflicts=curr->num_of_conflicts+constraint_table.getNumOfConflictsForStep(curr->pos, next_pos, next_timestep);
                next_arrived=curr->arrived | (next_pos==goal_pos);
                successors.push_back(new State(next_pos, next_orient, next_timestep, next_g, next_h, next_num_of_conflicts, next_arrived, curr));
            }
        }
    } else {
        std::cerr<<"TimeSpaceAStartPlanner: invalid orient: "<<orient<<endl;
        exit(-1);
    }


    // for actions that don't change the position
    next_pos=pos;
    if (!constraint_table.path_table_for_CT->constrained(pos,next_pos,next_timestep)) {

        weight_idx=pos*n_dirs+4;
        next_g=curr->g+weights[weight_idx];
        next_num_of_conflicts=curr->num_of_conflicts+constraint_table.getNumOfConflictsForStep(curr->pos, next_pos, next_timestep);
        next_arrived=curr->arrived | (next_pos==goal_pos);

        // CR
        next_orient=(orient+1+n_orients)%n_orients;
        next_h=curr->arrived?0:HT->get(next_pos, next_orient, goal_pos);
        successors.push_back(new State(next_pos, next_orient, next_timestep, next_g, next_h, next_num_of_conflicts, next_arrived, curr));

        // CCR
        next_orient=(orient-1+n_orients)%n_orients;
        next_h=curr->arrived?0:HT->get(next_pos, next_orient, goal_pos);
        successors.push_back(new State(next_pos, next_orient, next_timestep, next_g, next_h, next_num_of_conflicts, next_arrived, curr));

        // W
        next_orient=orient;
        next_h=curr->arrived?0:HT->get(next_pos, next_orient, goal_pos);
        successors.push_back(new State(next_pos, next_orient, next_timestep, next_g, next_h, next_num_of_conflicts, next_arrived, curr));
    }

#else

    // R
    if (x+1<cols){
        next_pos=pos+1;
        weight_idx=pos*n_dirs;
        if (map[next_pos]==0 && !constraint_table.path_table_for_CT->constrained(pos,next_pos,next_timestep)) {
            next_g=curr->g+weights[weight_idx];
            next_h=curr->arrived?0:HT->get(next_pos, goal_pos);
            next_num_of_conflicts=curr->num_of_conflicts+constraint_table.getNumOfConflictsForStep(curr->pos, next_pos, next_timestep);
            next_arrived=curr->arrived | (next_pos==goal_pos);
            successors.push_back(new State(next_pos, next_orient, next_timestep, next_g, next_h, next_num_of_conflicts, next_arrived, curr));
        }
    }

    // D
    if (y+1<rows) {
        next_pos=pos+cols;
        weight_idx=pos*n_dirs+1;
        if (map[next_pos]==0 && !constraint_table.path_table_for_CT->constrained(pos,next_pos,next_timestep)) {
            next_g=curr->g+weights[weight_idx];
            next_h=curr->arrived?0:HT->get(next_pos, goal_pos);
            next_num_of_conflicts=curr->num_of_conflicts+constraint_table.getNumOfConflictsForStep(curr->pos, next_pos, next_timestep);
            next_arrived=curr->arrived | (next_pos==goal_pos);
            successors.push_back(new State(next_pos, next_orient, next_timestep, next_g, next_h, next_num_of_conflicts, next_arrived, curr));
        }
    }

    // L
    if (x-1>=0) {
        next_pos=pos-1;
        weight_idx=pos*n_dirs+2;
        if (map[next_pos]==0 && !constraint_table.path_table_for_CT->constrained(pos,next_pos,next_timestep)) {
            next_g=curr->g+weights[weight_idx];
            next_h=curr->arrived?0:HT->get(next_pos, goal_pos);
            next_num_of_conflicts=curr->num_of_conflicts+constraint_table.getNumOfConflictsForStep(curr->pos, next_pos, next_timestep);
            next_arrived=curr->arrived | (next_pos==goal_pos);
            successors.push_back(new State(next_pos, next_orient, next_timestep, next_g, next_h, next_num_of_conflicts, next_arrived, curr));
        }
    }

    // U
    if (y-1>=0) {
        next_pos=pos-cols;
        weight_idx=pos*n_dirs+3;
        if (map[next_pos]==0 && !constraint_table.path_table_for_CT->constrained(pos,next_pos,next_timestep)) {
            next_g=curr->g+weights[weight_idx];
            next_h=curr->arrived?0:HT->get(next_pos, goal_pos);
            next_num_of_conflicts=curr->num_of_conflicts+constraint_table.getNumOfConflictsForStep(curr->pos, next_pos, next_timestep);
            next_arrived=curr->arrived | (next_pos==goal_pos);
            successors.push_back(new State(next_pos, next_orient, next_timestep, next_g, next_h, next_num_of_conflicts, next_arrived, curr));
        }
    }

    // W
    next_pos=pos;
    weight_idx=pos*n_dirs+4;    
    if (!constraint_table.path_table_for_CT->constrained(pos,next_pos,next_timestep)) {
        next_g=curr->g+weights[weight_idx];
        next_h=curr->arrived?0:HT->get(next_pos, goal_pos);
        next_num_of_conflicts=curr->num_of_conflicts+constraint_table.getNumOfConflictsForStep(curr->pos, next_pos, next_timestep);
        next_arrived=curr->arrived | (next_pos==goal_pos);
        successors.push_back(new State(next_pos, next_orient, next_timestep, next_g, next_h, next_num_of_conflicts, next_arrived, curr));
    }

#endif
      
}

}

} // namespace LNS