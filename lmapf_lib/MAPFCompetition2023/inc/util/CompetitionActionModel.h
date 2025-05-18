#pragma once
#include <string>
#include "Grid.h"
#include "States.h"
#include "Logger.h"
#include "SharedEnv.h"
#include "ActionModel.h"

#ifndef NO_ROT

// TODO(hj): move it outside the RHCR
class CompetitionActionModelWithRotate
{
public:
    list<std::tuple<std::string,int,int,int>> errors;

    CompetitionActionModelWithRotate(){};
    CompetitionActionModelWithRotate(SharedEnvironment * env): env(env),rows(env->rows),cols(env->cols){
        moves[0] = 1;
        moves[1] = cols;
        moves[2] = -1;
        moves[3] = -cols;

    };

    bool is_valid(const vector<State>& prev, const vector<Action> & action);
    void set_logger(Logger* logger){this->logger = logger;}

    vector<State> result_states(const vector<State>& prev, const vector<Action> & action){
        vector<State> next(prev.size());
        for (size_t i = 0 ; i < prev.size(); i ++){
            next[i] = result_state(prev[i], action[i]);
        }
        return next;
    };

    vector<State> get_state_neighbors(const State & curr,bool no_forward) {
        vector<State> neighbors;

        int x=curr.location%env->cols;
        int y=curr.location/env->cols;

        if (!no_forward){
            const auto & next_state=result_state(curr,Action::FW,true);

            // if won't go out of map
            if (next_state.location!=-1) {
                neighbors.push_back(result_state(curr,Action::FW));
            }
        }
        neighbors.push_back(result_state(curr,Action::CR));
        neighbors.push_back(result_state(curr,Action::CCR));
        neighbors.push_back(result_state(curr,Action::W));

        return neighbors;
    }

    vector<State> get_loc_neighbors(const State & curr,bool move_orient=true) {
        vector<State> neighbors;

        int x=curr.location%env->cols;
        int y=curr.location/env->cols;

        for (int i=0;i<4;++i) {
            int new_location=curr.location+moves[i];
            int new_orientation;
            
            if (move_orient) {
                new_orientation=i;
            } else {
                new_orientation=0;
            }

            if ((x==env->cols-1 && i==0) || (y==env->rows-1 && i==1) || (x==0 && i==2) || (y==0 && i==3)) {
                continue;
            }

            // if in the map but blocked
            if (env->map[new_location]) {
                continue;
            }

            neighbors.push_back(State(new_location,curr.timestep+1,new_orientation));
        }

        return neighbors;
    }

    SharedEnvironment * env=nullptr;
    int rows;
    int cols;
    int moves[4];
    Logger* logger = nullptr;

    State result_state(const State & prev, Action action, bool check=false)
    {
        int new_location = prev.location;
        int new_orientation = prev.orientation;
        if (action == Action::FW)
        {
            new_location = new_location += moves[prev.orientation];
            if (check) {
                int x=prev.location%env->cols;
                int y=prev.location/env->cols;

                int nx=new_location%env->cols;
                int ny=new_location/env->cols;

                if ((x==env->cols-1 && prev.orientation==0) || (y==env->rows-1 && prev.orientation==1) || (x==0 && prev.orientation==2) || (y==0 && prev.orientation==3)) {
                    // std::cerr<<"cannot forward: "<<(x==env->cols-1 && prev.orientation==0)<< (y==env->rows-1 && prev.orientation==1) << (x==0 && prev.orientation==2) << (y==0 )<<endl;
                    // std::cerr<<"pos"<<x<<","<<y<<endl;
                    return State(-1,prev.timestep+1,new_orientation=prev.orientation);
                }

                // if in the map but blocked
                if (env->map[new_location]) {
                    return State(-1,prev.timestep+1,new_orientation=prev.orientation);
                }
            }
        }
        else if (action == Action::CR)
        {
            new_orientation = (prev.orientation + 1) % 4;
      
        }
        else if (action == Action::CCR)
        {
            new_orientation = (prev.orientation - 1) % 4;
            if (new_orientation == -1)
                new_orientation = 3;
        }

        return State(new_location, prev.timestep + 1, new_orientation);
    }





};

#else 
class CompetitionActionModelWithRotate
{
public:
    list<std::tuple<std::string,int,int,int>> errors;

    CompetitionActionModelWithRotate(){};
    CompetitionActionModelWithRotate(SharedEnvironment * env): env(env),rows(env->rows),cols(env->cols){
        moves[Action::R] = 1;
        moves[Action::D] = cols;
        moves[Action::L] = -1;
        moves[Action::U] = -cols;
        moves[Action::W] = 0;

    };

    bool is_valid(const vector<State>& prev, const vector<Action> & action);
    void set_logger(Logger* logger){this->logger = logger;}

    vector<State> result_states(const vector<State>& prev, const vector<Action> & action){
        vector<State> next(prev.size());
        for (size_t i = 0 ; i < prev.size(); i ++){
            next[i] = result_state(prev[i], action[i]);
        }
        return next;
    };
    
    vector<State> get_loc_neighbors(const State & curr,bool move_orient=true) {
        vector<State> neighbors;

        int x=curr.location%env->cols;
        int y=curr.location/env->cols;

        for (int i=0;i<4;++i) {
            int new_location=curr.location+moves[i];
            int new_orientation;
            
            if (move_orient) {
                new_orientation=i;
            } else {
                new_orientation=0;
            }

            if ((x==env->cols-1 && i==0) || (y==env->rows-1 && i==1) || (x==0 && i==2) || (y==0 && i==3)) {
                continue;
            }

            // if in the map but blocked
            if (env->map[new_location]) {
                continue;
            }

            neighbors.push_back(State(new_location,curr.timestep+1,new_orientation));
        }

        return neighbors;
    }

    SharedEnvironment * env=nullptr;
    int rows;
    int cols;
    int moves[5];
    Logger* logger = nullptr;
    
    State result_state(const State & prev, Action action)
    {
        // NOTE(rivers): we don't care about orientation, just keep it intact for now.

        int new_location = prev.location;
        int new_orientation = prev.orientation;

        if (action == Action::NA) {
            // std::cerr<<"have action NA in result state"<<std::endl;
        } else {
            new_location = new_location + moves[action];
        }

        return State(new_location, prev.timestep + 1, new_orientation);
    }
    
};

#endif