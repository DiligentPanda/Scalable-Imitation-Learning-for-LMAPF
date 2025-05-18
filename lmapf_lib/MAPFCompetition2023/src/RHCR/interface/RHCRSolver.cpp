#include "RHCR/interface/RHCRSolver.h"
#include "States.h"
#include "ActionModel.h"
#include <chrono>
#include <thread>
#include "nlohmann/json.hpp"

namespace RHCR {

State convert_state_type(const ::State & state){
    return {state.location,state.timestep,state.orientation};
}

vector<State> convert_states_type(const vector<::State> & states){
    vector<State> new_states;
    for (int i=0;i<states.size();++i) {
        // TODO: rvalue and move constructor?
        new_states.push_back(convert_state_type(states[i]));
    }
    return new_states;
}

Action get_action_from_states(const State & state, const State & next_state, bool consider_rotation){
    assert(state.timestep+1==next_state.timestep);
    // TODO: assert consider_rotation

    if (consider_rotation){
        if (state.location==next_state.location){
            // either turn or wait
            if (state.orientation==next_state.orientation) {
                return Action::W;
            } else if ((state.orientation-next_state.orientation+4)%4==1) {
                return Action::CR;
            } else if ((state.orientation-next_state.orientation+4)%4==3) {
                return Action::CCR;
            } else {
                assert(false);
                return Action::W;
            }
        }
        else {
            return Action::FW;
        }
    } else {
        // TODO
        assert(false);
        return Action::W;
    }
}

void debug_agent(int agent,const vector<State> &starts,const vector< vector<pair<int, int> > > & goal_locations) {
    cerr<<"plan for agent "<<agent<<": ";
    cerr<<"start: "<<starts[agent].location<<". ";
    cerr<<"goals:";
    const vector<pair<int,int> > & goal_location=goal_locations[agent];
    for (int i=0;i<goal_location.size();++i){
        cerr<<" ["<<goal_location[i].second<<"]"<<goal_location[i].first;
    }
    cerr<<"."<<endl;
}

void debug_agent_path(int agent,const vector<Path> & paths,int start_timestep=0) {
    cerr<<"planed path for agent "<<agent<<" since timestep "<<start_timestep<<":";
    const Path & path=paths[agent];
    for (int i=start_timestep;i<paths[agent].size();++i){
        cerr<<" "<<path[i].location<<","<<path[i].orientation;
    }
    cerr<<endl;
}

void RHCRSolver::start_plan_task(){
    // stop_flag=false;
    // plan();
}

void RHCRSolver::stop_plan_task(){
    // if (!stop_flag) {
    //     stop_flag=true;
    //     task.join();
    // }
}

void RHCRSolver::initialize(const SharedEnvironment & env){
    graph.preprocessing(consider_rotation,env.file_storage_path);
    initialize_solvers();
    need_replan = true;
    timestep=0;
}

void RHCRSolver::update_goal_locations(const SharedEnvironment & env){
    for (int i=0;i<num_of_drives;++i){
        goal_locations[i].clear();
        for (int j=0;j<env.goal_locations[i].size();++j){
            // we require path of at least length one, even the start and the goal are the same.
            goal_locations[i].emplace_back(env.goal_locations[i][j].first,max(env.goal_locations[i][j].second-timestep,1));
        }
    }
}

void RHCRSolver::plan(const SharedEnvironment & env){
    // sleep for test purpose
    // std::this_thread::sleep_for (std::chrono::milliseconds(1000));

    auto _curr_states=convert_states_type(env.curr_states);
    if (paths.size()==0) {
        // initialize paths
        starts.resize(num_of_drives);
        goal_locations.resize(num_of_drives);
        paths.resize(num_of_drives);
        for (int i=0;i<num_of_drives;++i) {
            paths[i].push_back(_curr_states[i]);
        }
    } 

    if (need_replan) {
        // clear the previous plan
        for (int i=0;i<num_of_drives;++i) {
            paths[i].resize(timestep+1);
        }           

        update_start_locations();

        // TODO if we want to use hold_endpoints or dummy_paths, we need to adapt codes from KivaSystem::update_goal_locations()!
        update_goal_locations(env);

        // plan
        solve();

        total_feasible_timestep = timestep + simulation_window;

        // cerr<<paths[0].size()<<" "<<paths[1].size()<<endl;

        // int agent=117;
        // debug_agent(agent,starts,goal_locations);
        // debug_agent_path(agent,paths,5);
        // agent=4;
        // debug_agent(agent,starts,goal_locations);
        // debug_agent_path(agent,paths,5);
 
        cout<<"RHCRSolver solved for timestep "<<timestep<<endl;
    }
    else{
        cout<<"skip planning"<<endl;
    }
}

void RHCRSolver::get_step_actions(const SharedEnvironment & env, vector<Action> & actions){
    // check empty
    assert(actions.empty());

    for (int i=0;i<num_of_drives;++i) {
        // we will get action indexed at timestep+1
        if (paths[i].size()<=timestep+1){
            cerr<<"wierd error for agent "<<i<<". path length: "<<paths[i].size()<<", "<<"timestep+1: "<<timestep+1<<endl;
            assert(false);
        }
        actions.push_back(get_action_from_states(paths[i][timestep],paths[i][timestep+1],consider_rotation));
    }

    // TODO(hj) we probably still want to check the validness. so we need construct model or implement is_valid by ourselves.
    // check if not valid, this should not happen in general if the algorithm is correct? but maybe there exist deadlocks.
    // TODO(hj) detect deadlock?
    if (!model.is_valid(env.curr_states,actions)){
        cerr<<"planed actions are not valid in timestep "<<timestep+1<<"!"<<endl;
#ifdef DEV
        exit(-1);
#else
        actions.resize(num_of_drives, Action::W);
#endif
    } else {
        // NOTE(hj): only successfully executing a planned step will increase this internal timestep, which is different from the real timestep used in the simulation system.
        timestep+=1;
    }


    // TODO(hj): when we need to replan?
    need_replan=false;

    // 1. exceed simulation window
    if (timestep==total_feasible_timestep){
        need_replan=true;
    }
    
    // 2. goal changes: there different ways to check this. let's just keep the old goal and compare.
    for (int i=0;i<goal_locations.size();++i) {
        if (paths[i][timestep].location==goal_locations[i][0].first) {
            // arrive goal locations
            need_replan=true;
            break;
        }
    }
    
    // need_replan=true;
    
}
}