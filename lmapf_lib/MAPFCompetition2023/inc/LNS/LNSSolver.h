#pragma once
#include "util/CompetitionActionModel.h"
#include <random>
#include <memory>
#include "util/HeuristicTable.h"
#include "LaCAM2/executor.hpp"
#include "LaCAM2/slow_executor.hpp"
#include "SharedEnv.h"
#include "common.h"
#include <unordered_set>
#include <queue>
#include "LaCAM2/LaCAM2Solver.hpp"
#include "LNS/Parallel/GlobalManager.h"
#include "util/StatsTree.h"
#include "LaCAM2/instance.hpp"

namespace LNS {

class LNSSolver {

public:
    // [start] the following lines should be abstracted away.
    // std::vector<::Path> paths;
    CompetitionActionModelWithRotate action_model;
    std::mt19937* MT;   // seed for randomness
    // bool need_replan = true;
    int total_feasible_timestep = 0;
    int timestep = 0;
    // int executed_plan_step = -1;
    void initialize(const SharedEnvironment & env);
    void observe(const SharedEnvironment & env);
    void plan(const SharedEnvironment & env);
    void get_step_actions(const SharedEnvironment & env, vector<Action> & actions);
    // Action get_action_from_states(const State & state, const State & next_state);
    // [end]

    std::shared_ptr<Parallel::GlobalManager> lns;
    std::shared_ptr<Instance> instance;
    std::shared_ptr<HeuristicTable> HT; // instance
    std::shared_ptr<std::vector<float> > map_weights; // map weights

    std::shared_ptr<vector<LaCAM2::AgentInfo> > agent_infos;

    LaCAM2::Executor executor;
    LaCAM2::SlowExecutor slow_executor;
    std::shared_ptr<LaCAM2::LaCAM2Solver> lacam2_solver;
    // std::set<int> agent_ids_need_replan;

    std::shared_ptr<StatsTree> obstacle_stats_tree;
    std::shared_ptr<StatsTree> agent_stats_tree;

    nlohmann::json config;

    double max_plan_time=0;

    std::vector<::Path> planning_paths;
    std::vector<::Path> execution_paths;
    int executed_step=0;
    bool need_new_execution_paths=false;
    int execution_window; // TODO: read it from config
    int planning_window; // TODO: read it from config

    int num_task_completed=0;
    int max_task_completed;

    LNSSolver(
        const std::shared_ptr<HeuristicTable> & HT,
        SharedEnvironment * env,
        std::shared_ptr<std::vector<float> > & map_weights, // map weights
        nlohmann::json & config,
        std::shared_ptr<LaCAM2::LaCAM2Solver> & lacam2_solver,
        int max_task_completed
    );

    ~LNSSolver(){
        delete MT;
    };

    Action get_action_from_states(const State & state, const State & next_state){
#ifndef NO_ROT
        assert(state.timestep+1==next_state.timestep);
        
        if (state.location==next_state.location){
            // either turn or wait
            if (state.orientation==next_state.orientation) {
                return Action::W;
            } else if ((state.orientation-next_state.orientation+4)%4==3) {
                return Action::CR;
            } else if ((state.orientation-next_state.orientation+4)%4==1) {
                return Action::CCR;
            } else {
                assert(false);
                return Action::W;
            }
            }
        else {
            return Action::FW;
        }
#else 
        assert(state.timestep+1==next_state.timestep);
        
        for (int i=0;i<5;++i) {
            if (state.location+action_model.moves[i]==next_state.location) {
                return static_cast<Action>(i);
            }
        }

        cerr<<"Cannot get action from invalid movement between state"<<state<<" and "<<next_state<<endl;
        exit(-1);
#endif
    }

    void modify_goals(vector<int> & goals, const SharedEnvironment & env) {
        vector<bool> taken(env.map.size(), false);
        for (auto & goal: goals) {
            if (!taken[goal]) {
                taken[goal]=true;
            } else {
                // try to find a location that is close enough
                int new_goal = find_new_goal(goal, taken, env);
                taken[new_goal]=true;
                goal=new_goal;
            }
        }
    }

    int find_new_goal(int old_goal, const vector<bool> & taken, const SharedEnvironment & env) {
        std::queue<int> q;
        vector<bool> visited(env.map.size(), false);
        int start=old_goal;
        visited[start] =true;
        q.push(start);
        while (!q.empty()) {
            int curr = q.front();
            q.pop();

            int x=curr%env.cols;
            int y=curr/env.cols;

            if (x<env.cols-1) {
                int next = curr+1;
                if (!env.map[next]){
                    if (!taken[next]) return next;
                    if (!visited[next]) {
                        visited[next]=true;
                        q.push(next);
                    }
                } 
            }

            if (y<env.rows-1) {
                int next = curr+env.cols;
                if (!env.map[next]){
                    if (!taken[next]) return next;
                    if (!visited[next]) {
                        visited[next]=true;
                        q.push(next);
                    }
                } 
            }

            if (x>0) {
                int next = curr-1;
                if (!env.map[next]){
                    if (!taken[next]) return next;
                    if (!visited[next]) {
                        visited[next]=true;
                        q.push(next);
                    }
                } 
            }

            if (y>0) {
                int next = curr-env.cols;
                if (!env.map[next]){
                    if (!taken[next]) return next;
                    if (!visited[next]) {
                        visited[next]=true;
                        q.push(next);
                    }
                } 
            }
        }

        return -1;
    }

};


} // end namespace LNS
