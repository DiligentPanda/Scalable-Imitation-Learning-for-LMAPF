
#ifndef heuristics_hpp
#define heuristics_hpp

#include "Types.h"
#include "utils.hpp"
#include <queue>
#include "TrajLNS.h"
#include "search_node.h"
#include "SharedEnv.h"
#include <omp.h>

namespace TrafficMAPF{

void init_flow_heuristic(TrajLNS& lns, std::vector<int>& traffic, int i);
void init_flow_heuristic(TrajLNS& lns, std::vector<int>& traffic, FlowHeuristic& ht, int i, int goal_loc);

s_node* get_flow_heuristic(FlowHeuristic& ht, SharedEnvironment* env, std::vector<int>& traffic, std::vector<Int4>& flow, int source);

void init_heuristic(HeuristicTable& ht, SharedEnvironment* env, int goal_location);


int get_heuristic(HeuristicTable& ht, SharedEnvironment* env, std::vector<int>& traffic, std::vector<Int4>& flow, int source);




void compute_dist_2_path(std::vector<int>& my_heuristic, SharedEnvironment* env, Traj& path);

void init_dist_2_path(Dist2Path& dp, SharedEnvironment* env, Traj& path, std::vector<int>& traffic);

int get_dist_2_path(Dist2Path& dp, SharedEnvironment* env, std::vector<int>& traffic, int source);

// a wrapper class
class Dist2PathHeuristicTable {
public:
    TrajLNS & lns;
    SharedEnvironment & env;
    std::vector<int> & traffic;
    std::shared_ptr<std::vector<float> > & map_weights;
    std::vector<omp_lock_t> agent_locks;
    int thread;

    Dist2PathHeuristicTable(
            TrajLNS & _lns, 
            std::vector<int> & _traffic, 
            std::shared_ptr<std::vector<float> > & _map_weights,
            int _thread=1
        ): 
        lns(_lns), 
        env(*(_lns.env)), 
        traffic(_traffic),
        map_weights(_map_weights),
        thread(_thread)
         {
        
        if (thread!=1) {
            agent_locks.resize(lns.env->num_of_agents);
            for (int agent_idx=0;agent_idx<lns.env->num_of_agents;++agent_idx) {
                omp_init_lock(&agent_locks[agent_idx]);
            }
        }

    }

    float get(int agent_idx, int loc) {
        int min_heuristic;

        if (thread!=1)
            omp_set_lock(&agent_locks[agent_idx]);

#ifdef FLOW_GUIDANCE
        if (!lns.flow_heuristics.empty() && !lns.flow_heuristics[loc].empty()){
            s_node* s= get_flow_heuristic(lns.flow_heuristics[loc], lns.env, traffic, lns.flow, neighbor);
            min_heuristic = s==nullptr? MAX_TIMESTEP : s->get_g();
        }
#else

        if (!lns.traj_dists.empty() && !lns.traj_dists[agent_idx].empty())
            min_heuristic = get_dist_2_path(lns.traj_dists[agent_idx], lns.env, traffic, loc);
#endif
        else if (!lns.heuristics[lns.tasks.at(agent_idx)].empty())
            min_heuristic = get_heuristic(lns.heuristics[lns.tasks.at(agent_idx)], lns.env, traffic, lns.flow, loc);
            // min_heuristic = heuristics.at(tasks.at(curr_id)).at(neighbor);
            // else if (!dh.empty())
            // 	min_heuristic = dh.get_heuristic(neighbor, tasks.at(curr_id));
        else
            min_heuristic = manhattanDistance(loc,lns.tasks.at(agent_idx),lns.env);   

        if (thread!=1)
            omp_unset_lock(&agent_locks[agent_idx]);
        
        return min_heuristic;
    }

    ~ Dist2PathHeuristicTable() {
        if (thread!=1) {
            for (int agent_idx=0;agent_idx<lns.env->num_of_agents;++agent_idx) {
                omp_destroy_lock(&agent_locks[agent_idx]);
            }
        }
    }

};

}
#endif