#ifndef TRAJ_LNS_H
#define TRAJ_LNS_H

#include "Types.h"
#include "Memory.h"
#include "search_node.h"
#include "heap.h"
#include <random>
#include <set>

namespace TrafficMAPF{
enum ADAPTIVE {RANDOM, CONGESTION, COUNT};

struct FlowHeuristic{
    HeuristicTable* h; 
    int target;
    int origin;
    pqueue_min_of open;
    MemoryPool mem;


    bool empty(){
        return mem.generated() == 0;
    }
    void reset(){
        // op_flows.clear();
        // depths.clear();
        // dists.clear();
        open.clear();
        mem.reset();
    }

};

class TrajLNS{
    public:
    SharedEnvironment* env;
    std::vector<int> tasks;

    TimePoint start_time;
    int t_ms=0;


    std::vector<Traj> trajs;
    std::vector<Int4> flow;
    std::vector<HeuristicTable> heuristics;
    std::vector<Dist2Path> traj_dists;
    std::vector<s_node> goal_nodes;// store the goal node of single agent search for each agent. contains all cost information.
    // DH dh;
    std::vector<FlowHeuristic> flow_heuristics;

    std::vector<double> weights; //weights for adaptive lns

    double decay_factor = 0.001;
    double reaction_factor = 0.1;

    int group_size = LNS_GROUP_SIZE;


    std::vector<std::set<int>> occupations;
    std::vector<bool> tabu_list;
    int num_in_tablu=0;

    int traj_inited = 0;
    int dist2path_inited = 0;
    int tdh_build = 0;

    int op_flow = 0;
    int vertex_flow = 0;
    int soc = 0;

    MemoryPool mem;

    std::mt19937 rng;

    void init_mem(){
        mem.init(env->map.size());
    }

    void seed(uint _seed) {
        rng.seed(_seed);
    }

    TrajLNS(SharedEnvironment* env, uint seed):
        env(env),
        rng(seed),
        trajs(env->num_of_agents), tasks(env->num_of_agents),tabu_list(env->num_of_agents,false),
        flow(env->map.size(),Int4({0,0,0,0})), heuristics(env->map.size()),
        flow_heuristics(env->num_of_agents),
        traj_dists(env->num_of_agents),goal_nodes(env->num_of_agents),occupations(env->map.size()){
            weights.resize(ADAPTIVE::COUNT,1.0);
        };

    TrajLNS(): rng(10) {};
};
}
#endif