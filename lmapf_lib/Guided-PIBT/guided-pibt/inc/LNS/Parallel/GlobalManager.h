#pragma once
#include "LNS/Parallel/DataStructure.h"
#include "LNS/Parallel/NeighborGenerator.h"
#include "LNS/Parallel/LocalOptimizer.h"
#include "util/TimeLimiter.h"
#include <memory>
#include <omp.h>
#include <climits>
#include "heuristics.hpp"
#include <random>

namespace LNS {

namespace Parallel {

class GlobalManager
{
public:
    std::shared_ptr<NeighborGenerator> neighbor_generator;
    std::vector<std::shared_ptr<NeighborGenerator> > neighbor_generators;
    std::vector<std::shared_ptr<LocalOptimizer> > local_optimizers;

    float initial_sum_of_costs=MAX_TIMESTEP;
    float sum_of_costs=MAX_TIMESTEP;
    int num_of_failures=0;
    double average_group_size=0;
    float sum_of_distances = 0;
    int window_size_for_CT;
    int window_size_for_CAT;
    int window_size_for_PATH;
    list<IterationStats> iteration_stats;
    string init_algo_name;
    string replan_algo_name;
    Instance & instance;
    PathTable path_table;
    std::vector<Agent> agents;
    std::shared_ptr<TrafficMAPF::Dist2PathHeuristicTable> HT;
    std::shared_ptr<vector<float> > map_weights;
    int screen;
    bool verbose;
    int num_threads;
    int max_iterations;

    std::shared_ptr<std::vector<AgentInfo> > agent_infos;

    std::vector<std::vector<Neighbor>> updating_queues; // the generated neighbors for usage
    std::vector<omp_lock_t> updating_queue_locks;

    bool has_disabled_agents=false;

    bool async=false;

    GlobalManager(
        bool async,
        Instance & instance, std::shared_ptr<TrafficMAPF::Dist2PathHeuristicTable> HT, 
        std::shared_ptr<vector<float> > map_weights, std::shared_ptr<std::vector<AgentInfo> > agent_infos,
        int neighbor_size, destroy_heuristic destroy_strategy,
        bool ALNS, double decay_factor, double reaction_factor,
        string init_algo_name, string replan_algo_name, bool sipp,
        int window_size_for_CT, int window_size_for_CAT, int window_size_for_PATH, int execution_window,
        bool has_disabled_agents,
        bool fix_ng_bug,
        int screen,
        bool verbose=false,
        int num_threads=-1,
        int max_iterations=-1
    );

    ~GlobalManager();

    void getInitialSolution(Neighbor & neighbor);
    bool _run_async(TimeLimiter & time_limiter);
    bool _run(TimeLimiter & time_limiter);
    bool run(TimeLimiter & time_limiter);
    void update(Neighbor & neighbor, bool recheck);
    void update(Neighbor & neighbor);
    void reset();

    string getSolverName() const { return "LNS(" + init_algo_name + ";" + replan_algo_name + ")"; }

};

}

}