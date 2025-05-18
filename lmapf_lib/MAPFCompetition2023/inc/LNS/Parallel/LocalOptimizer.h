#pragma once
#include "LNS/PathTable.h"
#include "util/HeuristicTable.h"
#include "util/TimeLimiter.h"
#include "LNS/Parallel/TimeSpaceAStarPlanner.h"
#include "LaCAM2/instance.hpp"

namespace LNS {

namespace Parallel {

class LocalOptimizer
{
public:
    // TODO(rivers): think about what data structure needs a separate copy for each local optimizer.
    Instance & instance;
    PathTable path_table; // maintain a copy
    std::vector<Agent> agents; // remove in the future, currently we can visit it for agent id but not do anything else.
    std::shared_ptr<HeuristicTable> HT;
    std::shared_ptr<vector<float> > map_weights;
    std::shared_ptr<TimeSpaceAStarPlanner> path_planner;

    std::mt19937 MT; 

    std::shared_ptr<std::vector<LaCAM2::AgentInfo> > agent_infos;

    std::vector<Neighbor> updating_queue;

    string replan_algo_name;
    int window_size_for_CT;
    int window_size_for_CAT;
    int window_size_for_PATH;

    int screen=0;

    bool has_disabled_agents=false;

    LocalOptimizer(
        Instance & instance, std::vector<Agent> & agents, std::shared_ptr<HeuristicTable> HT, 
        std::shared_ptr<vector<float> > map_weights, std::shared_ptr<std::vector<LaCAM2::AgentInfo> > agent_infos,
        string replan_algo_name, bool sipp,
        int window_size_for_CT, int window_size_for_CAT, int window_size_for_PATH, int execution_window,
        bool has_disable_agents,
        int screen,
        int random_seed
    );

    // the global manager will push global changes to local optimizer though this function. 
    void update(Neighbor & neighbor);
    void optimize(Neighbor & neighbor, const TimeLimiter & time_limiter);
    void prepare(Neighbor & neighbor);

    bool runPP(Neighbor & neighbor, const TimeLimiter & time_limiter);

    void reset();

};

}

}