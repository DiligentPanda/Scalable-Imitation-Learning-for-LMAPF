#pragma once
#include "common.h"
#include "LNS/Instance.h"
#include "heuristics.hpp"

namespace LNS {

namespace Parallel {

struct AgentInfo {
public:
    int goal_location;
    float elapsed;
    float tie_breaker;
    int id;
    int stuck_order;
    bool disabled;

    AgentInfo():id(-1),goal_location(-1),elapsed(-1),tie_breaker(-1), stuck_order(0), disabled(false) {};
};

struct IterationStats
{
    int sum_of_costs;
    double runtime;
    int num_of_agents;
    string algorithm;
    int sum_of_costs_lowerbound;
    int num_of_colliding_pairs;
    IterationStats(int num_of_agents, int sum_of_costs, double runtime, const string& algorithm,
                   int sum_of_costs_lowerbound = 0, int num_of_colliding_pairs = 0) :
            num_of_agents(num_of_agents), sum_of_costs(sum_of_costs), runtime(runtime),
            sum_of_costs_lowerbound(sum_of_costs_lowerbound), algorithm(algorithm),
            num_of_colliding_pairs(num_of_colliding_pairs) {}
};

// enum constraint_type { LEQLENGTH, GLENGTH, RANGE, BARRIER, VERTEX, EDGE, POSITIVE_VERTEX, POSITIVE_EDGE, CONSTRAINT_COUNT};
enum destroy_heuristic { RANDOMAGENTS, RANDOMWALK, INTERSECTION, DESTORY_COUNT };

struct PathEntry {
    int location;
    int orientation;
    PathEntry(int location=-1, int orientation=-1): location(location), orientation(orientation) {}
};

struct Path {
    std::vector<PathEntry> nodes; // pos, orient
    float path_cost;
    inline void clear() {
        path_cost=0;
        nodes.clear();
    }

    inline PathEntry & operator [] (int i) { return nodes[i];}
    inline size_t size() {return nodes.size();}
    inline bool empty() {return nodes.empty();}

    inline PathEntry & back() {return nodes.back();}
    inline PathEntry & front() {return nodes.front();}

};

std::ostream & operator << (std::ostream &out, const PathEntry &pe);
std::ostream & operator << (std::ostream &out, const Path &p);

struct Agent
{
    int id;
    Path path;
    const Instance & instance;
    std::shared_ptr<TrafficMAPF::Dist2PathHeuristicTable> HT; // instance

    std::shared_ptr<std::vector<AgentInfo> > agent_infos;

    Agent(int id, const Instance& instance, std::shared_ptr<TrafficMAPF::Dist2PathHeuristicTable> & HT, std::shared_ptr<std::vector<AgentInfo> > &agent_infos): 
        id(id), HT(HT), instance(instance), agent_infos(agent_infos) {

    }

    Agent(const Agent & agent):
        id(agent.id), HT(agent.HT), instance(agent.instance), agent_infos(agent.agent_infos), path(agent.path) {

    }

    inline int getStartLocation() {return instance.start_locations[id];}
    inline int getStartOrientation() {return instance.start_orientations[id];}
    inline int getGoalLocation() {return instance.goal_locations[id];}


    static inline float get_action_cost(int pst, int ost, int ped, int oed, std::shared_ptr<TrafficMAPF::Dist2PathHeuristicTable> & HT) {
        auto & map_weights=*(HT->map_weights);

        int offset=ped-pst;
        if (offset==0) {
            // stay cost
            return map_weights[pst*5+4];
        } else if (offset==1) {
            // east
            return map_weights[pst*5+0];
        } else if (offset==HT->env.cols) {
            // south
            return map_weights[pst*5+1];
        } else if (offset==-1) {
            // west
            return map_weights[pst*5+2];
        } else if (offset==-HT->env.cols) {
            // north
            return map_weights[pst*5+3];
        } else {
            std::cerr<<"invalid move"<<endl;
            exit(-1);
        }
    }

    inline float getEstimatedPathLength(Path & path, int goal_location, std::shared_ptr<TrafficMAPF::Dist2PathHeuristicTable> & HT, bool arrival_break=false, int T=-1) {
        if ((*agent_infos)[id].disabled) { // if disabled, we just set its estimated path length to 0. namely ignore it anyway.
            return 0;
        }

        int max_steps=(int)path.size()-1;
        if (T!=-1) {
            max_steps=T;
        }

        // TODO(rivers): this is actually path cost, not path length
        float cost=0;
        bool arrived=false;
        for (int i=0;i<max_steps;++i) {
            cost+=get_action_cost(path[i].location,path[i].orientation,path[i+1].location,path[i+1].orientation, HT);
            if (path[i].location==goal_location) {
                arrived=true;
                if (arrival_break) {
                    break;
                }
            }
        }

        if (arrived || T!=-1) {
            return cost;
        } else {
#ifndef NO_ROT
            return cost+HT->get(path.back().location, path.back().orientation, goal_location);
#else
            return cost+HT->get(id, path.back().location);
#endif
        }
    }

    inline float getNumOfDelays() {
#ifndef NO_ROT
        // TODO(rivers): we may need two heuristic table: one for cost, one for path length estimation.
        return getEstimatedPathLength(path,instance.goal_locations[id],HT) - HT->get(instance.start_locations[id],instance.start_orientations[id],instance.goal_locations[id]);
#else
        return getEstimatedPathLength(path,instance.goal_locations[id],HT) - HT->get(id, instance.start_locations[id]);
#endif
    }

    void reset() {
        path.clear();
    }
};

struct Neighbor
{
    vector<int> agents;
    float sum_of_costs;
    float old_sum_of_costs;
    std::map<int, Path> m_paths; // for temporally storing the new paths. may change to vector later, agent id -> path
    // set<pair<int, int>> colliding_pairs;  // id1 < id2
    // set<pair<int, int>> old_colliding_pairs;  // id1 < id2
    // std::vector<Path> old_paths;
    std::map<int, Path> m_old_paths; // for temporally storing the old paths. may change to vector later, agent id -> path
    bool succ = false;
    int selected_neighbor;

    float num_arrived;
    float old_num_arrived;
};

}

} // namespace LNS