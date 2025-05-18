#include <random>
#include <vector>
#include <numeric>
#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <iostream>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include "TrajLNS.h"
#include "heuristics.hpp"

namespace TrafficMAPF {

// we have a better way to do this https://stackoverflow.com/a/53634099/24018910
std::vector<int> sort_with_probs(std::vector<float> probs, std::mt19937 & gen) {
    boost::random::discrete_distribution<> dist(probs);
    std::vector<int> samples;
    for (auto i = 0; i < probs.size(); i++) {
        int sample = dist(gen);
        samples.push_back(sample);
        probs[sample] = 0;
        dist = boost::random::discrete_distribution<>(probs);
    }

    return samples;
}

class PIBTSolver {
public:
    PIBTSolver(size_t random_seed=0): rng(random_seed) {}

    std::mt19937 rng;

    // locations: num_agents*2
    // heuristics: num_agents*action_dim, if an action is invalid, the heuristic should be < 0
    // action_choices: action_dim*2
    // map_size: (height, width)
    // occupied_by_now: loc_id -> agent_idx
    // occupied_by_next: loc_id -> agent_idx
    // next_locations: num_agents*2
    // return succ: bool
    bool select_action(
            int agent_idx,
            std::vector<int> & locations,
            std::vector<float> & heuristics,
            std::vector<int> & action_choices,
            std::vector<int> & map_size,
            std::unordered_map<int,int> & occupied_by_now,
            std::unordered_map<int,int> & occupied_by_next,
            std::vector<int> & next_locations,
            bool sampling
        ) {
        int y=locations[agent_idx*2];
        int x=locations[agent_idx*2+1];

        std::vector<int> ordered_actions;
        int num_actions=(int)action_choices.size()/2;
        if (sampling) {
            std::vector<float> probs;
            std::vector<int> actions;
            for (int action_idx=0;action_idx<num_actions;++action_idx) {
                int idx=agent_idx*num_actions+action_idx;
                float h=heuristics[idx];
                // if sampling, we regard heuristics as probablities
                if (h>0) {
                    probs.push_back(h);
                    actions.push_back(action_idx);
                }
            }
            // sampling
            auto ordered_indices=sort_with_probs(probs, rng);
            for (int i=0;i<ordered_indices.size();++i) {
                ordered_actions.push_back(actions[ordered_indices[i]]);
            }
        } else {
            std::vector<std::pair<float,int> > action_orders;
            for (int action_idx=0;action_idx<num_actions;++action_idx) {
                int idx=agent_idx*num_actions+action_idx;
                float h=heuristics[idx];
                // heuristic<0 means invalid action
                if (h>=0) {
                    action_orders.emplace_back(h,action_idx);
                }
            }
            std::sort(action_orders.begin(),action_orders.end());
            for (int i=0;i<action_orders.size();++i) {
                ordered_actions.push_back(action_orders[i].second);
            }
        }

        // std::cout<<"action orders for agent_idx"<<agent_idx<<std::endl;
        // for (auto &p:action_orders) {
        //     std::cout<<p.first<<" "<<p.second<<std::endl;
        // }

        for (int i=0;i<ordered_actions.size();++i) {
            int action_idx=ordered_actions[i];
            int dy=action_choices[action_idx*2];
            int dx=action_choices[action_idx*2+1];
            int ny=y+dy;
            int nx=x+dx;
            int nloc_id=ny*map_size[1]+nx;
            
            // vertex collision
            auto it=occupied_by_next.find(nloc_id);
            // the next location is taken by another agent
            if (it!=occupied_by_next.end()) continue;

            // edge collision
            auto it2=occupied_by_now.find(nloc_id);
            if (it2!=occupied_by_now.end() && it2->second!=agent_idx) {
                // the current location of another agent is the next location of the agent
                int another_agent_idx=it2->second;
                int another_ny=next_locations[another_agent_idx*2];
                int another_nx=next_locations[another_agent_idx*2+1];
                // the next location of another agent is the current location of the agent
                if (another_ny==y && another_nx==x) continue;
            }

            // reserve the next location for the agent
            occupied_by_next[nloc_id]=agent_idx;
            next_locations[agent_idx*2]=ny;
            next_locations[agent_idx*2+1]=nx;

            // priority inheritance
            // if the next location of the agent is currently taken by another different agent and that agent has not decided its next location
            // it needs to be pushed away.
            if (it2!=occupied_by_now.end() && it2->second!=agent_idx && next_locations[it2->second*2]<0) {
                bool succ=select_action(
                    it2->second,
                    locations,
                    heuristics,
                    action_choices,
                    map_size,
                    occupied_by_now,
                    occupied_by_next,
                    next_locations,
                    sampling
                );
                if (!succ) continue;
            }

            return true;
        }

        // if the agent fails to secure a next location, stay at its original location
        // this should be impossible if the actions contains the wait action.
        int loc_id=y*map_size[1]+x;
        occupied_by_next[loc_id]=agent_idx;
        next_locations[agent_idx*2]=y;
        next_locations[agent_idx*2+1]=x;

        return false;
    }

    // priorities: num_agents
    // locations: num_agents*2
    // heuristics: num_agents*action_dim, if an action is invalid, the heuristic should be < 0
    // action_choices: action_dim*2
    // map_size: (height, width)
    // return actions: num_agents
    std::vector<int> solve(
            TrafficMAPF::Dist2PathHeuristicTable & h_table,
            std::vector<float> & priorities, 
            std::vector<int> & locations, 
            std::vector<int> & action_choices,
            std::vector<int> & map_size,
            bool sampling
        ) {

        int num_agents=(int) priorities.size();
        int num_actions=(int) action_choices.size()/2;

        std::vector<std::pair<float,int> > agent_orders;
        for (int i=0;i<num_agents;++i) {
            // higher priority first, so we need the negation
            agent_orders.emplace_back(-priorities[i],i);
        }

        std::sort(agent_orders.begin(),agent_orders.end());

        // get heuristics
        std::vector<float> heuristics;
        for (int agent_idx=0;agent_idx<num_agents;++agent_idx) {
            int y=locations[agent_idx*2];
            int x=locations[agent_idx*2+1];
            // std::cout<<"y: "<<y<<" x: "<<x<<std::endl;
            for (int action_idx=0;action_idx<num_actions;++action_idx) {
                int dy=action_choices[action_idx*2];
                int dx=action_choices[action_idx*2+1];
                int ny=dy+y;
                int nx=dx+x;
                int next_loc = ny*map_size[1]+nx;
		        int min_heuristic;

                if (
                    ny<0 || ny>=map_size[0] ||
                    nx<0 || nx>=map_size[1] ||
                    h_table.lns.env->map[next_loc]) {
                    min_heuristic=-1;
                } else {
                    min_heuristic=h_table.get(agent_idx,next_loc);
                }
                heuristics.push_back((float) min_heuristic);          
            }
        }


        // initialize
        std::unordered_map<int,int> occupied_by_now;
        std::unordered_map<int,int> occupied_by_next;
        std::vector<int> next_locations(num_agents*2,-1);
        for (int i=0;i<num_agents;++i) {
            int y=locations[i*2];
            int x=locations[i*2+1];
            int loc_id=y*map_size[1]+x;
            occupied_by_now[loc_id]=i;
        }

        // dfs
        for (int i=0;i<num_agents;++i) {
            // plan based on the order
            int agent_idx = agent_orders[i].second;
            // if the next location has not been decided
            if (next_locations[agent_idx*2]<0) {
                bool succ=select_action(
                    agent_idx,
                    locations,
                    heuristics,
                    action_choices,
                    map_size,
                    occupied_by_now,
                    occupied_by_next,
                    next_locations,
                    sampling
                );
                if (!succ) {
                    std::cerr<<"a bug exists i: "<<i<<" agent_idx: "<<agent_idx<<std::endl;
                    exit(-1);
                }
            }
        }

        // get actions of locations and next_locations
        std::vector<int> actions;

        for (int i=0;i<num_agents;++i) {
            int y=locations[i*2];
            int x=locations[i*2+1];
            int ny=next_locations[i*2];
            int nx=next_locations[i*2+1];
            int dy=ny-y;
            int dx=nx-x;
            int j=0;
            for (;j<action_choices.size()/2;++j) {
                if (dy==action_choices[j*2] && dx==action_choices[j*2+1]) {
                    actions.push_back(j);
                    break;
                }
            }
            if (j>=action_choices.size()/2) {
                std::cerr<<"a bug exists"<<std::endl;
                exit(-1);
            }
        }

        return actions;
    }

    std::string playground(){
        return "hello, test!";
    }
};

}