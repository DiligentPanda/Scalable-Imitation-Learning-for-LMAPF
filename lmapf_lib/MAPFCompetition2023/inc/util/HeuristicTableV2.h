#pragma once
#include <boost/heap/pairing_heap.hpp>
#include <boost/unordered_set.hpp>
#include <vector>
#include <iostream>
#include <cfloat>
#include <memory>

namespace HT_v2 {

constexpr float max_heuristic_val=FLT_MAX/16;

class State {
public:
    int loc;
    float g;
    float h;
    float f;

    State(int _loc, float _g, float _h):
        loc(_loc), g(_g), h(_h), f(_g+_h) {}

    void copy(const State *s) {
        loc = s->loc;
        g = s->g;
        h = s->h;
        f = s->f;
    }

    struct StateHash {
        inline std::size_t operator()(const State *s) const {
            return s->loc;
        }
    };

    struct StateEqualTo {
        inline bool operator()(const State *s1, const State *s2) const {
            return s1->loc==s2->loc;
        }
    };

    struct StateCompare {
        inline bool operator()(const State *s1, const State *s2) const {
            if (s1->f==s2->f) {
                if (s1->h==s2->h) {
                    return s1->h>s2->h;
                }
            }
            return s1->f>s2->f;
        }
    };

    bool closed=false;

    boost::heap::pairing_heap<State *, boost::heap::compare<State::StateCompare> >::handle_type open_handle;
};


class LazyBDHeuristicTable {
public:

    // should we use map here?
    const std::vector<int> & map;
    int rows;
    int cols;
    const std::vector<float> & map_weights;
    float min_map_weight;

    std::vector<float> h_vals;
    boost::heap::pairing_heap<State*,boost::heap::compare<State::StateCompare> > open_list;
    boost::unordered_set<State*,State::StateHash,State::StateEqualTo> all_states;

    int start_loc=-1;
    int goal_loc=-1;

    int sy,sx,gy,gx;

    // R,D,L,U
    const int dy[4]={0,1,0,-1};
    const int dx[4]={1,0,-1,0};

    LazyBDHeuristicTable(const std::vector<int> & _map, int _rows, int _cols, const std::vector<float> & _map_weights, float _min_map_weight=-1):
        map(_map), rows(_rows), cols(_cols), map_weights(_map_weights), h_vals(_rows*_cols, -1), min_map_weight(_min_map_weight) {
        // check map weights' size
        if (map_weights.size()!=_rows*_cols*5) {
            std::cerr<<"map weights' size is not equal to rows*cols*5 (r,d,l,u,wait)"<<std::endl;
            exit(1);
        }

        if (min_map_weight<0) {
            min_map_weight=max_heuristic_val;
            for (size_t i=0;i<map_weights.size();++i) {
                if (map_weights[i]<min_map_weight) {
                    min_map_weight=map_weights[i];
                }
            }
        }

    }

    float get_h(int y, int x) {
        return min_map_weight*(std::abs(y-sy)+std::abs(x-sx));
    }

    void update_start_and_goal(int _start_loc, int _goal_loc) {
        if (goal_loc==_goal_loc) {
            return;
        }

        // std::cout<<"update start and goal to "<<_start_loc<<" "<<_goal_loc<<std::endl;

        start_loc=_start_loc;
        goal_loc=_goal_loc;
        sy=start_loc/cols;
        sx=start_loc%cols;
        gy=goal_loc/cols;
        gx=goal_loc%cols;


        for (size_t i=0;i<h_vals.size();++i) {
            h_vals[i]=-1;
        }

        // clear memory
        open_list.clear();
        for (auto & s: all_states) {
            delete s;
        }
        all_states.clear();

        // push root node
        State *root = new State(goal_loc, 0, get_h(gy,gx));
        root->closed = false;
        root->open_handle=open_list.push(root);
        all_states.insert(root);
    }

    float get(int loc) {
        float h=max_heuristic_val;

        // we always assume all nodes are connected
        if (h_vals[loc]<0) {
            bool found=astar(loc);
            if (!found) {
                int y=loc/cols;
                int x=loc%cols;

                std::cerr<<"loc "<<loc<<"("<<x<<","<<y<<")"<<" is not reachable from goal "<<goal_loc<<"("<<gy<<","<<gx<<")"<<std::endl;
                std::cerr<<map[loc]<<" "<<map[goal_loc]<<std::endl;
                exit(1);
            }
        }
        
        if (h_vals[loc]>=0) {
            h = h_vals[loc];
        }

        return h;
    }

    bool astar(int loc) {
        bool ret=false;

        while (!open_list.empty()) {
            State *cur = open_list.top();

            // record the shortest path to goal
            h_vals[cur->loc]=cur->g;

            if (cur->loc==loc) {
                ret=true;
                break;
            }

            // we cannot pop before the if line because we need to expand its neighbors
            open_list.pop();
            cur->closed=true;

            int r=cur->loc/cols;
            int c=cur->loc%cols;

            for (int i=0;i<4;++i) {
                int nr=r+dy[i];
                int nc=c+dx[i];
                int nloc=nr*cols+nc;

                if (nr>=0 && nr<rows && nc>=0 && nc<cols && map[nloc]!=1) {

                    // NOTE: we need to get reverse idx and cost
                    int ridx=(i+2)%4;
                    float rcost=map_weights[nloc*5+ridx];
                    // std::cout<<nloc<<"->"<<cur->loc<<" rcost "<<rcost<<std::endl;
                    float g=cur->g+rcost;
                    float h=get_h(nr,nc);

                    State *ns = new State(nloc, g, h);
                    auto it = all_states.find(ns);
                    if (it==all_states.end()) {
                        if (h_vals[nloc]>=0) {
                            std::cout<<nloc<<" sss "<<h_vals[nloc]<<std::endl;
                        }
                        ns->closed = false;
                        ns->open_handle=open_list.push(ns);
                        all_states.insert(ns);
                    } else {
                        auto & old_state=*it;
                        if (ns->g < old_state->g) {
                            old_state->copy(ns);
                            if (old_state->closed) {
                                // reopen: should no happen if we use consistent heuristic, but there might be some numerical issue
                                // let's just not reopen.
                                // old_state->closed=false;
                                // open_list.update(old_state->open_handle);
                                // std::cerr<<"diff: "<<old_state->g<<" "<<ns->g<<std::endl;
                                // std::cerr<<"reopen"<<std::endl;
                                // exit(1);
                            } else {
                                // increase because we have a better new state
                                open_list.update(old_state->open_handle);
                            }
                        }
                        delete ns;
                    }
                }
            }
        }

        return ret;

    }

};


class HeuristicTableV2 {
public:

    const std::vector<int> & map;
    int rows;
    int cols;
    const std::vector<float> & map_weights;
    float min_map_weight;

    std::vector<LazyBDHeuristicTable> heuristic_tables;

    HeuristicTableV2(int num_of_agents, const std::vector<int> & _map, int _rows, int _cols, const std::vector<float> & _map_weights, float _min_map_weight=-1):
        map(_map), rows(_rows), cols(_cols), map_weights(_map_weights), min_map_weight(_min_map_weight) {
        // check map weights' size
        if (map_weights.size()!=_rows*_cols*5) {
            std::cerr<<"map weights' size is not equal to rows*cols*5 (r,d,l,u,wait)"<<std::endl;
            exit(1);
        }

        if (min_map_weight<0) {
            min_map_weight=max_heuristic_val;
            for (size_t i=0;i<map_weights.size();++i) {
                if (map_weights[i]<min_map_weight) {
                    min_map_weight=map_weights[i];
                }
            }
        }

        for (int i=0;i<num_of_agents;++i) {
            heuristic_tables.emplace_back(map, rows, cols, map_weights, min_map_weight);
        }
    }

    void update_start_and_goal(int agent_idx, int start_loc, int goal_loc) {
        heuristic_tables[agent_idx].update_start_and_goal(start_loc, goal_loc);
    }

    float get(int agent_idx, int loc) {
        return heuristic_tables[agent_idx].get(loc);
    }

};

}; // namespace HT_v2