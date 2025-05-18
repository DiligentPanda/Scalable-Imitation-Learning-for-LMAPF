
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <iostream>
#include <set>

class STOPSolver {
public:
    STOPSolver(
        size_t _random_seed, 
        std::vector<int> & _graph, // H*W
        int _height, 
        int _width,
        std::vector<int> & _action_choices // num_actions * 2
    ): rng(_random_seed), graph(_graph), height(_height), width(_width), action_choices(_action_choices) {
        num_actions=(int) action_choices.size()/2;
        bool found=false;
        for (int i=0;i<num_actions;++i) {
            if (action_choices[2*i]==0 && action_choices[2*i+1]==0) {
                wait_action=i;
                found=true;
                break;
            }
        }
        if (!found) {
            std::cerr << "No wait action found" << std::endl;
            exit(-1);
        }
    }

    std::mt19937 rng;
    std::vector<int> graph;
    int height;
    int width;

    std::vector<int> action_choices;
    int wait_action;
    int num_actions;

    std::vector<int> solve(
        std::vector<int> & locations, // num_agents*2, [[y,x]]
        std::vector<int> & guiding_actions // num_agents
    ) {
        std::vector<int> actions=guiding_actions;
        std::vector<int> next_locations(locations.size(), 0);
        std::vector<int> sim_state(height*width,0);



        // for (int i=0;i<height*width;++i) {
        //     std::cout<<"sim_state "<<i<<" "<<sim_state[i]<<std::endl;
        // }


        int num_agents= (int) guiding_actions.size(); 

        if (num_agents*2!=locations.size()) {
            std::cerr << "Number of locations does not match number of agents" << std::endl;
            exit(-1);
        }

        for (int i=0;i<num_agents;++i) {
            int y=locations[2*i];
            int x=locations[2*i+1];
            int action=actions[i];

            int dy=action_choices[2*action];
            int dx=action_choices[2*action+1];

            int next_y=y+dy;
            int next_x=x+dx;

            if (next_y<0 || next_y>=height || next_x<0 || next_x>=width || graph[next_y*width+next_x]!=0) {
                next_y=y;
                next_x=x;
                actions[i]=wait_action;
            }

            // std::cout<<i<<" next_y "<<next_y<<" next_x "<<next_x<<std::endl;

            sim_state[next_y*width+next_x]+=1;
            next_locations[2*i]=next_y;
            next_locations[2*i+1]=next_x;
        }


        // for (int i=0;i<height*width;++i) {
        //     std::cout<<"sim_state "<<i<<" "<<sim_state[i]<<std::endl;
        // }

        // solve swap collision first,beacuse solve swap collision will generate new vertex collision, and solve vertex collisions will not gernate swap collsions
        // resolve swap collisions
        std::vector<std::pair<int,int> > swap_collisions;
        for (int i=0;i<num_agents;++i) {
            for (int j=i+1;j<num_agents;++j) {
                if (next_locations[2*i]==locations[2*j] && next_locations[2*i+1]==locations[2*j+1] && next_locations[2*j]==locations[2*i] && next_locations[2*j+1]==locations[2*i+1]) {
                    swap_collisions.emplace_back(i,j);
                    // std::cout<<"swap collision "<<i<<" "<<j<<std::endl;
                }
            }
        }

        for (auto & collision: swap_collisions) {
            int i=collision.first;
            sim_state[next_locations[2*i]*width+next_locations[2*i+1]]-=1;
            sim_state[locations[2*i]*width+locations[2*i+1]]+=1;
            next_locations[2*i]=locations[2*i];
            next_locations[2*i+1]=locations[2*i+1];
            actions[i]=wait_action;

            int j=collision.second;
            sim_state[next_locations[2*j]*width+next_locations[2*j+1]]-=1;
            sim_state[locations[2*j]*width+locations[2*j+1]]+=1;
            next_locations[2*j]=locations[2*j];
            next_locations[2*j+1]=locations[2*j+1];
            actions[j]=wait_action;
        }


        // for (int i=0;i<height*width;++i) {
        //     std::cout<<"sim_state "<<i<<" "<<sim_state[i]<<std::endl;
        // }


        // resolve vertex collisions
        while (true) {
            std::vector<int> colliding_indices;
            for (int idx=0;idx<height*width;++idx) {
                if (sim_state[idx]>1) {
                    colliding_indices.push_back(idx);
                    // std::cout<<"colliding_indices "<<idx<<" "<<sim_state[idx] << std::endl;
                }
            }

            // std::cout<<"colliding_indices.size() "<<colliding_indices.size()<<std::endl;

            if (colliding_indices.empty()) {
                break;
            }

            for (auto & colliding_idx: colliding_indices) {
                std::set<int> colliding_agents;
                for (int i=0;i<num_agents;++i) {
                    if (next_locations[2*i]==colliding_idx/width && next_locations[2*i+1]==colliding_idx%width) {
                        colliding_agents.insert(i);
                    }
                }

                if (colliding_agents.size()>1) {
                    bool agent_wait=false;
                    for (auto & agent: colliding_agents) {
                        if (actions[agent]==wait_action) {
                            agent_wait=true;
                            colliding_agents.erase(agent);
                            break;
                        }
                    }

                    if (!agent_wait) {
                        int i=(int)(rng()%colliding_agents.size());
                        auto it=colliding_agents.begin();
                        std::advance(it,i);
                        colliding_agents.erase(it);
                    }

                    for (auto & agent: colliding_agents) {
                        actions[agent]=wait_action;
                        sim_state[next_locations[2*agent]*width+next_locations[2*agent+1]]-=1;
                        sim_state[locations[2*agent]*width+locations[2*agent+1]]+=1;
                        next_locations[2*agent]=locations[2*agent];
                        next_locations[2*agent+1]=locations[2*agent+1];
                    }

                } else {
                    std::cerr<<"Error: colliding agents size "<<colliding_agents.size()<<" is not greater than 1"<<std::endl;
                    exit(-1);
                }
            }
        }


        for (int i=0;i<height*width;++i) {
            if (sim_state[i]>1 || sim_state[i]<0) {
                std::cerr<<"Error: sim_state "<<i<<" "<<sim_state[i]<<std::endl;
                exit(-1);
            }
        }


        return actions;

    }


    // std::vector<int> solve2(
    //     std::vector<int> & locations, // #A*2
    //     std::vector<int> & guiding_actions // #A
    // ) {
    //     std::vector<int> actions=guiding_actions;
    //     std::vector<int> next_locations(locations.size(), 0);
    //     std::unordered_map<int,std::set<int> > sim_state; // loc_id -> set of agent_idx

    //     // for (int i=0;i<height*width;++i) {
    //     //     std::cout<<"sim_state "<<i<<" "<<sim_state[i]<<std::endl;
    //     // }

    //     int num_agents = (int) guiding_actions.size(); 

    //     for (int i=0;i<num_agents;++i) {
    //         int y=locations[2*i];
    //         int x=locations[2*i+1];
    //         int loc=y*width+x;

    //         sim_state[loc]=std::set<int>();

    //     }

    //     if (num_agents*2!=locations.size()) {
    //         std::cerr << "Number of locations does not match number of agents" << std::endl;
    //         exit(-1);
    //     }

    //     for (int i=0;i<num_agents;++i) {
    //         int y=locations[2*i];
    //         int x=locations[2*i+1];
    //         int action=actions[i];

    //         int dy=action_choices[2*action];
    //         int dx=action_choices[2*action+1];

    //         int next_y=y+dy;
    //         int next_x=x+dx;

    //         if (next_y<0 || next_y>=height || next_x<0 || next_x>=width || graph[next_y*width+next_x]!=0) {
    //             next_y=y;
    //             next_x=x;
    //             actions[i]=wait_action;
    //         }

    //         // std::cout<<i<<" next_y "<<next_y<<" next_x "<<next_x<<std::endl;

    //         int next_loc=next_y*width+next_x;
    //         if (sim_state.find(next_loc)==sim_state.end()) {
    //             sim_state[next_loc]=std::set<int>();
    //         }
    //         sim_state[next_loc].insert(i);
    //         next_locations[2*i]=next_y;
    //         next_locations[2*i+1]=next_x;
    //     }


    //     // for (int i=0;i<height*width;++i) {
    //     //     std::cout<<"sim_state "<<i<<" "<<sim_state[i]<<std::endl;
    //     // }

    //     // solve swap collision first,beacuse solve swap collision will generate new vertex collision, and solve vertex collisions will not gernate swap collsions
    //     // resolve swap collisions
    //     std::vector<std::pair<int,int> > swap_collisions;
    //     for (int i=0;i<num_agents;++i) {
    //         for (int j=i+1;j<num_agents;++j) {
    //             if (next_locations[2*i]==locations[2*j] && next_locations[2*i+1]==locations[2*j+1] && next_locations[2*j]==locations[2*i] && next_locations[2*j+1]==locations[2*i+1]) {
    //                 swap_collisions.emplace_back(i,j);
    //                 // std::cout<<"swap collision "<<i<<" "<<j<<std::endl;
    //             }
    //         }
    //     }

    //     for (auto & collision : swap_collisions) {
    //         int i=collision.first;
    //         int next_loc_i=next_locations[2*i]*width+next_locations[2*i+1];
    //         int loc_i=locations[2*i]*width+locations[2*i+1];
    //         sim_state[next_loc_i].erase(i);
    //         sim_state[loc_i].insert(i);
    //         next_locations[2*i]=locations[2*i];
    //         next_locations[2*i+1]=locations[2*i+1];
    //         actions[i]=wait_action;

    //         int j=collision.second;
    //         int next_loc_j=next_locations[2*j]*width+next_locations[2*j+1];
    //         int loc_j=locations[2*j]*width+locations[2*j+1];
    //         sim_state[next_loc_j].erase(j);
    //         sim_state[loc_j].insert(j);
    //         next_locations[2*j]=locations[2*j];
    //         next_locations[2*j+1]=locations[2*j+1];
    //         actions[j]=wait_action;
    //     }


    //     // for (int i=0;i<height*width;++i) {
    //     //     std::cout<<"sim_state "<<i<<" "<<sim_state[i]<<std::endl;
    //     // }


    //     // resolve vertex collisions
    //     while (true) {
    //         std::vector<int> colliding_indices;
    //         for (auto & pair: sim_state) {
    //             if (pair.second.size()>1) {
    //                 colliding_indices.push_back(pair.first);
    //                 // std::cout<<"colliding_indices "<<idx<<" "<<sim_state[idx] << std::endl;
    //             }
    //         }

    //         // std::cout<<"colliding_indices.size() "<<colliding_indices.size()<<std::endl;

    //         if (colliding_indices.empty()) {
    //             break;
    //         }

    //         for (auto & colliding_idx: colliding_indices) {
    //             std::set<int> colliding_agents=sim_state[colliding_idx];

    //             if (colliding_agents.size()>1) {
    //                 bool agent_wait=false;
    //                 for (auto & agent: colliding_agents) {
    //                     if (actions[agent]==wait_action) {
    //                         agent_wait=true;
    //                         colliding_agents.erase(agent);
    //                         break;
    //                     }
    //                 }

    //                 if (!agent_wait) {
    //                     int i=(int)(rng()%colliding_agents.size());
    //                     auto it=colliding_agents.begin();
    //                     std::advance(it,i);
    //                     colliding_agents.erase(it);
    //                 }

    //                 for (auto & agent: colliding_agents) {
    //                     actions[agent]=wait_action;
    //                     sim_state[next_locations[2*agent]*width+next_locations[2*agent+1]].erase(agent);
    //                     sim_state[locations[2*agent]*width+locations[2*agent+1]].insert(agent);
    //                     next_locations[2*agent]=locations[2*agent];
    //                     next_locations[2*agent+1]=locations[2*agent+1];
    //                 }

    //             } else {
    //                 std::cerr<<"Error: colliding agents size "<<colliding_agents.size()<<" is not greater than 1"<<std::endl;
    //                 exit(-1);
    //             }
    //         }
    //     }

    //     return actions;

    // }


    std::string playground(){
        return "hello, test!";
    }

};


PYBIND11_MODULE(py_STOP, m) {
	// optional module docstring
    // m.doc() = ;
    pybind11::class_<STOPSolver>(m, "STOPSolver")
        .def(pybind11::init<size_t, std::vector<int> &, int, int, std::vector<int> &> ())
        .def("solve", &STOPSolver::solve)
        // .def("solve2", &STOPSolver::solve2)
        .def("playground", &STOPSolver::playground);
}
