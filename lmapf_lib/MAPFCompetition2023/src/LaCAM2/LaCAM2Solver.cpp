#include "LaCAM2/LaCAM2Solver.hpp"
#include "util/MyLogger.h"
#include "LaCAM2/post_processing.hpp"
#include "LaCAM2/utils.hpp"
#include "LaCAM2/SUO2/SpatialSUO.h"
#include <omp.h>
#include <thread>
// #include "util/StatsTree.h"

namespace LaCAM2 {

std::set<int> load_tabu_locs(string fp) {
    std::ifstream f(fp);
    try
    {
        auto tabu_locs = nlohmann::json::parse(f);
        std::set<int> tabu_locs_set;
        for (int i=0;i<tabu_locs.size();++i) {
            int loc=tabu_locs[i].get<int>();
            tabu_locs_set.insert(loc);
        }

        return tabu_locs_set;
    }
    catch (nlohmann::json::parse_error error)
    {
        std::cout << "Failed to load " << fp << std::endl;
        std::cout << "Message: " << error.what() << std::endl;
        exit(1);
    }
}

void LaCAM2Solver::initialize(const SharedEnvironment & env) {
    paths.resize(env.num_of_agents);
    agent_infos=std::make_shared<std::vector<AgentInfo> >(env.num_of_agents);
    for (int i=0;i<env.num_of_agents;++i) {
        (*agent_infos)[i].id=i;
    }
    // action_costs.resize(env.num_of_agents);
    // total_actions.resize(env.num_of_agents);
    G = std::make_shared<Graph>(env);
}

void LaCAM2Solver::disable_agents(const SharedEnvironment & env) {

    string strategy=read_param_json<string>(config,"disable_agent_strategy");

    int disabled_agents_num;
    if (strategy=="uniform") {
        // random select some agents to be disabled
        std::vector<int> agents_ids;
        for (int i=0;i<env.num_of_agents;++i) {
            agents_ids.push_back(i);
        }
        std::shuffle(agents_ids.begin(),agents_ids.end(),*MT);

        disabled_agents_num=env.num_of_agents-max_agents_in_use;
        for (int i=0;i<disabled_agents_num;++i) {
            (*agent_infos)[agents_ids[i]].disabled=true;
        }
    } else if (strategy=="tabu_locs") {
        // disable agents not in the tabu locs set
        string tabu_locs_fp=read_param_json<string>(config,"tabu_locs_fp");
        auto tabu_locs = load_tabu_locs(tabu_locs_fp);

        std::vector<int> agents_ids;
        for (int i=0;i<env.num_of_agents;++i) {
            if (tabu_locs.find(env.curr_states[i].location)==tabu_locs.end()) {
                agents_ids.push_back(i);
            }
        }
        std::shuffle(agents_ids.begin(),agents_ids.end(),*MT);

        std::vector<int> disabled_agents_ids;
        disabled_agents_num=env.num_of_agents-max_agents_in_use;
        disabled_agents_num=std::min(disabled_agents_num,(int)agents_ids.size());
        for (int i=0;i<disabled_agents_num;++i) {
            (*agent_infos)[agents_ids[i]].disabled=true;
            disabled_agents_ids.push_back(agents_ids[i]);
        }

        // nlohmann::json disabled_agents_json(disabled_agents_ids);
        // std::ofstream f("disabled_agents.json");
        // f<<disabled_agents_json;
        // f.close();
    }

    std::cout<<"strategy: "<<strategy<<" #disabled agents: "<<disabled_agents_num<<std::endl;
    
}

Instance LaCAM2Solver::build_instance(const SharedEnvironment & env, std::vector<Path> * precomputed_paths) {
    auto starts=vector<std::pair<uint,int> >();
    auto goals=vector<std::pair<uint,int> >();

    int ctr=0;
    for (int i=0;i<env.num_of_agents;++i) {
        starts.emplace_back(env.curr_states[i].location, env.curr_states[i].orientation);
        assert(env.goal_locations[i].size()>0);
        int goal_location=env.goal_locations[i][0].first;
        auto & agent_info=(*agent_infos)[i];
        if (disable_corner_target_agents) {
            int g_x=goal_location%env.cols;
            int g_y=goal_location/env.cols;
            // disable agent if its goal is a corner
            if (G->U[goal_location]->neighbor.size()<=1) {
                agent_info.disabled=true;
                ++ctr;
            }
        }
        if (disable_agent_goals && agent_info.disabled) {
            goal_location=env.curr_states[i].location;
        }
        goals.emplace_back(goal_location, -1);
        // cerr<<"0\trandom-32-32-20.map\t32\t32\t"<<starts[i]%32<<"\t"<<starts[i]/32<<"\t"<<goals[i]%32<<"\t"<<goals[i]/32<<"\t0"<<endl;
        if (goal_location!=agent_info.goal_location){
            agent_info.goal_location=goal_location;
            agent_info.elapsed=0;
            agent_info.tie_breaker=get_random_float(MT,0,1);
            agent_info.stuck_order=0;
        } else {
            agent_info.elapsed+=1;
        }
    }
    // std::cout<<"ctr: "<<ctr<<std::endl;
    return Instance(*G, starts, goals, *agent_infos, read_param_json<int>(config,"planning_window",-1), precomputed_paths);
}

int LaCAM2Solver::get_neighbor_orientation(int loc1,int loc2) {

    // 0:east, 1:south, 2:west, 3:north

    if (loc1+1==loc2) {
        return 0;
    }

    if (loc1+G->width==loc2) {
        return 1;
    }

    if (loc1-1==loc2) {
        return 2;
    }

    if (loc1-G->width==loc2) {
        return 3;
    }

    cerr<<"loc1 and loc2 are not neighbors: "<<loc1<<", "<<loc2<<endl;
    exit(-1);

}

float LaCAM2Solver::get_action_cost(int pst, int ost, int ped, int oed) {
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

float LaCAM2Solver::eval_solution(const Instance & instance, const Solution & solution) {
    float cost=0;
    for (int aid=0;aid<instance.N;++aid) {
        // TODO(rivers): should we consider the case of arrival here?
        bool arrived=false;
        for (int i=0;i<solution.size()-1;++i) {
            if (solution[i].arrivals[aid]) {
                arrived=true;
                break;
            }
            int loc=solution[i].locs[aid]->index;
            int orient=solution[i].orients[aid];
            int next_loc=solution[i+1].locs[aid]->index;
            int next_orient=solution[i+1].orients[aid];
            cost+=get_action_cost(loc,orient,next_loc,next_orient);
        }
        if (!arrived){
            int loc=solution.back().locs[aid]->index;
            int orient=solution.back().orients[aid];
#ifndef NO_ROT
            cost += HT->get(loc,orient,instance.goals.locs[aid]->index);
#else
            cost += HT->get(loc,instance.goals.locs[aid]->index);
#endif
        }
    }

    return cost;
}

void LaCAM2Solver::plan(const SharedEnvironment & env, std::vector<Path> * precomputed_paths, std::vector<::State> * starts, std::vector<::State> * goals){
    ONLYDEV(g_timer.record_p("lacam2_plan_pre_s");)
    // std::cerr<<"random :"<<get_random_int(MT,0,100)<<std::endl;

    // g_timer.record_p("stats_tree_s");
    // StatsTree * stats_tree = new StatsTree(env.cols,env.rows);
    // for (int i=0;i<env.num_of_agents;++i) {
    //     int x=env.curr_states[i].location%env.cols;
    //     int y=env.curr_states[i].location/env.cols;
    //     stats_tree->update(x,y,1);
    // }
    // g_timer.record_d("stats_tree_s","stats_tree");

    // if (timestep==0) {
    //     for (int i=0;i<env.num_of_agents;++i) {
    //         paths[i].push_back(env.curr_states[i]);
    //     }
    // }

    if (env.curr_timestep==0 && max_agents_in_use<env.num_of_agents) {
        disable_agents(env);
    }


    if (need_replan) {
        const int verbose = 10;
        const int time_limit_sec = 2;
        ONLYDEV(g_timer.record_p("lacam_build_instance_s");)
        auto instance = build_instance(env, precomputed_paths);
        if (starts!=nullptr) {
            if (goals==nullptr) {
                std::cerr<<"not supported now! goals must be specified as well"<<endl;
                exit(-1);
            }
            instance.set_starts_and_goals(starts,goals);
        }

        if (precomputed_paths!=nullptr) {
            // we need to check the initial states are the same.
            for (int i=0;i<env.num_of_agents;++i) {
                if ((*precomputed_paths)[i].size()==0 || (*precomputed_paths)[i][0].location!=instance.starts.locs[i]->index) {
                    cerr<<"agent "<<i<<" has zero-length precomputed paths or initial states are not the same!"<<endl;
                    cerr<<"size: "<<(*precomputed_paths)[i].size()<<endl;
                    cerr<<"states: "<<(*precomputed_paths)[i][0]<<" vs "<<instance.starts.locs[i]<<endl;
                    exit(-1);
                }
            }
        }

        ONLYDEV(g_timer.record_d("lacam_build_instance_s","lacam_build_instance");)
        const auto deadline = Deadline(time_limit_sec * 1000);
        bool use_swap=false; // TODO: we need try use_swap
        bool use_orient_in_heuristic=read_param_json<bool>(config,"use_orient_in_heuristic");

        vector<::Path> precomputed_paths;
        if (read_param_json<int>(config["SUO"],"iterations")>0) {
            ONLYDEV(g_timer.record_p("suo_init_s");)
            SUO2::Spatial::SUO suo(
                env,
                *map_weights,
                HT,
                read_param_json<float>(config["SUO"],"vertex_collision_cost"),
                read_param_json<int>(config["SUO"],"iterations"),
                read_param_json<int>(config["SUO"],"max_expanded"),
                read_param_json<int>(config["SUO"],"window"),
                read_param_json<float>(config["SUO"],"h_weight")
            );
            ONLYDEV(g_timer.record_d("suo_init_s","suo_init");)
            ONLYDEV(g_timer.record_p("suo_plan_s");)
            suo.plan();
            ONLYDEV(g_timer.record_d("suo_plan_s","suo_plan");)
            g_timer.print_all_d();

            ONLYDEV(g_timer.record_p("copy_suo_paths_s");)
            

            // std::cout<<"here"<<std::endl;
            precomputed_paths.resize(env.num_of_agents);
            for (int i=0;i<env.num_of_agents;++i){
                if (suo.paths[i][0].pos!=env.curr_states[i].location || suo.paths[i][0].orient!=env.curr_states[i].orientation){
                    cerr<<"agent "<<i<<"'s current state doesn't match with the plan"<<endl;
                    cerr<<"current state: "<<env.curr_states[i].location<<" "<<env.curr_states[i].orientation<<endl;
                    cerr<<"plan state: "<<suo.paths[i][0].pos<<" "<<suo.paths[i][0].orient<<endl;
                    exit(-1);
                }
                for (int j=0;j<suo.paths[i].size();++j){
                    precomputed_paths[i].emplace_back(suo.paths[i][j].pos,-1,suo.paths[i][j].orient);
                }
                // if (i==0){
                //     std::cout<<"agent "<<i<<" starts:"<<env.curr_states[i]<<" "<<env.goal_locations[i][0].first<<endl;
                //     std::cout<<precomputed_paths[i]<<endl;
                // }
            }
            // we need to change precomputed_paths to suo_paths. because the former one means hard constraints to follow
            // but the latter one is just a suggesion.
            instance.precomputed_paths=&precomputed_paths;
            ONLYDEV(g_timer.record_d("copy_suo_paths_s","copy_suo_paths");)
        }

        float best_cost=FLT_MAX;
        Solution best_solution;
        ONLYDEV(g_timer.record_d("lacam2_plan_pre_s","lacam2_plan_pre");)

        // #pragma omp parallel for
        // for (int i=1;i<2;++i) {
            
            int order_strategy=1;
            string _order_strategy=read_param_json<string>(config,"order_strategy");
            if (_order_strategy=="early_time") {
                order_strategy=1;
            } else if (_order_strategy=="short_dist") {
                order_strategy=0;
            } else {
                cout<<"unknown order strategy: "<<_order_strategy<<endl;
                exit(-1);
            }

            ONLYDEV(g_timer.record_p("lacam_build_planner_s");)
            auto planner = Planner(&instance,HT,map_weights,&deadline,MT,0,LaCAM2::OBJ_SUM_OF_LOSS,0.0F,
                use_swap,
                use_orient_in_heuristic,
                use_external_executor,
                disable_agent_goals
            );
            ONLYDEV(g_timer.record_d("lacam_build_planner_s","lacam_build_planner");)
            auto additional_info = std::string("");
            ONLYDEV(g_timer.record_p("lacam_solve_s");)
            auto solution=planner.solve(additional_info,order_strategy);
            ONLYDEV(g_timer.record_d("lacam_solve_s","lacam_solve");)
            auto cost=eval_solution(instance,solution);
            // #pragma omp critical
            {
                if (cost<best_cost) {
                    best_cost=cost;
                    best_solution=solution;
                }
                // std::cerr<<i<<precomputed_paths[i]<<endl;
            }
        // }

        // std::cout<<"old:"<<std::endl;
        // for (int i=0;i<env.num_of_agents;++i) {
        //     std::cout<<i<<" "<<paths[i]<<std::endl;
        // }
#ifndef NO_ROT
        if (use_external_executor) {

            solution_convert(env,best_solution,paths);

        } else {
#endif
            
            ONLYDEV(g_timer.record_p("lacam2_plan_copy_path_s");)
            for (int i=0;i<env.num_of_agents;++i) {
                // cerr<<"xagent "<<i<<": ";
                // for (int j=1;j<solution.size();++j) {
                //     cerr<<solution[j][i]->index<<" ";
                // }
                // cerr<<endl;
                for (int j=0;j<best_solution.size();++j) {
                    paths[i].emplace_back(best_solution[j].locs[i]->index,j,best_solution[j].orients[i]);
                }
                if (paths[i].size()==1) {
                    paths[i].emplace_back(paths[i].back().location,paths[i].size(),paths[i].back().orientation);
                }
            }
            ONLYDEV(g_timer.record_d("lacam2_plan_copy_path_s","lacam2_plan_copy_path");)
    
#ifndef NO_ROT
        }
#endif
        //std::cout<<"replan:"<<paths[0].size()<<std::endl;


        // for (int i=0;i<env.num_of_agents;++i) {
        //     std::cout<<i<<" "<<paths[i]<<std::endl;
        // }

        // // failure
        // if (solution.empty()) {
        //     info(1, verbose, "failed to solve");
        //     exit(1);
        // }

        // // check feasibility
        // if (!is_feasible_solution(instance, solution, verbose)) {
        //     info(0, verbose, "invalid solution");
        //     exit(2);
        // }
        ONLYDEV(g_timer.record_p("lacam2_plan_post_s");)
        // post processing
        // auto comp_time_ms = deadline.elapsed_ms();
        ONLYDEV(g_timer.record_p("lacam2_plan_print_s");)
        // print_stats(verbose, instance, HT, best_solution, comp_time_ms);
        // cout<<"solution length:"<<best_solution.size()<<endl;
        // cout<<"solution cost"<<best_cost<<endl;
        ONLYDEV(g_timer.record_d("lacam2_plan_print_s","lacam2_plan_print");)

        ONLYDEV(g_timer.record_d("lacam2_plan_post_s","lacam2_plan_post");)

        // if (!read_param_json<bool>(config,"consider_rotation")) {
        //     for (int i=0;i<env.num_of_agents;++i) {
        //         // cerr<<"xagent "<<i<<": ";
        //         // for (int j=1;j<solution.size();++j) {
        //         //     cerr<<solution[j][i]->index<<" ";
        //         // }
        //         // cerr<<endl;
        //         for (int j=1;j<solution.size();++j) {
        //             paths[i].emplace_back(solution[j][i]->index,env.curr_states[i].timestep+j,-1);
        //         }
        //     }
        // }
    }

    // if (read_param_json<bool>(config,"consider_rotation")) {
    //     vector<State> planned_next_states;
    //     vector<State> next_states;
    //     for (int i=0;i<env.num_of_agents;++i) {
    //         planned_next_states.emplace_back(next_config[i]->index,-1,-1);
    //         next_states.emplace_back(-1,-1,-1);
    //     }

    //     if (!read_param_json<bool>(config,"use_slow_executor")) {
    //         executor.execute(&(env.curr_states),&planned_next_states,&next_states);
    //     } else {
    //         slow_executor.execute(&(env.curr_states),&planned_next_states,&next_states);
    //     }

    //     for (int i=0;i<env.num_of_agents;++i) {
    //         if (next_states[i].timestep!=env.curr_states[i].timestep+1) {
    //             std::cerr<<i<<" "<<next_states[i].timestep<<" "<<env.curr_states[i].timestep<<endl;
    //             exit(-1);
    //         }

    //         paths[i].emplace_back(next_states[i]);
    //         // std::cerr<<i<<" "<<env.curr_states[i]<<" "<<next_states[i]<<endl;
    //     }
    // }

    // bool ready_to_forward = true;
    // for (int i=0;i<env.num_of_agents;++i) {
    //     auto & curr_state = paths[i][timestep];
    //     if (curr_state.location!=next_config[i]->index) {
    //         int expected_orient = get_neighbor_orientation(curr_state.location,next_config[i]->index);
    //         int curr_orient = curr_state.orientation;
    //         if (expected_orient!=curr_orient){
    //             ready_to_forward = false;
    //             break;
    //         }
    //     }
        
    // }

    // cout<<"ready to forward: "<<ready_to_forward<<endl;

    // if (!ready_to_forward) {
    //     for (int i=0;i<env.num_of_agents;++i) {
    //         auto & curr_state = paths[i][timestep];
    //         if (curr_state.location==next_config[i]->index) {
    //             paths[i].emplace_back(curr_state.location,curr_state.timestep+1,curr_state.orientation);
    //         } else {
    //             int expected_orient = get_neighbor_orientation(curr_state.location,next_config[i]->index);
    //             int curr_orient = curr_state.orientation;
    //             if (expected_orient==curr_orient){
    //                 paths[i].emplace_back(curr_state.location,curr_state.timestep+1,expected_orient);
    //             } else {
    //                 int d1=(curr_orient+4-expected_orient)%4;
    //                 int d2=(expected_orient+4-curr_orient)%4;

    //                 int next_orient=-1;
    //                 if (d1<d2) {
    //                     next_orient=(curr_orient-1+4)%4;
    //                 } else {
    //                     next_orient=(curr_orient+1+4)%4;
    //                 }
    //                 paths[i].emplace_back(curr_state.location,curr_state.timestep+1,next_orient);
    //             }
    //         }
    //     }
    // } else {
    //     for (int i=0;i<env.num_of_agents;++i) {
    //         auto & curr_state = paths[i][timestep];
    //         if (curr_state.location==next_config[i]->index) {
    //             paths[i].emplace_back(curr_state.location,curr_state.timestep+1,curr_state.orientation);
    //         } else {
    //             paths[i].emplace_back(next_config[i]->index,curr_state.timestep+1,curr_state.orientation);
    //         }
    //     }
    // }

    // total_feasible_timestep+=1;

}

void LaCAM2Solver::solution_convert(const SharedEnvironment & env, Solution & solution, std::vector<Path> & _paths) {

    int num_steps=0;

    int N=_paths.size();

    int planning_window=read_param_json<int>(config,"planning_window");

    auto & curr_config=solution[0];

    std::vector<::State> curr_states;
    for (int aid=0;aid<N;++aid) {
        curr_states.emplace_back(curr_config.locs[aid]->index,0,curr_config.orients[aid]);
        _paths[aid].push_back(curr_states[aid]);
    }

    // std::cout<<solution.size()<<std::endl;

    for (int i=1;i<solution.size();++i) {
        auto & next_config=solution[i];
        while (true) {
            std::vector<::State> planned_next_states;
            std::vector<::State> next_states;
        
            planned_next_states.reserve(N);
            next_states.reserve(N);

            for (int i=0;i<N;++i){
                planned_next_states.emplace_back(next_config.locs[i]->index,-1,-1);
                next_states.emplace_back(-1,-1,-1);
            }

            executor.execute(&curr_states,&planned_next_states,&next_states);

            curr_states=next_states;
            ++num_steps;

            for (int aid=0;aid<N;++aid) {
                _paths[aid].push_back(next_states[aid]);
            }

            if (num_steps>=planning_window) {
                break;
            }

            // check if arrived
            bool arrived=true;
            for (int aid=0;aid<N;++aid) {
                if (planned_next_states[aid].location!=next_states[aid].location) {
                    arrived=false;
                    break;
                }
            }

            if (arrived) {
                break;
            }
        }

        if (num_steps>=planning_window) {
            break;
        }
    }
}

void LaCAM2Solver::get_step_actions(const SharedEnvironment & env, vector<Action> & actions) {
    // check empty
    assert(actions.empty());


    if (num_task_completed>=max_task_completed) { // only for competition purpose, don't reveal too much information, otherwise it is too tired to overfit... do something fun instead!
        for (int i=0;i<env.num_of_agents;++i) {
            actions.push_back(Action::W);
        }
    } else {
        for (int i=0;i<env.num_of_agents;++i) {
            // we will get action indexed at timestep+1
            if (paths[i].size()<=timestep+1){
                cerr<<"wierd error for agent "<<i<<". path length: "<<paths[i].size()<<", "<<"timestep+1: "<<timestep+1<<endl;
                assert(false);
            }
            actions.push_back(get_action_from_states(paths[i][timestep],paths[i][timestep+1]));
        
            // assume perfect execution
            if (paths[i][timestep+1].location==env.goal_locations[i][0].first){
                ++num_task_completed;
            }
        }
    }

    // for (int i=0;i<env.num_of_agents;++i) {
    //     int action_cost=get_action_cost(paths[i][timestep].location,paths[i][timestep].orientation,paths[i][timestep+1].location,paths[i][timestep+1].orientation);
    //     action_costs[i].push_back(action_cost);
    //     int a=get_action_from_states(paths[i][timestep],paths[i][timestep+1]);
    //     total_actions[i].push_back(a);
    // }

    // if (env.curr_timestep==3) {
    //     for (int i=0;i<env.num_of_agents;++i) {
    //         int curr_loc=env.curr_states[i].location;
    //         int y=curr_loc/env.cols;
    //         if (y==7){
    //             cout<<"agent "<<i<<": ";
    //             cout<<paths[i]<<endl;
    //             cout<<endl;
    //             for (int j=0;j<action_costs[i].size();++j) {
    //                 cout<<action_costs[i][j]<<" ";
    //             }
    //             cout<<endl;
    //         }
    //     }
    // }


    // int waiting_ctr=0;
    // for (auto action:actions) {
    //     if (action==Action::W) {
    //         ++waiting_ctr;
    //     }
    // }
    // double waiting_ratio=(double)waiting_ctr/actions.size();


    // if (waiting_ratio>0.8 && !this->flag) {
    //     std::cerr<<"dead lock!"<<endl;
    //     exit(-1);
    //     std::this_thread::sleep_for(std::chrono::seconds(2));
    //     this->flag=true;
    // }

    // for (int i=0;i<env.num_of_agents;++i) {
    //     cout<<actions[i]<<" ";
    // }
    // cout<<endl;

    // TODO(hj) we probably still want to check the validness. so we need construct model or implement is_valid by ourselves.
    // check if not valid, this should not happen in general if the algorithm is correct? but maybe there exist deadlocks.
    // TODO(hj) detect deadlock?
    if (!action_model.is_valid(env.curr_states,actions)){
        cerr<<"planed actions are not valid in timestep "<<timestep+1<<"!"<<endl;
#ifdef DEV
        exit(-1);
#else
        actions.resize(env.num_of_agents, Action::W);
#endif
    } 
    else {
        // NOTE(hj): only successfully executing a planned step will increase this internal timestep, which is different from the real timestep used in the simulation system.
        timestep+=1;
    }

    std::cout<<"timestep: "<<timestep<<" "<<paths[0].size()<<std::endl;

    // TODO(hj): when we need to replan?
    if (timestep+planning_window-execution_window==paths[0].size()-1) {
        need_replan=true;
    } else {
        need_replan=false;
    }

    std::cout<<"need_replan"<<need_replan<<std::endl;

    // need_replan=true;
   
    // need_replan=false;

    // bool all_arrived=true;
    // for (int i=0;i<env.num_of_agents;++i) {
    //     if (paths[i][timestep].location!=next_config.locs[i]->index) {
    //         // arrive goal locations
    //         all_arrived=false;
    //         break;
    //     }
    // }    
    // if (all_arrived) {
    //     need_replan=true;
    // }

    // // 1. exceed simulation window
    // // if (timestep==total_feasible_timestep){
    // //     need_replan=true;
    // // }
    
    // // 2. goal changes: there different ways to check this. let's just keep the old goal and compare.
    // for (int i=0;i<env.num_of_agents;++i) {
    //     if (paths[i][timestep].location==env.goal_locations[i][0].first) {
    //         // arrive goal locations
    //         need_replan=true;
    //         break;
    //     }
    // }
    
    // if (!read_param_json<bool>(config,"use_slow_executor")) {
    //     need_replan=true;
    // }

    if (need_replan) {
        for (int i=0;i<env.num_of_agents;++i) {
            // paths[i].resize(timestep+1);
            paths[i].clear();
            timestep=0;
        }
    }
}



}