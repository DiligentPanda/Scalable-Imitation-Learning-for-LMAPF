#include "LNS/Parallel/GlobalManager.h"
#include "util/Timer.h"
#include "omp.h"
#include "util/TimeLimiter.h"
#include <cstdlib>

namespace LNS {

namespace Parallel {

GlobalManager::GlobalManager(
    bool async,
    Instance & instance, std::shared_ptr<HeuristicTable> HT, 
    std::shared_ptr<vector<float> > map_weights, std::shared_ptr<std::vector<LaCAM2::AgentInfo> > agent_infos,
    int neighbor_size, destroy_heuristic destroy_strategy,
    bool ALNS, double decay_factor, double reaction_factor,
    string init_algo_name, string replan_algo_name, bool sipp,
    int window_size_for_CT, int window_size_for_CAT, int window_size_for_PATH, int execution_window,
    bool has_disabled_agents,
    bool fix_ng_bug,
    int screen,
    bool _verbose,
    int _num_threads,
    int _max_iterations
): 
    async(async),
    instance(instance), path_table(instance.map_size,window_size_for_PATH), HT(HT), map_weights(map_weights),
    init_algo_name(init_algo_name), replan_algo_name(replan_algo_name),
    window_size_for_CT(window_size_for_CT), window_size_for_CAT(window_size_for_CAT), window_size_for_PATH(window_size_for_PATH),
    screen(screen), agent_infos(agent_infos), has_disabled_agents(has_disabled_agents), num_threads(_num_threads), max_iterations(_max_iterations), verbose(_verbose) {

    if (_max_iterations < 0) {
        char * max_iterations_env = std::getenv("LNS_MAX_ITERATIONS");
        if (max_iterations_env!=nullptr) {
            max_iterations=std::atoi(max_iterations_env);
        } else {
            max_iterations = INT_MAX;
        }
    }

    if (num_threads==-1) {
        char * num_threads_env = std::getenv("LNS_NUM_THREADS");
        if (num_threads_env!=nullptr) {
            num_threads=std::atoi(num_threads_env);
        } else {
            num_threads=omp_get_max_threads();
        }
    }

    omp_set_num_threads(num_threads);

    if (verbose)
        cout<<"LNS use "<<num_threads<<" threads"<<endl;

    // for (auto w: *map_weights){
    //     if (w!=1){
    //         DEV_ERROR("we cannot support weighted map now for LNS! because in that way, we may need two different heuristic table. one for path cost estimation, one for path length estimation.");
    //         exit(-1);
    //     }
    // }
    
    for (int i=0;i<instance.num_of_agents;++i) {
        agents.emplace_back(i,instance,HT,agent_infos);
    }

    // cout<<num_threads<<endl;
    // exit(-1);

    for (auto i=0;i<num_threads;++i) {
        auto local_optimizer=std::make_shared<LocalOptimizer>(
            instance, agents, HT, map_weights, agent_infos,
            replan_algo_name, sipp,
            window_size_for_CT, window_size_for_CAT, window_size_for_PATH, execution_window,
            has_disabled_agents,
            screen, i*2023+1
        );
        local_optimizers.push_back(local_optimizer);
    }

    if (!async) {
        std::cerr<<"WARNING: some functionalities might be missing in the sync version. If you want use it, please check the code and remove the following exit."<<std::endl;
        exit(-1);
        neighbor_generator=std::make_shared<NeighborGenerator>(
            instance, HT, path_table, agents, agent_infos,
            neighbor_size, destroy_strategy, 
            ALNS, decay_factor, reaction_factor, 
            num_threads, fix_ng_bug, screen, 0
        );
    } else {
        for (auto i=0;i<num_threads;++i) {
            auto neighbor_generator=std::make_shared<NeighborGenerator>(
                instance, HT, local_optimizers[i]->path_table, local_optimizers[i]->agents, agent_infos,
                neighbor_size, destroy_strategy, 
                ALNS, decay_factor, reaction_factor, 
                num_threads, fix_ng_bug, screen, i*2023+1314
            );
            neighbor_generators.push_back(neighbor_generator);
        }
        updating_queues.resize(num_threads);
        updating_queue_locks.resize(num_threads);
        for (auto & lock: updating_queue_locks) {
            omp_init_lock(&lock);
        }
    }
}

void GlobalManager::reset() {
    initial_sum_of_costs=MAX_COST;
    sum_of_costs=MAX_COST;
    num_of_failures=0;
    average_group_size=0;  
    sum_of_distances=0;

    iteration_stats.clear();
    path_table.reset();
    for (auto & agent: agents) {
        agent.reset();
    }

    if (!async) {
        // call reset of neighbor_generator
        neighbor_generator->reset();
    } else {
        #pragma omp parallel for
        for (auto & neighbor_generator: neighbor_generators) {
            neighbor_generator->reset();
        }
        for (int i=0;i<num_threads;++i) {
            omp_set_lock(&(updating_queue_locks[i]));
            updating_queues[i].clear();
            omp_unset_lock(&(updating_queue_locks[i]));
        }
    }

    // call reset of local_optimizers
    #pragma omp parallel for
    for (auto & local_optimizer: local_optimizers) {
        local_optimizer->reset();
    }

}

GlobalManager::~GlobalManager() {
    if (async) {
        for (auto & lock: updating_queue_locks) {
            omp_destroy_lock(&lock);
        }
    }
}

void GlobalManager::update(Neighbor & neighbor, bool recheck) {

    if (neighbor.succ){
        // before we update, we check if the solution is valid. because in the parallel setting, we may have later update that makes the solution invalid.

        if (recheck) {
            g_timer.record_p("manager_update_s");
        } else {
            g_timer.record_p("init_manager_update_s");
        }

        if (recheck) {
            g_timer.record_p("recheck_s");
            // re-check validness
            bool valid=true;
            for (auto & aid: neighbor.agents) {
                auto & path=neighbor.m_paths[aid];
                for (int i=0;i<path.size()-1;++i) {
                    int from=path[i].location;
                    int to=path[i+1].location;
                    int to_time=i+1;
                    // TODO(rivers): we need to ignore the conflicts with agents in the neighbor.
                    if (path_table.constrained(from,to,to_time,neighbor.agents)) {
                        // ONLYDEV(std::cout<<aid<<" "<<i<<" invalid"<<std::endl;)
                        valid=false;
                        break;
                    }
                }
                if (!valid) break;
            }

            if (!valid) {
                neighbor.succ=false;
                // ONLYDEV(std::cout<<"invalid"<<std::endl;)
                return;
            }

            // re-check if the cost is still smaller
            float old_sum_of_costs=0;
            for (auto & aid: neighbor.agents) {
                old_sum_of_costs+=agents[aid].path.path_cost;
            }

            // TODO(rivers): use < or <= here?
            if (old_sum_of_costs<=neighbor.sum_of_costs) {
                neighbor.succ=false;
                // ONLYDEV(std::cout<<"incost"<<std::endl;)
                return;
            } 

            // re-update old_sum_of_costs here.
            neighbor.old_sum_of_costs=old_sum_of_costs;
            for (auto & id: neighbor.agents) {
                neighbor.m_old_paths[id]=agents[id].path;
            }

            g_timer.record_d("recheck_s","recheck");

        }

        update(neighbor);

        if (recheck) {
            g_timer.record_d("manager_update_s","manager_update");
        } else {
            g_timer.record_d("init_manager_update_s","init_manager_update");
        }
    }
}

void GlobalManager::update(Neighbor & neighbor) {
    // apply update
    g_timer.record_p("path_table_delete_s");
    for (auto & aid: neighbor.agents) {
        bool verbose=false;
        // if (neighbor.agents.size()<10){
        //     verbose=true;
        // } 
        path_table.deletePath(aid, neighbor.m_old_paths[aid],verbose);
    }
    g_timer.record_d("path_table_delete_s","path_table_delete");

    // std::cerr<<std::endl;
    g_timer.record_p("path_table_insert_s");
    for (auto & aid: neighbor.agents) {
        // update agents' paths here
        agents[aid].path = neighbor.m_paths[aid];
        bool verbose=false;
        // if (neighbor.agents.size()<10){
        //     verbose=true;
        // } 
        // update path table here
        path_table.insertPath(aid, neighbor.m_paths[aid],verbose);
    }
    g_timer.record_d("path_table_insert_s","path_table_insert");

        // std::cerr<<std::endl;
    // update costs here
    sum_of_costs += neighbor.sum_of_costs - neighbor.old_sum_of_costs;
}

bool GlobalManager::run(TimeLimiter & time_limiter) {
    if (async) {
        return _run_async(time_limiter);
    } else {
        return _run(time_limiter);
    }
}

bool GlobalManager::_run_async(TimeLimiter & time_limiter) {

    initial_sum_of_costs=0;
    sum_of_costs=0;
    num_of_failures=0;
    average_group_size=0;
    iteration_stats.clear();

    double elapse=0;

    g_timer.record_p("lns_init_sol_s");
    sum_of_distances = 0;
    for (const auto & agent : agents)
    {
#ifndef NO_ROT
        sum_of_distances += HT->get(instance.start_locations[agent.id],instance.start_orientations[agent.id],instance.goal_locations[agent.id]);
#else
        sum_of_distances += HT->get(instance.start_locations[agent.id],instance.goal_locations[agent.id]);
#endif
    }

    // 0. get initial solution
    Neighbor init_neighbor;
    init_neighbor.agents.resize(agents.size());
    for (int i=0;i<agents.size();++i) {
        init_neighbor.agents[i]=i;
    }
    getInitialSolution(init_neighbor);

    update(init_neighbor,false);

    // synchonize to local optimizer
    ONLYDEV(g_timer.record_p("init_loc_opt_update_s");)
    #pragma omp parallel for
    for (int i=0;i<num_threads;++i) {
        local_optimizers[i]->update(init_neighbor);
    }
    ONLYDEV(g_timer.record_d("init_loc_opt_update_s","init_loc_opt_update");)

    bool runtime=g_timer.record_d("lns_init_sol_s","lns_init_sol");

    elapse=time_limiter.get_elapse();
    iteration_stats.emplace_back(agents.size(), initial_sum_of_costs, runtime, init_algo_name);

    if (screen >= 1)
        cout << getSolverName() << " (only init executed): "
        << "runtime = " << runtime << ", "
        << "iterations = " << iteration_stats.size() << ", "
        << "solution cost = " << sum_of_costs << ", "
        << "initial solution cost = " << initial_sum_of_costs << ", "
        << "failed iterations = " << num_of_failures << endl;

    if (!init_neighbor.succ) {
        cerr << "Failed to get initial solution." << endl;
        exit(-1);
        return false;
    }

    g_timer.record_p("lns_opt_s");

    //std::cout<<"num_threads: "<<num_threads<<" "<<neighbor_generators.size()<<std::endl;


    // because already limited by the max iterations
    if (max_iterations>0 && max_iterations<INT_MAX) {
        time_limiter.disable();
    }

    #pragma omp parallel for
    for (int i=0;i<num_threads;++i) {
        int thread_id=omp_get_thread_num();
        int ctr=0;
        bool stop_flag=false;
        while (!stop_flag) {
            // if (ctr==1) {
            //     break;
            // }
            ++ctr;
            if (time_limiter.timeout())
                break;

            // 1. generate neighbor
            Neighbor neighbor=neighbor_generators[i]->generate(time_limiter,i);
                // for (auto aid:neighbor.agents) {
                //     cout<<aid<<" ";
                // }
                // cout<<endl;
            if (time_limiter.timeout())
                break;

            // 2. optimize the neighbor
            // auto & neighbor_ptr=neighbor_generators[i]->neighbors[i];
            // auto & neighbor=*neighbor_ptr;
            // #pragma omp critical
            // {
            //     for (auto aid: neighbor.agents) {
            //         if (aid<0||aid>instance.num_of_agents) {
            //             cout<<"thread id "<<thread_id<<" "<<i<<" invalid agent id "<<aid<<endl;
            //         }
            //     }
            // }

            local_optimizers[i]->optimize(neighbor, time_limiter);
            if (time_limiter.timeout())
                break;

            // cout<<"optimized"<<endl;

            // 3. update path table, statistics & maybe adjust strategies
            #pragma omp critical
            {
                // for (int i=0;i<1;++i) {
                    // if (time_limiter.timeout())
                    //     break;

                    // update alns
                    // auto & neighbor_ptr=neighbor_generators[i]->neighbors[i];
                    // auto & neighbor=*neighbor_ptr;
                    if (!time_limiter.timeout()){
                        neighbor_generators[i]->update(neighbor);
                    }
                    // if (time_limiter.timeout())
                    //     break;

                    if (!time_limiter.timeout()){
                        update(neighbor,true);
                    }
                    // if (time_limiter.timeout())
                    //     break;

                    if (!time_limiter.timeout()){
                        if (!neighbor.succ) {
                            ++num_of_failures;
                        } else {
                            for (int j=0;j<num_threads;++j) {
                                updating_queues[j].push_back(neighbor);
                            }
                        }

                        local_optimizers[i]->updating_queue=updating_queues[i];
                        updating_queues[i].clear();

                        elapse=time_limiter.get_elapse();
                        if (screen >= 1)
                            cout << "Iteration " << iteration_stats.size() << ", "
                                << "group size = " << neighbor.agents.size() << ", "
                                << "solution cost = " << sum_of_costs << ", "
                                << "remaining time = " << time_limiter.time_limit-elapse << endl;
                        iteration_stats.emplace_back(neighbor.agents.size(), sum_of_costs, elapse, replan_algo_name);
                        // including the init solution.
                        if (iteration_stats.size()>(size_t)max_iterations) {
                            stop_flag=true;
                        }

                    }
                // }
            }

            // synchonize to local optimizer
            if (time_limiter.timeout())
                break;
            for (auto & _neighbor: local_optimizers[i]->updating_queue) {
                if (time_limiter.timeout())
                    break;
                local_optimizers[i]->update(_neighbor);
            }
            local_optimizers[i]->updating_queue.clear();
        }
    }
    g_timer.record_d("lns_opt_s","lns_opt");

    average_group_size = - iteration_stats.front().num_of_agents;
    for (const auto& data : iteration_stats)
        average_group_size += data.num_of_agents;
    if (average_group_size > 0)
        average_group_size /= (double)(iteration_stats.size() - 1);

    elapse=time_limiter.get_elapse();
    // cout << getSolverName() << ": "
    //     << "runtime = " << elapse << ", "
    //     << "iterations = " << iteration_stats.size()-1 << ", "
    //     << "solution cost = " << sum_of_costs << ", "
    //     << "initial solution cost = " << initial_sum_of_costs << ", "
    //     << "failed iterations = " << num_of_failures << endl;

    if (verbose) {
        cout << getSolverName() << ": "
            << "runtime = " << elapse << ", "
            << "iterations = " << iteration_stats.size()-1 << ", "
            << "solution cost = " << sum_of_costs << ", "
            << "initial solution cost = " << initial_sum_of_costs << ", "
            << "failed iterations = " << num_of_failures << endl;
    }

    // because already limited by the max iterations
    if (max_iterations>0 && max_iterations<INT_MAX) {
        time_limiter.enable();
    }

    return true;
}

// TODO(rivers): we will do single-thread code refactor first, then we will do the parallelization
bool GlobalManager::_run(TimeLimiter & time_limiter) {

    initial_sum_of_costs=0;
    sum_of_costs=0;
    num_of_failures=0;
    average_group_size=0;
    iteration_stats.clear();

    double elapse=0;

    g_timer.record_p("lns_init_sol_s");
    sum_of_distances = 0;
    for (const auto & agent : agents)
    {
#ifndef NO_ROT
        sum_of_distances += HT->get(instance.start_locations[agent.id],instance.start_orientations[agent.id],instance.goal_locations[agent.id]);
#else
        sum_of_distances += HT->get(instance.start_locations[agent.id],instance.goal_locations[agent.id]);
#endif
    }

    // 0. get initial solution
    Neighbor init_neighbor;
    init_neighbor.agents.resize(agents.size());
    for (int i=0;i<agents.size();++i) {
        init_neighbor.agents[i]=i;
    }
    getInitialSolution(init_neighbor);

    update(init_neighbor,false);
    // synchonize to local optimizer
    ONLYDEV(g_timer.record_p("init_loc_opt_update_s");)
    #pragma omp parallel for
    for (int i=0;i<num_threads;++i) {
        local_optimizers[i]->update(init_neighbor);
    }
    ONLYDEV(g_timer.record_d("init_loc_opt_update_s","init_loc_opt_update");)
        

    bool runtime=g_timer.record_d("lns_init_sol_s","lns_init_sol");

    elapse=time_limiter.get_elapse();
    iteration_stats.emplace_back(agents.size(), initial_sum_of_costs, runtime, init_algo_name);

    if (screen >= 1)
        cout << getSolverName() << " (only init executed): "
        << "runtime = " << runtime << ", "
        << "iterations = " << iteration_stats.size() << ", "
        << "solution cost = " << sum_of_costs << ", "
        << "initial solution cost = " << initial_sum_of_costs << ", "
        << "failed iterations = " << num_of_failures << endl;

    if (!init_neighbor.succ) {
        cerr << "Failed to get initial solution." << endl;
        exit(-1);
        return false;
    }

    g_timer.record_p("lns_opt_s");
    while (true) {
        if (time_limiter.timeout())
            break;

        // 1. generate neighbors
        g_timer.record_p("neighbor_generate_s");
        neighbor_generator->generate_parallel(time_limiter);
        g_timer.record_d("neighbor_generate_s","neighbor_generate");

        if (time_limiter.timeout())
            break;

        // 2. optimize the neighbor
        // auto neighbor_ptr = neighbor_generator.neighbors.front();
        // auto & neighbor = *neighbor_ptr;
        // neighbor_generator.neighbors.pop();

        // in the single-thread setting, local_optimizer directly modify paths
        // but we should first recover the path table, then redo it after all local optimizers finishes.
        g_timer.record_p("loc_opt_s");
        #pragma omp parallel for
        for (auto i=0;i<neighbor_generator->neighbors.size();++i) {
            // cerr<<i<<" "<<neighbor_generator.neighbors[i]->agents.size()<<endl;
            auto & neighbor_ptr = neighbor_generator->neighbors[i];
            auto & neighbor = *neighbor_ptr;
            local_optimizers[i]->optimize(neighbor, time_limiter);
        }
        g_timer.record_d("loc_opt_s","loc_opt");

        if (time_limiter.timeout())
            break;

        // TODO(rivers): validate solution

        // 3. update path_table, statistics & maybe adjust strategies
        g_timer.record_p("neighbor_update_s");
        // TODO(rivers): it seems we should not modify neighbor in the previous update
        for (auto & neighbor_ptr: neighbor_generator->neighbors){
            if (time_limiter.timeout()) {
                break;
            } 
            auto & neighbor=*neighbor_ptr;
            neighbor_generator->update(neighbor);
        }
        g_timer.record_d("neighbor_update_s","neighbor_update");

        if (time_limiter.timeout()) {
            break;
        } 


        for (auto & neighbor_ptr: neighbor_generator->neighbors){
            if (time_limiter.timeout()) {
                break;
            } 
            auto & neighbor=*neighbor_ptr;
            update(neighbor,true);

            // synchonize to local optimizer
            ONLYDEV(g_timer.record_p("loc_opt_update_s");)
            #pragma omp parallel for
            for (int i=0;i<num_threads;++i) {
                local_optimizers[i]->update(neighbor);
            }
            ONLYDEV(g_timer.record_d("loc_opt_update_s","loc_opt_update");)

            if (!neighbor.succ) {
                ++num_of_failures;
            } 

            elapse=time_limiter.get_elapse();
            if (screen >= 1)
                cout << "Iteration " << iteration_stats.size() << ", "
                    << "group size = " << neighbor.agents.size() << ", "
                    << "solution cost = " << sum_of_costs << ", "
                    << "remaining time = " << time_limiter.time_limit-elapse << endl;
            iteration_stats.emplace_back(neighbor.agents.size(), sum_of_costs, elapse, replan_algo_name);
        }
    }
    g_timer.record_d("lns_opt_s","lns_opt");

    // TODO(rivers): validate solution

    average_group_size = - iteration_stats.front().num_of_agents;
    for (const auto& data : iteration_stats)
        average_group_size += data.num_of_agents;
    if (average_group_size > 0)
        average_group_size /= (double)(iteration_stats.size() - 1);

    // elapse=time_limiter.get_elapse();
    // cout << getSolverName() << ": "
    //     << "runtime = " << elapse << ", "
    //     << "iterations = " << iteration_stats.size()-1 << ", "
    //     << "solution cost = " << sum_of_costs << ", "
    //     << "initial solution cost = " << initial_sum_of_costs << ", "
    //     << "failed iterations = " << num_of_failures << endl;

    return true;
}

void GlobalManager::getInitialSolution(Neighbor & neighbor) {
    // currently, we only support initial solution directly passed in.
    neighbor.old_sum_of_costs=0;
    neighbor.sum_of_costs=0;
    for (int i=0;i<agents.size();++i) {
        // cerr<<agents[i].id<<" "<< agents[i].path.size()-1<<endl;
        if (agents[i].path.back().location!=instance.goal_locations[i] && agents[i].path.size()<=window_size_for_CT) {
            cerr<<"A precomputed agent "<<i<<"'s path "<<agents[i].path.size()
                <<" should be longer than window size for CT "<<window_size_for_CT
                <<" unless it arrives at its goal:"<<agents[i].path.back().location
                <<" vs "<<instance.goal_locations[i]<<endl;
            exit(-1);
        }

        if (agents[i].path.back().location!=instance.goal_locations[i] && agents[i].path.size()!=window_size_for_PATH+1) {
            cerr<<"we require agent either arrives at its goal earlier or has a planned path of length window_size_for_PATH. "<<agents[i].path.size()<<" vs "<<window_size_for_PATH<<endl;
            exit(-1);
        }

        neighbor.sum_of_costs+=agents[i].path.path_cost;

        neighbor.m_paths[i]=agents[i].path;
        neighbor.m_old_paths[i]=Path();
        // cerr<<agents[i].id<<" "<< agents[i].path.size()-1<<endl;
    }

    neighbor.succ=true;

    initial_sum_of_costs = neighbor.sum_of_costs;
}

}

}