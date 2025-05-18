#include "LNS/Parallel/LocalOptimizer.h"
#include "util/Timer.h"
#include "util/Dev.h"

namespace LNS {

namespace Parallel {

LocalOptimizer::LocalOptimizer(
    Instance & instance, std::vector<Agent> & agents, std::shared_ptr<HeuristicTable> HT, 
    std::shared_ptr<vector<float> > map_weights, std::shared_ptr<std::vector<LaCAM2::AgentInfo> > agent_infos,
    string replan_algo_name, bool sipp,
    int window_size_for_CT, int window_size_for_CAT, int window_size_for_PATH, int execution_window,
    bool has_disabled_agents,
    int screen,
    int random_seed
):
    instance(instance), path_table(instance.map_size,window_size_for_CT), HT(HT), map_weights(map_weights), agent_infos(agent_infos),
    replan_algo_name(replan_algo_name),
    window_size_for_CT(window_size_for_CT), window_size_for_CAT(window_size_for_CAT), window_size_for_PATH(window_size_for_PATH),
    has_disabled_agents(has_disabled_agents),
    screen(screen), MT(random_seed) {

    // TODO(rivers): for agent_id, we just use 0 to initialize the path planner. but we need to change it (also starts and goals) everytime before planning
    path_planner = std::make_shared<TimeSpaceAStarPlanner>(instance, HT, map_weights, execution_window);

    for (int i=0;i<instance.num_of_agents;++i) {
        this->agents.emplace_back(i,instance,HT,agent_infos);
    }

}

void LocalOptimizer::reset() {
    path_table.reset();
    // for (auto & agent: agents) {
    //     agent.reset();
    // }
}

void LocalOptimizer::update(Neighbor & neighbor) {
    if (neighbor.succ) {
        for (auto & aid: neighbor.agents) {
            path_table.deletePath(aid, neighbor.m_old_paths[aid]);
        }

        for (auto & aid: neighbor.agents) {
            path_table.insertPath(aid, neighbor.m_paths[aid]);
            agents[aid].path = neighbor.m_paths[aid];
        }
    }
}

void LocalOptimizer::prepare(Neighbor & neighbor) {
    // store the neighbor information
    //ONLYDEV(g_timer.record_p("store_neighbor_info_s");)
    neighbor.old_sum_of_costs = 0;
    for (auto & aid: neighbor.agents)
    {
        auto & agent=agents[aid];
        if (replan_algo_name == "PP")
            neighbor.m_old_paths[aid] = agent.path;
        // path_table.deletePath(neighbor.agents[i], agent.path);
        neighbor.old_sum_of_costs += agent.path.path_cost;
        path_table.deletePath(aid, neighbor.m_old_paths[aid]);
    }   

    //ONLYDEV(g_timer.record_d("store_neighbor_info_s","store_neighbor_info_e","store_neighbor_info");)    
}

void LocalOptimizer::optimize(Neighbor & neighbor, const TimeLimiter & time_limiter) {

    prepare(neighbor);

    // replan
    //ONLYDEV(g_timer.record_p("replan_s");)
    bool succ=false;
    if (replan_algo_name == "PP")
        succ = runPP(neighbor, time_limiter);
    else
    {
        cerr << "Wrong replanning strategy" << endl;
        exit(-1);
    }
    if (!succ) {
        if (screen>=1)
            DEV_DEBUG("replan failed");
    }
    //ONLYDEV(g_timer.record_d("replan_s","replan_e","replan");)

    // the cleanup is done in runPP: e.g., if runPP fails, the old paths are restored.

    neighbor.succ=succ;
}

bool LocalOptimizer::runPP(Neighbor & neighbor, const TimeLimiter & time_limiter)
{
    //ONLYDEV(g_timer.record_p("run_pp_s");)
    auto shuffled_agents = neighbor.agents;
    std::shuffle(shuffled_agents.begin(), shuffled_agents.end(), MT);

    if (has_disabled_agents) {
        std::stable_sort(shuffled_agents.begin(), shuffled_agents.end(), [&](int a, int b) {
            return (*agent_infos)[a].disabled<(*agent_infos)[b].disabled; // not disabled first.
        });
    }
    // TODO: we need also to remove the cost of disabled agents.

    if (screen >= 2) {
        for (auto aid : shuffled_agents)
            cout << aid << "(" << agents[aid].getNumOfDelays()<<"), ";
        cout << endl;
    }
    int remaining_agents = (int)shuffled_agents.size();
    auto p = shuffled_agents.begin();
    neighbor.sum_of_costs = 0;
    CBSNode node;
    int suboptimality=1.2;
    int search_priority=1;
    bool use_soft_constraint=true;
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, &path_table, nullptr, window_size_for_CT, window_size_for_CAT, window_size_for_PATH);

    // TODO(rivers): we require the path to be at least window_size_for_PATH
    // TODO(rivers): we use hold goal location assumption here, which is not necessary.
    // if (window_size_for_PATH!=MAX_TIMESTEP) {
    //     constraint_table.length_min=window_size_for_PATH;
    // }

    while (p != shuffled_agents.end()) {
        if (time_limiter.timeout())
            break;

        int id = *p;
        int start_pos=instance.start_locations[id];
        int start_orient=instance.start_orientations[id];
        int goal_pos=instance.goal_locations[id];

        if (screen >= 3)
            cout << "Remaining agents = " << remaining_agents <<
                 ", remaining time = " << time_limiter.get_remaining_time() << " seconds. " << endl
                 << "Agent " << agents[id].id << endl;
        if (search_priority==1) {
            //ONLYDEV(g_timer.record_p("findPath_s");)
            path_planner->findPath(start_pos,start_orient,goal_pos,constraint_table, time_limiter);
            neighbor.m_paths[id] = path_planner->path;
            //ONLYDEV(g_timer.record_d("findPath_s","findPath_e","findPath");)
        } else if (search_priority==2) {
            std::cerr<<"not supported now, need double checks"<<std::endl;
            exit(-1);
            // vector<Path *> paths(agents.size(),nullptr);
            // int min_f_val;
            // tie(neighbor.m_paths[id],min_f_val) = path_planner->findSuboptimalPath(node, constraint_table, paths, agents[id].id, 0,suboptimality);
        }
        if (neighbor.m_paths[id].empty()) break;
        
        // always makes a fixed length.
        // TODO(rivers): this might not be a smart choice, but it keeps everything consistent.
        // if (neighbor.m_paths[id].size()>constraint_table.window_size_for_PATH+1) {
        //     neighbor.m_paths[id].nodes.resize(constraint_table.window_size_for_PATH+1);
        // }

        // do we need to pad here?
        // assume hold goal location
        // if (neighbor.m_paths[id].size()<constraint_table.window_size_for_PATH+1) {
        //     neighbor.m_paths[id].nodes.resize(constraint_table.window_size_for_PATH+1,neighbor.m_paths[id].nodes.back());
        // }

        if (neighbor.m_paths[id].back().location!=agents[id].getGoalLocation()) {
            if (neighbor.m_paths[id].size()!=constraint_table.window_size_for_PATH+1) {
                std::cerr<<"agent "<<agents[id].id<<"'s path length "<<neighbor.m_paths[id].size()<<" should be equal to window size for path "<<constraint_table.window_size_for_PATH<< "if it doesn't arrive at its goal"<<endl;
                exit(-1);
            } 
        }

        // float _path = agents[id].getEstimatedPathLength(neighbor.m_paths[id], goal_pos, HT);
        // if (_path!=neighbor.m_paths[id].path_cost) {
        //     std::cerr<<"path cost "<<neighbor.m_paths[id].path_cost<<" is not equal to estimated path cost "<<_path<<std::endl;
        //     exit(-1);
        // }
        neighbor.m_paths[id].path_cost = agents[id].getEstimatedPathLength(neighbor.m_paths[id], goal_pos, HT);
        neighbor.sum_of_costs += neighbor.m_paths[id].path_cost;

        if (neighbor.sum_of_costs >= neighbor.old_sum_of_costs){
            // because it is not inserted into path table yet.
            neighbor.m_paths[id].clear();
            break;
        }
        remaining_agents--;
        path_table.insertPath(agents[id].id, neighbor.m_paths[id]);
        ++p;
    }

    // remove any insertion from new paths
    for (auto & aid: neighbor.agents) {
        path_table.deletePath(aid, neighbor.m_paths[aid]);
    }

    // restore old paths
    if (!neighbor.m_old_paths.empty())
    {
        for (auto & aid : neighbor.agents) {
            path_table.insertPath(aid, neighbor.m_old_paths[aid]);
        }
    }
    //ONLYDEV(g_timer.record_d("run_pp_s","run_pp_e","run_pp");)


    if (remaining_agents == 0 && neighbor.sum_of_costs <= neighbor.old_sum_of_costs) // accept new paths
    {
        return true;
    }
    else 
    {   
        return false;
    }
}

}

}