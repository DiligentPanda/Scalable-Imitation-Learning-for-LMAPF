#include "LNS/Parallel/NeighborGenerator.h"
#include "util/Timer.h"
#include "util/Dev.h"
#include <algorithm>
#include "omp.h"

namespace LNS {

namespace Parallel {

NeighborGenerator::NeighborGenerator(
    Instance & instance, std::shared_ptr<TrafficMAPF::Dist2PathHeuristicTable> HT, PathTable & path_table, 
    std::vector<Agent> & agents, std::shared_ptr<std::vector<AgentInfo> > agent_infos,
    int neighbor_size, destroy_heuristic destroy_strategy, 
    bool ALNS, double decay_factor, double reaction_factor, 
    int num_threads, bool fix_ng_bug, int screen, int random_seed
):
    instance(instance), HT(HT), path_table(path_table), 
    agents(agents), agent_infos(agent_infos),
    neighbor_size(neighbor_size), destroy_strategy(destroy_strategy),
    ALNS(ALNS), decay_factor(decay_factor), reaction_factor(reaction_factor),
    num_threads(num_threads), fix_ng_bug(fix_ng_bug), screen(screen), MT(random_seed) {

    destroy_weights.assign(DESTORY_COUNT,1);

    // if (intersections.empty())
    // {
    for (int i = 0; i < instance.map_size; i++)
    {
        if (!instance.isObstacle(i) && instance.getDegree(i) > 2)
            intersections.push_back(i);
    }
    // }

    tabu_list_list.resize(num_threads);
    neighbors.resize(num_threads);

}

void NeighborGenerator::reset() {
    destroy_weights.assign(DESTORY_COUNT,1);
    for (auto & tabu_list:tabu_list_list) {
        tabu_list.clear();
    }
}

void NeighborGenerator::update(Neighbor & neighbor){
    if (ALNS) // update destroy heuristics
    {
        if (neighbor.old_sum_of_costs > neighbor.sum_of_costs ) {
            if (fix_ng_bug) {
                destroy_weights[neighbor.selected_neighbor] =
                        reaction_factor * (neighbor.old_sum_of_costs - neighbor.sum_of_costs)/neighbor.agents.size()
                        + (1 - reaction_factor) * destroy_weights[neighbor.selected_neighbor];
            } else {
                destroy_weights[neighbor.selected_neighbor] =
                        reaction_factor * std::max((neighbor.old_sum_of_costs - neighbor.sum_of_costs)/neighbor.agents.size(),1.0f)
                        + (1 - reaction_factor) * destroy_weights[neighbor.selected_neighbor];
            }
        }
        else
            destroy_weights[neighbor.selected_neighbor] =
                    (1 - decay_factor) * destroy_weights[neighbor.selected_neighbor];
    }
}

void NeighborGenerator::generate_parallel(const TimeLimiter & time_limiter) {
    // TODO(rivers): since here we use parallel, we need a random seed generator that support parallelization, that is why we only could use rand() but not rand(), but we can fix this later.
    #pragma omp parallel for
    for (int i = 0; i < num_threads; i++) {
        generate(time_limiter,i);
    }
}

Neighbor NeighborGenerator::generate(const TimeLimiter & time_limiter,int idx) {
    std::shared_ptr<Neighbor> neighbor_ptr = std::make_shared<Neighbor>();
    Neighbor & neighbor = *neighbor_ptr;
    //Neighbor neighbor;
    // cout<<"start generate neighbor"<<endl;
    bool succ=false;
    while (!succ){
        if (time_limiter.timeout())
            break;

        if (ALNS)
            chooseDestroyHeuristicbyALNS();

        // ONLYDEV(g_timer.record_p("generate_neighbor_s");)
        switch (destroy_strategy)
        {
            case RANDOMWALK:
                {
                    succ = generateNeighborByRandomWalk(neighbor,idx);
                    neighbor.selected_neighbor = 0;
                    break;
                }
            case INTERSECTION:
                {
                    succ = generateNeighborByIntersection(neighbor);
                    neighbor.selected_neighbor = 1;
                    break;
                }
            case RANDOMAGENTS:
                // TODO(rivers): this implementation is too bad
                // neighbor.agents.resize(agents.size());
                // for (int i = 0; i < (int)agents.size(); i++)
                //     neighbor.agents[i] = i;
                // if (neighbor.agents.size() > neighbor_size)
                // {
                //     std::random_shuffle(neighbor.agents.begin(), neighbor.agents.end());
                //     neighbor.agents.resize(neighbor_size);
                // }
                // succ = true;
                // neighbor.selected_neighbor = 2;
                {
                    auto s=std::set<int>();
                    while (s.size()<neighbor_size) {
                        s.insert(rand()%agents.size());
                    }
                    for (auto i:s) {
                        neighbor.agents.push_back(i);
                    } 
                    succ = true;
                    neighbor.selected_neighbor = 2;
                    break;
                }
            default:
                cerr << "Wrong neighbor generation strategy" << endl;
                exit(-1);
        }
        // ONLYDEV(g_timer.record_d("generate_neighbor_s","generate_neighbor_e","generate_neighbor");)

        if (!succ) {      
            // cout<<"generate neighbors failed"<<endl;  
            // TODO: we need to count how many times we failed to generate a neighbor
            if (screen>=1)
                cerr<<"generate neighbors failed"<<endl;
        }
    }

    // #pragma omp critical
    // {
    //     for (auto aid:neighbor.agents) {
    //         if (aid<0 || aid>instance.num_of_agents) {
    //             cout<<aid<<" "<<neighbor.selected_neighbor<<endl;
    //             for (auto a:neighbor.agents) {
    //                 cout<<a<<" ";
    //             }
    //             cout<<time_limiter.timeout()<<endl;
    //             break;
    //         }
    //     }
    // }

    neighbors[idx]=neighbor_ptr;

    return neighbor;


}

void NeighborGenerator::chooseDestroyHeuristicbyALNS() {
    int selected_neighbor=rouletteWheel();
    switch (selected_neighbor)
    {
        case 0 : destroy_strategy = RANDOMWALK; break;
        case 1 : destroy_strategy = INTERSECTION; break;
        case 2 : destroy_strategy = RANDOMAGENTS; break;
        default : cerr << "ERROR" << endl; exit(-1);
    }
}

int NeighborGenerator::rouletteWheel()
{
    double sum = 0;
    for (const auto& h : destroy_weights)
        sum += h;
    if (screen >= 2)
    {
        cout << "destroy weights = ";
        for (const auto& h : destroy_weights)
            cout << h / sum << ",";
        cout << endl;
    }
    double r = (double) rand() / RAND_MAX;
    double threshold = destroy_weights[0];
    int selected_neighbor = 0;
    while (threshold < r * sum)
    {
        selected_neighbor++;
        threshold += destroy_weights[selected_neighbor];
    }
    return selected_neighbor;
}

bool NeighborGenerator::generateNeighborByRandomWalk(Neighbor & neighbor, int idx) {
    if (neighbor_size >= (int)agents.size())
    {
        neighbor.agents.resize(agents.size());
        for (int i = 0; i < (int)agents.size(); i++)
            neighbor.agents[i] = i;
        return true;
    }

    int a = findMostDelayedAgent(idx);
    if (a < 0)
        return false;
    
    set<int> neighbors_set;
    neighbors_set.insert(a);
    randomWalk(a, 0, neighbors_set, neighbor_size);

    // TODO(rivers): we iterate for at most 10 iterations (not shown in the pseudo-code) to 
    // address the situation where the agent density is too low for us to collect N agents
    int count = 0;
    while (neighbors_set.size() < neighbor_size && count < 10) {
        int t = rand() % agents[a].path.size();
        randomWalk(a, t, neighbors_set, neighbor_size);
        count++;
        // select the next agent randomly
        int idx = rand() % neighbors_set.size();
        int i = 0;
        for (auto n : neighbors_set)
        {
            if (i == idx)
            {
                a = i;
                break;
            }
            i++;
        }
    }

    // #pragma omp critical
    // {
    //     for (auto aid:neighbors_set) {
    //         if (aid<0 || aid>=instance.num_of_agents) {
    //             cout<<"err"<<aid<<endl;
    //             for (auto a:neighbors_set) {
    //                 cout<<a<<" ";
    //             }
    //             cout<<endl;
    //             break;
    //         }
    //     }
    // }

    if (neighbors_set.size() < 2)
        return false;

    neighbor.agents.assign(neighbors_set.begin(), neighbors_set.end());
    if (screen >= 2)
        cout << "Generate " << neighbor.agents.size() << " neighbors by random walks of agent " << a
#ifndef NO_ROT
             << "(" << HT->get(agents[a].getStartLocation(),agents[a].getStartOrientation(),agents[a].getGoalLocation())
#else
             << "(" << HT->get(agents[a].id,agents[a].getStartLocation())
#endif
             << "->" << agents[a].path.size() - 1 << ")" << endl;

    return true;
}

bool NeighborGenerator::generateNeighborByIntersection(Neighbor & neighbor) {
    set<int> neighbors_set;
    auto pt = intersections.begin();
    std::advance(pt, rand() % intersections.size());
    int location = *pt;
    path_table.get_agents(neighbors_set, neighbor_size, location);
    if (neighbors_set.size() < neighbor_size)
    {
        set<int> closed;
        closed.insert(location);
        std::queue<int> open;
        open.push(location);
        while (!open.empty() && (int) neighbors_set.size() < neighbor_size)
        {
            int curr = open.front();
            open.pop();
            for (auto next : instance.getNeighbors(curr))
            {
                if (closed.count(next) > 0)
                    continue;
                open.push(next);
                closed.insert(next);
                if (instance.getDegree(next) >= 3)
                {
                    path_table.get_agents(neighbors_set, neighbor_size, next);
                    if ((int) neighbors_set.size() == neighbor_size)
                        break;
                }
            }
        }
    }
    neighbor.agents.assign(neighbors_set.begin(), neighbors_set.end());
    if (neighbor.agents.size() > neighbor_size)
    {
        std::shuffle(neighbor.agents.begin(), neighbor.agents.end(),MT);
        neighbor.agents.resize(neighbor_size);
    }
    if (screen >= 2)
        cout << "Generate " << neighbor.agents.size() << " neighbors by intersection " << location << endl;
    return true;
}

int NeighborGenerator::findMostDelayedAgent(int idx){
    int a = -1;
    float max_delays = -1;
    auto & tabu_list=tabu_list_list[idx];
    for (int i = 0; i < agents.size(); i++)
    {
        // TODO(rivers): currently we just use index to split threads
        if (i%num_threads!=idx) continue;
        if (tabu_list.find(i) != tabu_list.end())
            continue;
        if ((*agent_infos)[i].disabled) {
            tabu_list.insert(i);
            continue;
        }
        float delays = agents[i].getNumOfDelays();
        if (max_delays < delays)
        {
            a = i;
            max_delays = delays;
        }
    }
    if (max_delays == 0)
    {
        tabu_list.clear();
        return -1;
    }
    tabu_list.insert(a);
    // TODO(rivers): this is a bug
    if (tabu_list.size() == (agents.size()/num_threads))
        tabu_list.clear();
    return a;
}

std::list<std::pair<int,int> > NeighborGenerator::getSuccessors(int pos, int orient) {

#ifndef NO_ROT
    std::list<std::pair<int,int> > successors;

    int & cols=instance.num_of_cols;
    int & rows=instance.num_of_rows;
    auto & map=instance.my_map;

    int x=pos%(cols);
    int y=pos/(cols);

    // FW
    int next_pos;
    int next_orient=orient;
    if (orient==0) {
        // east
        if (x+1<cols){
            next_pos=pos+1;
            if (map[next_pos]==0) {
                successors.emplace_back(next_pos,next_orient);
            }
        }
    } else if (orient==1) {
        // south
        if (y+1<rows) {
            next_pos=pos+cols;
            if (map[next_pos]==0) {
                successors.emplace_back(next_pos,next_orient);
            }
        }
    } else if (orient==2) {
        // west
        if (x-1>=0) {
            next_pos=pos-1;
            if (map[next_pos]==0) {
                successors.emplace_back(next_pos,next_orient);
            }
        }
    } else if (orient==3) {
        // north
        if (y-1>=0) {
            next_pos=pos-cols;
            if (map[next_pos]==0) {
                successors.emplace_back(next_pos,next_orient);
            }
        }
    } else {
        std::cerr<<"NeighborGenerator: invalid orient: "<<orient<<endl;
        exit(-1);
    }


    // for actions that don't change the position
    next_pos=pos;

    // CR
    next_orient=(orient+1+n_orients)%n_orients;
    successors.emplace_back(next_pos, next_orient);

    // CCR
    next_orient=(orient-1+n_orients)%n_orients;
    successors.emplace_back(next_pos, next_orient);

    // W
    next_orient=orient;
    successors.emplace_back(next_pos, next_orient);
    
    return successors;
#else

    std::list<std::pair<int,int> > successors;

    int & cols=instance.num_of_cols;
    int & rows=instance.num_of_rows;
    auto & map=instance.my_map;

    int x=pos%(cols);
    int y=pos/(cols);

    int next_pos;
    int next_orient=orient;

    // R
    if (x+1<cols){
        next_pos=pos+1;
        if (map[next_pos]==0) {
            successors.emplace_back(next_pos,next_orient);
        }
    }
    // D
    if (y+1<rows) {
        next_pos=pos+cols;
        if (map[next_pos]==0) {
            successors.emplace_back(next_pos,next_orient);
        }
    }
    // L
    if (x-1>=0) {
        next_pos=pos-1;
        if (map[next_pos]==0) {
            successors.emplace_back(next_pos,next_orient);
        }
    }
    // U
    if (y-1>=0) {
        next_pos=pos-cols;
        if (map[next_pos]==0) {
            successors.emplace_back(next_pos,next_orient);
        }
    }

    // wait
    next_pos=pos;
    successors.emplace_back(next_pos,next_orient);

    return successors;

#endif
}

// a random walk with path that is shorter than upperbound and has conflicting with neighbor_size agents
void NeighborGenerator::randomWalk(int agent_id, int start_timestep, set<int>& conflicting_agents, int neighbor_size)
{
    auto & path = agents[agent_id].path;
    int loc = path[start_timestep].location;
    int orient = path[start_timestep].orientation;
    auto & agent = agents[agent_id];

    // if (fix_ng_bug) {
        float partial_path_cost=agent.getEstimatedPathLength(path,agent.getGoalLocation(),HT, false, start_timestep);
        // int slack=1;
        for (int t = start_timestep; t < path.size(); ++t)
        {
            auto successors=getSuccessors(loc,orient);
            while (!successors.empty())
            {
                int step = rand() % successors.size();
                auto iter = successors.begin();
                advance(iter, step);

                int next_loc = iter->first;
                int next_orient = iter->second;
                
                float action_cost = agent.get_action_cost(loc, orient, next_loc, next_orient, HT);
#ifndef NO_ROT
                float next_h_val = HT->get(next_loc, next_orient,instance.goal_locations[agent_id]);
#else
                float next_h_val = HT->get(agent_id, next_loc);
#endif
                // if we can find a path with a smaller distance, we try to see who becomes the obstacle.
                // TODO(rivers): this is not correct if we have weighted distance map
                if (partial_path_cost + action_cost + next_h_val < path.path_cost) // move to this location
                {
                    path_table.getConflictingAgents(agent_id, conflicting_agents, loc, next_loc, t + 1);
                    loc = next_loc;
                    orient = next_orient;
                    partial_path_cost += action_cost;
                    break;
                }
                successors.erase(iter);
            }
            if (successors.empty() || conflicting_agents.size() >= neighbor_size)
                break;
        }
    // } else {
    //     // float partial_path_cost=agent.getEstimatedPathLength(path,agent.getGoalLocation(),HT, false, start_timestep);
    //     // int slack=1;
    //     for (int t = start_timestep; t < path.size(); ++t)
    //     {
    //         auto successors=getSuccessors(loc,orient);
    //         while (!successors.empty())
    //         {
    //             int step = rand() % successors.size();
    //             auto iter = successors.begin();
    //             advance(iter, step);

    //             int next_loc = iter->first;
    //             int next_orient = iter->second;
                
    //             // float action_cost = agent.get_action_cost(loc, orient, next_loc, next_orient, HT);
    //             float next_h_val = HT->get(next_loc, next_orient,instance.goal_locations[agent_id]);
    //             // if we can find a path with a smaller distance, we try to see who becomes the obstacle.
    //             // TODO(rivers): this is not correct if we have weighted distance map
    //             if (t + next_h_val < path.path_cost) // move to this location
    //             {
    //                 path_table.getConflictingAgents(agent_id, conflicting_agents, loc, next_loc, t + 1);
    //                 loc = next_loc;
    //                 orient = next_orient;
    //                 // partial_path_cost += action_cost;
    //                 break;
    //             }
    //             successors.erase(iter);
    //         }
    //         if (successors.empty() || conflicting_agents.size() >= neighbor_size)
    //             break;
    //     }
    // }
}

}

}