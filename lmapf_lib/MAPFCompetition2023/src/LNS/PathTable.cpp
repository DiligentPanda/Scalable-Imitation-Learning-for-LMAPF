#include "LNS/PathTable.h"
#include "util/MyLogger.h"

namespace LNS {

void PathTable::insertPath(int agent_id, const Path& _path, bool verbose)
{

    if (verbose) {
        std::cerr<<"insertPath for agent "<<agent_id<<" paths: ";
        for (auto p : _path) {
            std::cerr<<p.location<<" ";
        }
        std::cerr<<std::endl;
    }

    if (_path.empty())
        return;
    
    int T = (int) _path.size();
    if (window_size>0 && window_size+1<_path.size()) {
        T = window_size+1;
    }
    Path path(_path.begin(), _path.begin() + T);

    for (int t = 0; t < (int)path.size(); t++)
    {
        if (table[path[t].location].size() <= t)
            table[path[t].location].resize(t + 1, NO_AGENT);
        // assert(table[path[t].location][t] == NO_AGENT);
        table[path[t].location][t] = agent_id;
    }
    assert(goals[path.back().location] == MAX_TIMESTEP);
    goals[path.back().location] = (int) path.size() - 1;
    makespan = max(makespan, (int) path.size() - 1);
}

void PathTable::deletePath(int agent_id, const Path& _path,bool verbose)
{
    if (verbose) {
        std::cerr<<"deletePath for agent "<<agent_id<<" paths: ";
        for (auto p : _path) {
            std::cerr<<p.location<<" ";
        }
        std::cerr<<std::endl;
    }

    if (_path.empty())
        return;
    
    int T = (int) _path.size();
    if (window_size>0 && window_size+1<_path.size()) {
        T = window_size+1;
    }
    Path path(_path.begin(), _path.begin() + T);
    
    for (int t = 0; t < (int)path.size() ; t++)
    {
        assert(table[path[t].location].size() > t && table[path[t].location][t] == agent_id);
        table[path[t].location][t] = NO_AGENT;
    }
    goals[path.back().location] = MAX_TIMESTEP;

    // TODO(rivers): when we use window size, we ignore this for now, maybe we can use a better data structure such as ordered_set/heap to maintain the makespan. 
    // if (makespan == (int) path.size() - 1) // re-compute makespan
    // {
    //     makespan = 0;
    //     for (int time : goals)
    //     {
    //         if (time < MAX_TIMESTEP && time > makespan)
    //             makespan = time;
    //     }

    // }
}


void PathTable::insertPath(int agent_id, const Parallel::Path& _path, bool verbose)
{

    if (verbose) {
        std::cerr<<"insertPath for agent "<<agent_id<<" paths: ";
        for (auto p : _path.nodes) {
            std::cerr<<p.location<<","<<p.orientation<<" ";
        }
        std::cerr<<std::endl;
    }

    if (_path.nodes.empty())
        return;
    
    int T = (int) _path.nodes.size();
    if (window_size>0 && window_size+1<_path.nodes.size()) {
        T = window_size+1;
    }

    auto & path=_path.nodes;


    for (int t = 0; t < T; t++)
    {
        if (table[path[t].location].size() <= t)
            table[path[t].location].resize(t + 1, NO_AGENT);
        // assert(table[path[t].location][t] == NO_AGENT);
            
        // cerr<<t<<","<<table[path[t].location][t]<<" ";
        table[path[t].location][t] = agent_id;
    }
    // cerr<<endl;

    // TODO(rivers): check whether we need maintain goals and makespan in the life-long setting
    // assert(goals[path.back().location] == MAX_TIMESTEP);
    // goals[path.back().location] = T-1;
    // makespan = max(makespan, T-1);
}

void PathTable::deletePath(int agent_id, const Parallel::Path& _path,bool verbose)
{
    if (verbose) {
        std::cerr<<"deletePath for agent "<<agent_id<<" paths: ";
        for (auto p : _path.nodes) {
            std::cerr<<p.location<<","<<p.orientation<<" ";
        }
        std::cerr<<std::endl;
    }

    if (_path.nodes.empty())
        return;
    
    int T = (int) _path.nodes.size();
    if (window_size>0 && window_size+1<_path.nodes.size()) {
        T = window_size+1;
    }

    auto & path=_path.nodes;
    
    for (int t = 0; t < T ; t++)
    {
        assert(table[path[t].location].size() > t && table[path[t].location][t] == agent_id);
        table[path[t].location][t] = NO_AGENT;
    }
    // TODO(rivers): check whether we need maintain goals and makespan  in the life-long setting
    // goals[path.back().location] = MAX_TIMESTEP;

    // TODO(rivers): when we use window size, we ignore this for now, maybe we can use a better data structure such as ordered_set/heap to maintain the makespan. 
    // if (makespan == (int) path.size() - 1) // re-compute makespan
    // {
    //     makespan = 0;
    //     for (int time : goals)
    //     {
    //         if (time < MAX_TIMESTEP && time > makespan)
    //             makespan = time;
    //     }

    // }
}





bool PathTable::constrained(int from, int to, int to_time) const
{
    if (!table.empty())
    {
        if (table[to].size() > to_time && table[to][to_time] != NO_AGENT)
            return true;  // vertex conflict with agent table[to][to_time]
        else if (table[to].size() >= to_time && table[from].size() > to_time && !table[to].empty() &&
                 table[to][to_time - 1] != NO_AGENT && table[from][to_time] == table[to][to_time - 1])
            return true;  // edge conflict with agent table[to][to_time - 1]
    }
    // if (!goals.empty())
    // {
    //     if (goals[to] <= to_time)
    //         return true; // target conflict
    // }
    return false;
}

bool PathTable::constrained(int from, int to, int to_time, std::vector<int> & ignored_agents) const
{
    if (!table.empty())
    {
        if (table[to].size() > to_time && table[to][to_time] != NO_AGENT) {
            bool found=false;
            for (int i=0;i<ignored_agents.size();++i) {
                if (table[to][to_time]==ignored_agents[i]) {
                    found=true;
                    break;
                }
            }
            if (!found) {
                return true;  // vertex conflict with agent table[to][to_time]
            }
        }
        
        if (table[to].size() >= to_time && table[from].size() > to_time && !table[to].empty() &&
                 table[to][to_time - 1] != NO_AGENT && table[from][to_time] == table[to][to_time - 1]) {
            bool found=false;
            for (int i=0;i<ignored_agents.size();++i) {
                if (table[to][to_time-1]==ignored_agents[i]) {
                    found=true;
                    break;
                }
            }
            if (!found) {
                return true;  // edge conflict with agent table[to][to_time - 1]
            }
        }
    }

    // TODO(rivers): we just ignore this for now in our lifelong setting with a planning window.
    // if (!goals.empty())
    // {
    //     if (goals[to] <= to_time)
    //         return true; // target conflict
    // }
    return false;
}

void PathTable::getConflictingAgents(int agent_id, set<int>& conflicting_agents, int from, int to, int to_time) const
{
    if (table.empty())
        return;
    if (table[to].size() > to_time && table[to][to_time] != NO_AGENT)
        conflicting_agents.insert(table[to][to_time]); // vertex conflict
    if (table[to].size() >= to_time && table[from].size() > to_time &&
        table[to][to_time - 1] != NO_AGENT && table[from][to_time] == table[to][to_time - 1])
        conflicting_agents.insert(table[from][to_time]); // edge conflict
    // TODO: collect target conflicts as well.
}

void PathTable::get_agents(set<int>& conflicting_agents, int loc) const
{
    if (loc < 0)
        return;
    for (auto agent : table[loc])
    {
        if (agent >= 0)
            conflicting_agents.insert(agent);
    }
}

void PathTable::get_agents(set<int>& conflicting_agents, int neighbor_size, int loc) const
{
    if (loc < 0 || table[loc].empty())
        return;
    int t_max = (int) table[loc].size() - 1;
    while (table[loc][t_max] == NO_AGENT && t_max > 0)
        t_max--;
    if (t_max == 0)
        return;
    int t0 = rand() % t_max;
    if (table[loc][t0] != NO_AGENT)
        conflicting_agents.insert(table[loc][t0]);
    int delta = 1;
    while (t0 - delta >= 0 || t0 + delta <= t_max)
    {
        if (t0 - delta >= 0 && table[loc][t0 - delta] != NO_AGENT)
        {
            conflicting_agents.insert(table[loc][t0 - delta]);
            if((int) conflicting_agents.size() == neighbor_size)
                return;
        }
        if (t0 + delta <= t_max && table[loc][t0 + delta] != NO_AGENT)
        {
            conflicting_agents.insert(table[loc][t0 + delta]);
            if((int) conflicting_agents.size() == neighbor_size)
                return;
        }
        delta++;
    }
}

// get the holding time after the earliest_timestep for a location
int PathTable::getHoldingTime(int location, int earliest_timestep = 0) const
{
    if (table.empty() or (int) table[location].size() <= earliest_timestep)
        return earliest_timestep;
    int rst = (int) table[location].size();
    while (rst > earliest_timestep and table[location][rst - 1] == NO_AGENT)
        rst--;
    return rst;
}

void PathTableWC::insertPath(int agent_id, const Path& path)
{
    paths[agent_id] = &path;
    if (path.empty())
        return;
    for (int t = 0; t < (int)path.size(); t++)
    {
        if (table[path[t].location].size() <= t)
            table[path[t].location].resize(t + 1);
        table[path[t].location][t].push_back(agent_id);
    }
    assert(goals[path.back().location] == MAX_TIMESTEP);
    goals[path.back().location] = (int) path.size() - 1;
    makespan = max(makespan, (int) path.size() - 1);
}
void PathTableWC::insertPath(int agent_id)
{
    assert(paths[agent_id] != nullptr);
    insertPath(agent_id, *paths[agent_id]);
}
void PathTableWC::deletePath(int agent_id)
{
    const Path & path = *paths[agent_id];
    if (path.empty())
        return;
    for (int t = 0; t < (int)path.size(); t++)
    {
        assert(table[path[t].location].size() > t &&
               std::find (table[path[t].location][t].begin(), table[path[t].location][t].end(), agent_id)
               != table[path[t].location][t].end());
        table[path[t].location][t].remove(agent_id);
    }
    goals[path.back().location] = MAX_TIMESTEP;
    if (makespan == (int) path.size() - 1) // re-compute makespan
    {
        makespan = 0;
        for (int time : goals)
        {
            if (time < MAX_TIMESTEP && time > makespan)
                makespan = time;
        }

    }
}

int PathTableWC::getFutureNumOfCollisions(int loc, int time) const
{
    assert(goals[loc] == MAX_TIMESTEP);
    int rst = 0;
    if (!table.empty() && (int)table[loc].size() > time)
    {
        for (int t = time + 1; t < (int)table[loc].size(); t++)
            rst += (int)table[loc][t].size();  // vertex conflict
    }
    return rst;
}

int PathTableWC::getNumOfCollisions(int from, int to, int to_time) const
{
    int rst = 0;
    if (!table.empty())
    {
        if ((int)table[to].size() > to_time)
            rst += (int)table[to][to_time].size();  // vertex conflict
        if (from != to && table[to].size() >= to_time && table[from].size() > to_time)
        {
            for (auto a1 : table[to][to_time - 1])
            {
                for (auto a2: table[from][to_time])
                {
                    if (a1 == a2)
                        rst++; // edge conflict
                }
            }
        }
    }
    if (!goals.empty())
    {
        if (goals[to] < to_time)
            rst++; // target conflict
    }
    return rst;
}
bool PathTableWC::hasCollisions(int from, int to, int to_time) const
{
    if (!table.empty())
    {
        if ((int)table[to].size() > to_time and !table[to][to_time].empty())
            return true; // vertex conflict
        if (from != to && table[to].size() >= to_time && table[from].size() > to_time)
        {
            for (auto a1 : table[to][to_time - 1])
            {
                for (auto a2: table[from][to_time])
                {
                    if (a1 == a2)
                        return true; // edge conflict
                }
            }
        }
    }
    if (!goals.empty())
    {
        if (goals[to] < to_time)
            return true; // target conflict
    }
    return false;
}
bool PathTableWC::hasEdgeCollisions(int from, int to, int to_time) const
{
    if (!table.empty() && from != to && table[to].size() >= to_time && table[from].size() > to_time)
    {
        for (auto a1 : table[to][to_time - 1])
        {
            for (auto a2: table[from][to_time])
            {
                if (a1 == a2)
                    return true; // edge conflict
            }
        }
    }
    return false;
}

int PathTableWC::getAgentWithTarget(int target_location, int latest_timestep) const
{
    if (table.empty() or goals.empty() or goals[target_location] > latest_timestep)
        return -1;
    for (auto id : table[target_location][goals[target_location]]) // look at all agents at the goal time
    {
        if (paths[id]->back().location == target_location) // if agent id's goal is to, then this is the agent we want
        {
            return id;
        }
    }
    assert(false); // this should never happen
    return -1;
}

int PathTableWC::getLastCollisionTimestep(int location) const
{
    if (table.empty())
        return -1;
    for (int t = (int)table[location].size() - 1; t >= 0; t--)
    {
        if (!table[location][t].empty())
            return t;
    }
    return -1;
}

void PathTableWC::clear()
{
    table.clear();
    goals.clear();
    paths.clear();
}
} // end namespace LNS
