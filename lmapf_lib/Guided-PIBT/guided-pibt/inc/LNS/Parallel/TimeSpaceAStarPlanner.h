#pragma once
#include "LNS/Parallel/TimeSpaceAStarState.h"
#include "boost/unordered_set.hpp"
#include "LNS/Instance.h"
#include <utility>
#include "LNS/Parallel/DataStructure.h"
#include "util/TimeLimiter.h"
#include "LNS/PathTable.h"

namespace LNS {

namespace Parallel {

class TimeSpaceAStarPlanner {
public:
    Instance & instance;
    std::shared_ptr<TrafficMAPF::Dist2PathHeuristicTable> HT;
    std::shared_ptr<vector<float> > weights;
    int execution_window;

    boost::heap::pairing_heap<TimeSpaceAStarState*, boost::heap::compare<TimeSpaceAStarState::Compare> > open_list;
    boost::unordered_set<TimeSpaceAStarState*, TimeSpaceAStarState::Hash, TimeSpaceAStarState::Equal> all_states;

    std::vector<TimeSpaceAStarState*> successors;

    int n_expanded;
    int n_generated;

    // results
    Path path;

    static const int n_dirs=5; // east, south, west, north, stay
    static const int n_orients=4; // east, south, west, north

    TimeSpaceAStarPlanner(Instance & instance, std::shared_ptr<TrafficMAPF::Dist2PathHeuristicTable> HT, std::shared_ptr<vector<float> > weights, int execution_window);
    void findPath(int agent_idx, int start_pos, int start_orient, int goal_pos, PathTable & path_table, const TimeLimiter & time_limiter);
    void clear();
    void buildPath(TimeSpaceAStarState * curr, int goal_pos);
    void getSuccessors(int agent_idx, TimeSpaceAStarState * state, int goal_pos, PathTable & path_table);
};

}

} // namespace LNS