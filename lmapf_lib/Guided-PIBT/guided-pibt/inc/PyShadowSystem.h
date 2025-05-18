#pragma once
#include "CompetitionSystem.h"
#include <memory>

class PyShadowSystem{
public:
    Logger * logger=nullptr;
    InfAssignSystem * system=nullptr;
    MAPFPlanner * planner=nullptr;
    Grid * grid=nullptr;
    ActionModelWithRotate * model=nullptr;

    std::vector<int> agents;
    std::vector<int> tasks;

    PyShadowSystem(const string & map_fp);
    PyShadowSystem(const string & map_name, const std::vector<int> & map, int h, int w);

    // TODO: do we need priorities?
    void reset(std::vector<int> & agents, std::vector<int> & tasks, int seed);
    void step(std::vector<int> & actions);
    std::vector<int> query();


    void sync(
        int timestep,
        std::vector<int> & curr_locations,
        std::vector<int> & goal_locations
    );
    std::vector<int> query_locations();
    std::vector<int> query_goals();
    std::vector<float> query_heuristics(
        std::vector<int> & locs, 
        std::vector<int> & view_y,
        std::vector<int> & view_x
    );
    std::vector<int> query_pibt_actions();
    std::vector<int> query_lns_actions(std::vector<int> & init_plan, int planning_window, int num_theads, int max_iterations);

    void seed(uint _seed);
    ~PyShadowSystem();
};