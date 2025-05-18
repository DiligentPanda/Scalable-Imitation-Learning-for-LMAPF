#pragma once

#ifndef NO_ROT

#include <ctime>
#include "SharedEnv.h"
#include "ActionModel.h"
#include "RHCR/interface/RHCRSolver.h"
#include "nlohmann/json.hpp"
#include "RHCR/main/SingleAgentSolver.h"
#include "RHCR/main/MAPFSolver.h"
#include "RHCR/main/WHCAStar.h"
#include "RHCR/main/ECBS.h"
#include "RHCR/main/LRAStar.h"
#include "RHCR/main/PBS.h"
#include "RHCR/main/ID.h"
#include "RHCR/interface/RHCRSolver.h"
#include "RHCR/interface/CompetitionGraph.h"
#include "common.h"
#include <memory>
#include "LaCAM2/LaCAM2Solver.hpp"
#include "LNS/LNSSolver.h"

class MAPFPlanner
{
public:

    SharedEnvironment* env;

	MAPFPlanner(SharedEnvironment* env): env(env){};
    MAPFPlanner(){env = new SharedEnvironment();};
	virtual ~MAPFPlanner(){delete env;};


    virtual void initialize(int preprocess_time_limit);

    // return next states for all agents
    virtual void plan(int time_limit, std::vector<Action> & plan);

    // Start kit dummy implementation
    std::list<pair<int,int>>single_agent_plan(int start,int start_direct, int end);
    int getManhattanDistance(int loc1, int loc2);
    std::list<pair<int,int>> getNeighbors(int location, int direction);
    bool validateMove(int loc,int loc2);

    bool consider_rotation=true;
    string lifelong_solver_name;
    std::shared_ptr<RHCR::RHCRSolver> rhcr_solver;
    std::shared_ptr<LaCAM2::LaCAM2Solver> lacam2_solver;
    std::shared_ptr<LNS::LNSSolver> lns_solver;

    double max_step_time=0;
    int max_execution_steps;

    std::shared_ptr<std::vector<float> > map_weights;
    nlohmann::json config;
    void load_configs();
    std::string load_map_weights(string weights_path);

    RHCR::MAPFSolver* rhcr_build_mapf_solver(nlohmann::json & config, RHCR::CompetitionGraph & graph);
    void rhcr_config_solver(std::shared_ptr<RHCR::RHCRSolver> & solver,nlohmann::json & config);

};

#else

#include <ctime>
#include "SharedEnv.h"
#include "ActionModel.h"
#include "nlohmann/json.hpp"
#include "common.h"
#include <memory>
#include "LaCAM2/LaCAM2Solver.hpp"
#include "LNS/LNSSolver.h"

class MAPFPlanner
{
public:

    SharedEnvironment* env;

	MAPFPlanner(SharedEnvironment* env): env(env){};
    MAPFPlanner(){env = new SharedEnvironment();};
	virtual ~MAPFPlanner(){delete env;};


    virtual void initialize(int preprocess_time_limit);

    // return next states for all agents
    virtual void plan(int time_limit, std::vector<Action> & plan);

    // Start kit dummy implementation
    std::list<pair<int,int>>single_agent_plan(int start,int start_direct, int end);
    int getManhattanDistance(int loc1, int loc2);
    std::list<pair<int,int>> getNeighbors(int location, int direction);
    bool validateMove(int loc,int loc2);

    bool consider_rotation=true;
    string lifelong_solver_name;
    std::shared_ptr<LaCAM2::LaCAM2Solver> lacam2_solver;
    std::shared_ptr<LNS::LNSSolver> lns_solver;

    double max_step_time=0;
    int max_execution_steps;

    std::shared_ptr<std::vector<float> > map_weights;
    nlohmann::json config;
    void load_configs();
    std::string load_map_weights(string weights_path);
};

#endif