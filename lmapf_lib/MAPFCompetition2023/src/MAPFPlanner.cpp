#ifndef NO_ROT

#include <MAPFPlanner.h>
#include <random>
#include "RHCR/interface/CompetitionGraph.h"
#include "util/HeuristicTable.h"
#include "util/Analyzer.h"
#include "util/MyLogger.h"
#include "boost/format.hpp"
#include "util/MyCommon.h"
#include "util/TimeLimiter.h"

struct AstarNode {
    int location;
    int direction;
    int f,g,h;
    AstarNode* parent;
    int t = 0;
    bool closed = false;
    AstarNode(int _location,int _direction, int _g, int _h, AstarNode* _parent):
        location(_location), direction(_direction),f(_g+_h),g(_g),h(_h),parent(_parent) {}
    AstarNode(int _location,int _direction, int _g, int _h, int _t, AstarNode* _parent):
        location(_location), direction(_direction),f(_g+_h),g(_g),h(_h),t(_t),parent(_parent) {}
};


struct cmp {
    bool operator()(AstarNode* a, AstarNode* b) {
        if(a->f == b->f) return a->g <= b->g;
        else return a->f > b->f;
    }
};

void MAPFPlanner::load_configs() {
    // load configs
	string config_path="configs/"+env->map_name.substr(0,env->map_name.find_last_of("."))+".json";
    char * _config_path=getenv("CONFIG_PATH");
    if (_config_path!=NULL) {
        config_path=std::string(_config_path);
        std::cout<<"load config from "<<config_path<<std::endl;
    }
    std::ifstream f(config_path);
    try
    {
        config = nlohmann::json::parse(f);

        char * env_weight_path=getenv("MAP_WEIGHT_PATH");
        if (env_weight_path!=NULL) {
            config["map_weights_path"]=env_weight_path;
            std::cout<<"load weight from "<<env_weight_path<<std::endl;
        }

        std::cerr<<config<<std::endl;
        config["lifelong_solver_name"]=read_conditional_value(config,"lifelong_solver_name",env->num_of_agents);
        config["map_weights_path"]=read_conditional_value(config,"map_weights_path",env->num_of_agents);

        if (config.contains("max_task_completed")) {
            config["max_task_completed"]=read_conditional_value(config,"max_task_completed",env->num_of_agents);
        }

        if (config.contains("max_execution_steps")) {
            config["max_execution_steps"]=read_conditional_value(config,"max_execution_steps",env->num_of_agents);
        }

        config["LaCAM2"]["order_strategy"]=read_conditional_value(config["LaCAM2"],"order_strategy",env->num_of_agents);
        config["LNS"]["LaCAM2"]["order_strategy"]=read_conditional_value(config["LNS"]["LaCAM2"],"order_strategy",env->num_of_agents);
        config["disable_corner_target_agents"]=read_conditional_value(config,"disable_corner_target_agents",env->num_of_agents);
        config["max_agents_in_use"]=read_conditional_value(config,"max_agents_in_use",env->num_of_agents);

        config["LNS"]["fix_ng_bug"]=read_conditional_value(config["LNS"],"fix_ng_bug",env->num_of_agents);
        config["LNS"]["window_size_for_CT"]=read_conditional_value(config["LNS"],"window_size_for_CT",env->num_of_agents);
        config["LNS"]["window_size_for_CAT"]=read_conditional_value(config["LNS"],"window_size_for_CAT",env->num_of_agents);
        config["LNS"]["window_size_for_PATH"]=read_conditional_value(config["LNS"],"window_size_for_PATH",env->num_of_agents);
        config["LNS"]["LaCAM2"]["planning_window"]=read_conditional_value(config["LNS"]["LaCAM2"],"planning_window",env->num_of_agents);


        string s=config.dump();
        std::replace(s.begin(),s.end(),',','|');
        config["details"]=s;
    }
    catch (nlohmann::json::parse_error error)
    {
        std::cout << "Failed to load " << config_path << std::endl;
        std::cout << "Message: " << error.what() << std::endl;
        exit(1);
    }
}

RHCR::MAPFSolver* MAPFPlanner::rhcr_build_mapf_solver(nlohmann::json & config, RHCR::CompetitionGraph & graph) {
    // build single agent solver
    string solver_name = read_param_json<string>(config,"single_agent_solver");
	RHCR::SingleAgentSolver* path_planner;
	RHCR::MAPFSolver* mapf_solver;
	if (solver_name == "ASTAR")
	{
		path_planner = new RHCR::StateTimeAStar();
	}
	else if (solver_name == "SIPP")
	{
		path_planner = new RHCR::SIPP();
	}
	else
	{
		cout << "Single-agent solver " << solver_name << "does not exist!" << endl;
		exit(-1);
	}

    // build multi-agent solver
	solver_name = read_param_json<string>(config,"solver");
	if (solver_name == "ECBS")
	{
		RHCR::ECBS* ecbs = new RHCR::ECBS(graph, *path_planner);
		ecbs->potential_function = read_param_json<string>(config,"potential_function");
		ecbs->potential_threshold = read_param_json<double>(config,"potential_threshold");
		ecbs->suboptimal_bound = read_param_json<double>(config,"suboptimal_bound");
		mapf_solver = ecbs;
	}
	else if (solver_name == "PBS")
	{
		RHCR::PBS* pbs = new RHCR::PBS(graph, *path_planner);
		pbs->lazyPriority = read_param_json<bool>(config,"lazyP");
        auto prioritize_start = read_param_json<bool>(config,"prioritize_start");
        auto hold_endpoints = read_param_json<bool>(config,"hold_endpoints");
        auto dummy_paths = read_param_json<bool>(config,"dummy_paths");
        if (hold_endpoints || dummy_paths)
            prioritize_start = false;
        pbs->prioritize_start = prioritize_start;
        auto CAT = read_param_json<bool>(config,"CAT");
        pbs->setRT(CAT, prioritize_start);
		mapf_solver = pbs;
	}
	else if (solver_name == "WHCA")
	{
		mapf_solver = new RHCR::WHCAStar(graph, *path_planner);
	}
	else if (solver_name == "LRA")
	{
		mapf_solver = new RHCR::LRAStar(graph, *path_planner);
	}
	else
	{
		cout << "Solver " << solver_name << " does not exist!" << endl;
		exit(-1);
	}

    auto id = read_param_json<bool>(config,"id");
	if (id)
	{
		return new RHCR::ID(graph, *path_planner, *mapf_solver);
	}
	else
	{
		return mapf_solver;
	}
}

void MAPFPlanner::rhcr_config_solver(std::shared_ptr<RHCR::RHCRSolver> & solver,nlohmann::json & config) {
    solver->outfile = read_param_json<string>(config,"output");
    solver->screen = read_param_json<int>(config,"screen");
    solver->log = read_param_json<bool>(config,"log");
    solver->num_of_drives = env->num_of_agents;
    solver->time_limit = read_param_json<int>(config,"cutoffTime");
    solver->simulation_window = read_param_json<int>(config,"simulation_window");
    solver->planning_window = read_param_json<int>(config,"planning_window");
    if (solver->simulation_window>solver->planning_window){
        cerr<<boost::format("Error: the simulation window %d can not be larger than the planning window %d!")% \
        solver->simulation_window % solver->planning_window<<endl;
        exit(1);
    }
    solver->travel_time_window = read_param_json<int>(config,"travel_time_window");
    solver->consider_rotation = read_param_json<bool>(config,"consider_rotation");
    solver->k_robust = read_param_json<int>(config,"robust");
    solver->hold_endpoints = read_param_json<bool>(config,"hold_endpoints");
    solver->useDummyPaths = read_param_json<bool>(config,"dummy_paths");
    auto seed = read_param_json<int>(config,"seed");
    if (seed>=0) {
        solver->seed = seed;
    } else {
        solver->seed = (int)time(0);
    }
    srand(solver->seed);
}

std::string MAPFPlanner::load_map_weights(string weights_path) {
    // TODO(rivers): make weights float
    // we have at least 5 weights for a location: right,down,left,up,stay
    map_weights=std::make_shared<std::vector<float> >(env->rows*env->cols*5,1);
    std::string suffix = "all_one";

    if (weights_path!=""){
        std::ifstream f(weights_path);
        try
        {
            nlohmann::json _weights = nlohmann::json::parse(f);
            if (_weights.size()!=map_weights->size()) {
                std::cerr<<"map weights size mismatch"<<std::endl;
                exit(-1);
            }

            for (int i=0;i<map_weights->size();++i){
                (*map_weights)[i]=_weights[i].get<float>();
            }
            
        }
        catch (nlohmann::json::parse_error error)
        {
            std::cerr << "Failed to load " << weights_path << std::endl;
            std::cerr << "Message: " << error.what() << std::endl;
            exit(1);
        }

        boost::filesystem::path _weights_path(weights_path);
        suffix=_weights_path.stem().string();
    }
    return suffix;
}

void MAPFPlanner::initialize(int preprocess_time_limit) {
    cout << "planner initialization begins" << endl;
    load_configs();

    ONLYDEV(
        analyzer.timestamp();
        analyzer.init_from_config(config);
        analyzer.set_dump_path(config["analysis_output"].get<string>());
        g_logger.init("logs/run");
    )

    max_execution_steps = read_param_json<int>(config,"max_execution_steps",1000000);

    std::cout<<"max execution steps: "<<max_execution_steps<<std::endl;

    std::string weights_path=read_param_json<std::string>(config,"map_weights_path");
    std::string suffix=load_map_weights(weights_path);

    lifelong_solver_name=config["lifelong_solver_name"];

    // TODO(hj): memory management is a disaster here...
    if (lifelong_solver_name=="RHCR") {
        auto graph = new RHCR::CompetitionGraph(*env);
        graph->preprocessing(consider_rotation,env->file_storage_path);
        auto mapf_solver=rhcr_build_mapf_solver(config["RHCR"],*graph);
        rhcr_solver = std::make_shared<RHCR::RHCRSolver>(*graph,*mapf_solver,env);
        rhcr_config_solver(rhcr_solver,config["RHCR"]);
        rhcr_solver->initialize(*env);
        cout<<"RHCRSolver initialized"<<endl;
    } else if (lifelong_solver_name=="LaCAM2") {
        if (!read_param_json<bool>(config["LaCAM2"],"consider_rotation")) {
            std::cerr<<"In LaCAM2, must consider rotation when compiled with NO_ROT unset"<<std::endl;
            exit(-1);
        }

        auto heuristics =std::make_shared<HeuristicTable>(env,map_weights,read_param_json<bool>(config["LaCAM2"],"use_orient_in_heuristic"));
        heuristics->preprocess(suffix);
        int max_agents_in_use=read_param_json<int>(config,"max_agents_in_use",-1);
        if (max_agents_in_use==-1) {
            max_agents_in_use=env->num_of_agents;
        }
        bool disable_corner_target_agents=read_param_json<bool>(config,"disable_corner_target_agents",false);
        int max_task_completed=read_param_json<int>(config,"max_task_completed",1000000);
        lacam2_solver = std::make_shared<LaCAM2::LaCAM2Solver>(heuristics,env,map_weights,max_agents_in_use,disable_corner_target_agents,max_task_completed,config["LaCAM2"]);
        lacam2_solver->initialize(*env);
        cout<<"LaCAMSolver2 initialized"<<endl;
    } else if (lifelong_solver_name=="LNS") {
        if (read_param_json<bool>(config["LNS"]["LaCAM2"],"consider_rotation")) {
            std::cerr<<"In LNS, must not consider rotation when compiled with NO_ROT unset"<<std::endl;
            exit(-1);
        }
        auto heuristics =std::make_shared<HeuristicTable>(env,map_weights,true);
        heuristics->preprocess(suffix);
        //heuristics->preprocess();
        int max_agents_in_use=read_param_json<int>(config,"max_agents_in_use",-1);
        if (max_agents_in_use==-1) {
            max_agents_in_use=env->num_of_agents;
        }
        bool disable_corner_target_agents=read_param_json<bool>(config,"disable_corner_target_agents",false);
        int max_task_completed=read_param_json<int>(config,"max_task_completed",1000000);
        auto lacam2_solver = std::make_shared<LaCAM2::LaCAM2Solver>(heuristics,env,map_weights,max_agents_in_use,disable_corner_target_agents,max_task_completed,config["LNS"]["LaCAM2"]);
        lacam2_solver->initialize(*env);
        lns_solver = std::make_shared<LNS::LNSSolver>(heuristics,env,map_weights,config["LNS"],lacam2_solver,max_task_completed);
        lns_solver->initialize(*env);
        cout<<"LNSSolver initialized"<<endl;
    } else if (lifelong_solver_name=="DUMMY") {
        cout<<"using dummy solver"<<endl;
    }
    else {
        cerr<<"unknown lifelong solver name"<<lifelong_solver_name<<endl;
        exit(-1);
    }

    cout << "planner initialization ends" << endl;
}


// plan using simple A* that ignores the time dimension
void MAPFPlanner::plan(int time_limit,vector<Action> & actions) 
{
    // NOTE we need to return within time_limit, but we can exploit this time duration as much as possible

    // check if we need to restart a plan task (thread)
    // if so, we need to stop the current one and then restart
    // we also need to clean the current action plan if restart

    // TODO if time_limit approachs, just return a valid move, e.g. all actions are wait.
    
    ONLYDEV(
        g_timer.record_p("_step_s");
    )

    if (env->curr_timestep>=max_execution_steps-1) {
        for (int i=0;i<env->num_of_agents;++i) {
            actions.push_back(Action::W);
        }
        return;
    }

    if (lifelong_solver_name=="RHCR") {
        cout<<"using RHCR"<<endl;
        rhcr_solver->plan(*env);
        rhcr_solver->get_step_actions(*env, actions);
    } else if (lifelong_solver_name=="LaCAM2") {
        ONLYDEV(cout<<"using LaCAM2"<<endl;)
        ONLYDEV(g_timer.record_p("mapf_lacam2_plan_s");)
        lacam2_solver->plan(*env);
        ONLYDEV(g_timer.record_d("mapf_lacam2_plan_s","mapf_lacam2_plan");)
        ONLYDEV(g_timer.record_p("mapf_lacam2_get_step_s");)
        lacam2_solver->get_step_actions(*env,actions);
        ONLYDEV(g_timer.record_d("mapf_lacam2_get_step_s","mapf_lacam2_get_step");)
    } else if (lifelong_solver_name=="LNS") {
        ONLYDEV(cout<<"using LNS"<<endl;)
        lns_solver->observe(*env);
        lns_solver->plan(*env); 
        lns_solver->get_step_actions(*env,actions);
    } else if (lifelong_solver_name=="DUMMY") {
        cout<<"using DUMMY"<<endl;
    } 
    else {
        cerr<<"unknown lifelong solver name"<<lifelong_solver_name<<endl;
        exit(-1);
    }

    ONLYDEV(
        double step_time=g_timer.record_d("_step_s","_step_e","_step");
        if (step_time>max_step_time) {
            max_step_time=step_time;
        }

        std::cerr<<"max_step_time: "<<max_step_time<<endl;
        g_timer.remove_d("_step");
    )

    // for (int i = 0; i < env->num_of_agents; i++) 
    // {
    //     list<pair<int,int>> path;
    //     if (env->goal_locations[i].empty()) 
    //     {
    //         path.push_back({env->curr_states[i].location, env->curr_states[i].orientation});
    //     } 
    //     else 
    //     {
    //         path = single_agent_plan(env->curr_states[i].location,
    //                                 env->curr_states[i].orientation,
    //                                 env->goal_locations[i].front().first);
    //     }
    //     if (path.front().first != env->curr_states[i].location)
    //     {
    //         actions[i] = Action::FW; //forward action
    //     } 
    //     else if (path.front().second!= env->curr_states[i].orientation)
    //     {
    //         int incr = path.front().second - env->curr_states[i].orientation;
    //         if (incr == 1 || incr == -3)
    //         {
    //             actions[i] = Action::CR; //C--counter clockwise rotate
    //         } 
    //         else if (incr == -1 || incr == 3)
    //         {
    //             actions[i] = Action::CCR; //CCR--clockwise rotate
    //         } 
    //     }

    // }


  return;
}

list<pair<int,int>> MAPFPlanner::single_agent_plan(int start,int start_direct,int end) {
    list<pair<int,int>> path;
    priority_queue<AstarNode*,vector<AstarNode*>,cmp> open_list;
    unordered_map<int,AstarNode*> all_nodes;
    unordered_set<int> close_list;
    AstarNode* s = new AstarNode(start, start_direct, 0, getManhattanDistance(start,end), nullptr);
    open_list.push(s);
    all_nodes[start*4 + start_direct] = s;

    while (!open_list.empty()) {
        AstarNode* curr = open_list.top();
        open_list.pop();
        close_list.emplace(curr->location*4 + curr->direction);
        if (curr->location == end) {
            while(curr->parent!=NULL) {
                path.emplace_front(make_pair(curr->location, curr->direction));
                curr = curr->parent;
            }
            break;
        }
        list<pair<int,int>> neighbors = getNeighbors(curr->location, curr->direction);
        for (const pair<int,int>& neighbor: neighbors) {
            if (close_list.find(neighbor.first*4 + neighbor.second) != close_list.end())
                continue;
            if (all_nodes.find(neighbor.first*4 + neighbor.second) != all_nodes.end()) {
                AstarNode* old = all_nodes[neighbor.first*4 + neighbor.second];
                if (curr->g + 1 < old->g) {
                    old->g = curr->g+1;
                    old->f = old->h+old->g;
                    old->parent = curr;
                }
            } else {
                AstarNode* next_node = new AstarNode(neighbor.first, neighbor.second,
                    curr->g+1,getManhattanDistance(neighbor.first,end), curr);
                open_list.push(next_node);
                all_nodes[neighbor.first*4+neighbor.second] = next_node;
            }
        }
    }
    for (auto n: all_nodes)
    {
        delete n.second;
    }
    all_nodes.clear();
    return path;
}


int MAPFPlanner::getManhattanDistance(int loc1, int loc2) {
    int loc1_x = loc1/env->cols;
    int loc1_y = loc1%env->cols;
    int loc2_x = loc2/env->cols;
    int loc2_y = loc2%env->cols;
    return abs(loc1_x - loc2_x) + abs(loc1_y - loc2_y);
}

bool MAPFPlanner::validateMove(int loc, int loc2)
{
    int loc_x = loc/env->cols;
    int loc_y = loc%env->cols;

    if (loc_x >= env->rows || loc_y >= env->cols || env->map[loc] == 1)
        return false;

    int loc2_x = loc2/env->cols;
    int loc2_y = loc2%env->cols;
    if (abs(loc_x-loc2_x) + abs(loc_y-loc2_y) > 1)
        return false;
    return true;

}


list<pair<int,int>> MAPFPlanner::getNeighbors(int location,int direction) {
    list<pair<int,int>> neighbors;
    //forward
    int candidates[4] = { location + 1,location + env->cols, location - 1, location - env->cols};
    int forward = candidates[direction];
    int new_direction = direction;
    if (forward>=0 && forward < env->map.size() && validateMove(forward,location))
        neighbors.emplace_back(make_pair(forward,new_direction));
    //turn left
    new_direction = direction-1;
    if (new_direction == -1)
        new_direction = 3;
    neighbors.emplace_back(make_pair(location,new_direction));
    //turn right
    new_direction = direction+1;
    if (new_direction == 4)
        new_direction = 0;
    neighbors.emplace_back(make_pair(location,new_direction));
    neighbors.emplace_back(make_pair(location,direction)); //wait
    return neighbors;
}

#else 

#include <MAPFPlanner.h>
#include <random>
#include "util/HeuristicTable.h"
#include "util/Analyzer.h"
#include "util/MyLogger.h"
#include "boost/format.hpp"
#include "util/MyCommon.h"

struct AstarNode {
    int location;
    int direction;
    int f,g,h;
    AstarNode* parent;
    int t = 0;
    bool closed = false;
    AstarNode(int _location,int _direction, int _g, int _h, AstarNode* _parent):
        location(_location), direction(_direction),f(_g+_h),g(_g),h(_h),parent(_parent) {}
    AstarNode(int _location,int _direction, int _g, int _h, int _t, AstarNode* _parent):
        location(_location), direction(_direction),f(_g+_h),g(_g),h(_h),t(_t),parent(_parent) {}
};


struct cmp {
    bool operator()(AstarNode* a, AstarNode* b) {
        if(a->f == b->f) return a->g <= b->g;
        else return a->f > b->f;
    }
};

void MAPFPlanner::load_configs() {
    // load configs
	string config_path="configs/"+env->map_name.substr(0,env->map_name.find_last_of("."))+"_no_rot.json";
    char * _config_path=getenv("CONFIG_PATH");
    if (_config_path!=NULL) {
        config_path=std::string(_config_path);
        std::cout<<"load config from "<<config_path<<std::endl;
    }
    std::ifstream f(config_path);
    try
    {
        config = nlohmann::json::parse(f);

        char * env_weight_path=getenv("MAP_WEIGHT_PATH");
        if (env_weight_path!=NULL) {
            config["map_weights_path"]=env_weight_path;
            std::cout<<"load weight from "<<env_weight_path<<std::endl;
        }

        std::cerr<<config<<std::endl;
        config["lifelong_solver_name"]=read_conditional_value(config,"lifelong_solver_name",env->num_of_agents);
        config["map_weights_path"]=read_conditional_value(config,"map_weights_path",env->num_of_agents);

        if (config.contains("max_task_completed")) {
            config["max_task_completed"]=read_conditional_value(config,"max_task_completed",env->num_of_agents);
        }

        if (config.contains("max_execution_steps")) {
            config["max_execution_steps"]=read_conditional_value(config,"max_execution_steps",env->num_of_agents);
        }

        config["LaCAM2"]["order_strategy"]=read_conditional_value(config["LaCAM2"],"order_strategy",env->num_of_agents);
        config["LNS"]["LaCAM2"]["order_strategy"]=read_conditional_value(config["LNS"]["LaCAM2"],"order_strategy",env->num_of_agents);
        config["disable_corner_target_agents"]=read_conditional_value(config,"disable_corner_target_agents",env->num_of_agents);
        config["max_agents_in_use"]=read_conditional_value(config,"max_agents_in_use",env->num_of_agents);

        config["LNS"]["fix_ng_bug"]=read_conditional_value(config["LNS"],"fix_ng_bug",env->num_of_agents);
        config["LNS"]["window_size_for_CT"]=read_conditional_value(config["LNS"],"window_size_for_CT",env->num_of_agents);
        config["LNS"]["window_size_for_CAT"]=read_conditional_value(config["LNS"],"window_size_for_CAT",env->num_of_agents);
        config["LNS"]["window_size_for_PATH"]=read_conditional_value(config["LNS"],"window_size_for_PATH",env->num_of_agents);
        config["LNS"]["LaCAM2"]["planning_window"]=read_conditional_value(config["LNS"]["LaCAM2"],"planning_window",env->num_of_agents);


        string s=config.dump();
        std::replace(s.begin(),s.end(),',','|');
        config["details"]=s;
    }
    catch (nlohmann::json::parse_error error)
    {
        std::cout << "Failed to load " << config_path << std::endl;
        std::cout << "Message: " << error.what() << std::endl;
        exit(1);
    }
}

std::string MAPFPlanner::load_map_weights(string weights_path) {
    // TODO(rivers): make weights float
    // we have at least 5 weights for a location: right,down,left,up,stay
    map_weights=std::make_shared<std::vector<float> >(env->rows*env->cols*5,1);
    std::string suffix = "all_one";

    if (weights_path!=""){
        std::ifstream f(weights_path);
        try
        {
            nlohmann::json _weights = nlohmann::json::parse(f);
            if (_weights.size()!=map_weights->size()) {
                std::cerr<<"map weights size mismatch"<<std::endl;
                exit(-1);
            }

            for (int i=0;i<map_weights->size();++i){
                (*map_weights)[i]=_weights[i].get<float>();
            }
            
        }
        catch (nlohmann::json::parse_error error)
        {
            std::cerr << "Failed to load " << weights_path << std::endl;
            std::cerr << "Message: " << error.what() << std::endl;
            exit(1);
        }

        boost::filesystem::path _weights_path(weights_path);
        suffix=_weights_path.stem().string();
    }
    return suffix;
}

void MAPFPlanner::initialize(int preprocess_time_limit) {
    cout << "planner initialization begins" << endl;
    load_configs();

    ONLYDEV(
        analyzer.timestamp();
        analyzer.init_from_config(config);
        analyzer.set_dump_path(config["analysis_output"].get<string>());
        g_logger.init("logs/run");
    )

    max_execution_steps = read_param_json<int>(config,"max_execution_steps",1000000);

    std::cout<<"max execution steps: "<<max_execution_steps<<std::endl;

    std::string weights_path=read_param_json<std::string>(config,"map_weights_path");
    std::string suffix=load_map_weights(weights_path);

    lifelong_solver_name=config["lifelong_solver_name"];

    // TODO(hj): memory management is a disaster here...
    if (lifelong_solver_name=="LaCAM2") {
        if (
        read_param_json<bool>(config["LaCAM2"],"use_orient_in_heuristic")
        || read_param_json<bool>(config["LaCAM2"],"consider_rotation")
        || read_param_json<bool>(config["LaCAM2"],"use_slow_executor")
        ) {
            std::cerr<<"no allowed to use orient when compiled with NO_ROT"<<std::endl;
            exit(-1);
        }

        if (!read_param_json<bool>(config["LaCAM2"],"use_external_executor")) {
            std::cerr<<"must use external executor when compiled with NO_ROT"<<std::endl;
            exit(-1);
        }

        auto heuristics =std::make_shared<HeuristicTable>(env,map_weights,false);
        heuristics->preprocess(suffix);
        int max_agents_in_use=read_param_json<int>(config,"max_agents_in_use",-1);
        if (max_agents_in_use==-1) {
            max_agents_in_use=env->num_of_agents;
        }
        bool disable_corner_target_agents=read_param_json<bool>(config,"disable_corner_target_agents",false);
        int max_task_completed=read_param_json<int>(config,"max_task_completed",1000000);
        lacam2_solver = std::make_shared<LaCAM2::LaCAM2Solver>(heuristics,env,map_weights,max_agents_in_use,disable_corner_target_agents,max_task_completed,config["LaCAM2"]);
        lacam2_solver->initialize(*env);
        cout<<"LaCAMSolver2 initialized"<<endl;
    } else if (lifelong_solver_name=="LNS") {
        if (
        read_param_json<bool>(config["LNS"]["LaCAM2"],"use_orient_in_heuristic")
        || read_param_json<bool>(config["LNS"]["LaCAM2"],"consider_rotation")
        || read_param_json<bool>(config["LNS"]["LaCAM2"],"use_slow_executor")
        ) {
            std::cerr<<"no allowed to use orient when compiled with NO_ROT"<<std::endl;
            exit(-1);
        }

        if (!read_param_json<bool>(config["LaCAM2"],"use_external_executor")) {
            std::cerr<<"must use external executor when compiled with NO_ROT"<<std::endl;
            exit(-1);
        }

        auto heuristics =std::make_shared<HeuristicTable>(env,map_weights,false);
        heuristics->preprocess(suffix);
        //heuristics->preprocess();

        int max_agents_in_use=read_param_json<int>(config,"max_agents_in_use",-1);
        if (max_agents_in_use==-1) {
            max_agents_in_use=env->num_of_agents;
        }

        bool disable_corner_target_agents=read_param_json<bool>(config,"disable_corner_target_agents",false);
        int max_task_completed=read_param_json<int>(config,"max_task_completed",1000000);
        auto lacam2_solver = std::make_shared<LaCAM2::LaCAM2Solver>(heuristics,env,map_weights,max_agents_in_use,disable_corner_target_agents,max_task_completed,config["LNS"]["LaCAM2"]);
        lacam2_solver->initialize(*env);
        lns_solver = std::make_shared<LNS::LNSSolver>(heuristics,env,map_weights,config["LNS"],lacam2_solver,max_task_completed);
        lns_solver->initialize(*env);
        cout<<"LNSSolver initialized"<<endl;
    }
    else {
        cerr<<"unknown lifelong solver name"<<lifelong_solver_name<<endl;
        exit(-1);
    }

    cout << "planner initialization ends" << endl;
}


// plan using simple A* that ignores the time dimension
void MAPFPlanner::plan(int time_limit,vector<Action> & actions) 
{
    // NOTE we need to return within time_limit, but we can exploit this time duration as much as possible

    // check if we need to restart a plan task (thread)
    // if so, we need to stop the current one and then restart
    // we also need to clean the current action plan if restart

    // TODO if time_limit approachs, just return a valid move, e.g. all actions are wait.

    ONLYDEV(
        g_timer.record_p("_step_s");
    )

    if (env->curr_timestep>=max_execution_steps-1) {
        for (int i=0;i<env->num_of_agents;++i) {
            actions.push_back(Action::W);
        }
        return;
    }

    if (lifelong_solver_name=="LaCAM2") {
        ONLYDEV(cout<<"using LaCAM2"<<endl;)
        ONLYDEV(g_timer.record_p("mapf_lacam2_plan_s");)
        lacam2_solver->plan(*env);
        ONLYDEV(g_timer.record_d("mapf_lacam2_plan_s","mapf_lacam2_plan");)
        ONLYDEV(g_timer.record_p("mapf_lacam2_get_step_s");)
        lacam2_solver->get_step_actions(*env,actions);
        ONLYDEV(g_timer.record_d("mapf_lacam2_get_step_s","mapf_lacam2_get_step");)
    } else if (lifelong_solver_name=="LNS") {
        ONLYDEV(cout<<"using LNS"<<endl;)
        lns_solver->observe(*env);
        lns_solver->plan(*env); 
        lns_solver->get_step_actions(*env,actions);
    } else if (lifelong_solver_name=="DUMMY") {
        cout<<"using DUMMY"<<endl;
    } 
    else {
        cerr<<"unknown lifelong solver name"<<lifelong_solver_name<<endl;
        exit(-1);
    }

    ONLYDEV(
        double step_time=g_timer.record_d("_step_s","_step_e","_step");
        if (step_time>max_step_time) {
            max_step_time=step_time;
        }

        std::cerr<<"max_step_time: "<<max_step_time<<endl;
        g_timer.remove_d("_step");
    )

    // for (int i = 0; i < env->num_of_agents; i++) 
    // {
    //     list<pair<int,int>> path;
    //     if (env->goal_locations[i].empty()) 
    //     {
    //         path.push_back({env->curr_states[i].location, env->curr_states[i].orientation});
    //     } 
    //     else 
    //     {
    //         path = single_agent_plan(env->curr_states[i].location,
    //                                 env->curr_states[i].orientation,
    //                                 env->goal_locations[i].front().first);
    //     }
    //     if (path.front().first != env->curr_states[i].location)
    //     {
    //         actions[i] = Action::FW; //forward action
    //     } 
    //     else if (path.front().second!= env->curr_states[i].orientation)
    //     {
    //         int incr = path.front().second - env->curr_states[i].orientation;
    //         if (incr == 1 || incr == -3)
    //         {
    //             actions[i] = Action::CR; //C--counter clockwise rotate
    //         } 
    //         else if (incr == -1 || incr == 3)
    //         {
    //             actions[i] = Action::CCR; //CCR--clockwise rotate
    //         } 
    //     }

    // }


  return;
}

list<pair<int,int>> MAPFPlanner::single_agent_plan(int start,int start_direct,int end) {
    list<pair<int,int>> path;
    priority_queue<AstarNode*,vector<AstarNode*>,cmp> open_list;
    unordered_map<int,AstarNode*> all_nodes;
    unordered_set<int> close_list;
    AstarNode* s = new AstarNode(start, start_direct, 0, getManhattanDistance(start,end), nullptr);
    open_list.push(s);
    all_nodes[start*4 + start_direct] = s;

    while (!open_list.empty()) {
        AstarNode* curr = open_list.top();
        open_list.pop();
        close_list.emplace(curr->location*4 + curr->direction);
        if (curr->location == end) {
            while(curr->parent!=NULL) {
                path.emplace_front(make_pair(curr->location, curr->direction));
                curr = curr->parent;
            }
            break;
        }
        list<pair<int,int>> neighbors = getNeighbors(curr->location, curr->direction);
        for (const pair<int,int>& neighbor: neighbors) {
            if (close_list.find(neighbor.first*4 + neighbor.second) != close_list.end())
                continue;
            if (all_nodes.find(neighbor.first*4 + neighbor.second) != all_nodes.end()) {
                AstarNode* old = all_nodes[neighbor.first*4 + neighbor.second];
                if (curr->g + 1 < old->g) {
                    old->g = curr->g+1;
                    old->f = old->h+old->g;
                    old->parent = curr;
                }
            } else {
                AstarNode* next_node = new AstarNode(neighbor.first, neighbor.second,
                    curr->g+1,getManhattanDistance(neighbor.first,end), curr);
                open_list.push(next_node);
                all_nodes[neighbor.first*4+neighbor.second] = next_node;
            }
        }
    }
    for (auto n: all_nodes)
    {
        delete n.second;
    }
    all_nodes.clear();
    return path;
}


int MAPFPlanner::getManhattanDistance(int loc1, int loc2) {
    int loc1_x = loc1/env->cols;
    int loc1_y = loc1%env->cols;
    int loc2_x = loc2/env->cols;
    int loc2_y = loc2%env->cols;
    return abs(loc1_x - loc2_x) + abs(loc1_y - loc2_y);
}

bool MAPFPlanner::validateMove(int loc, int loc2)
{
    int loc_x = loc/env->cols;
    int loc_y = loc%env->cols;

    if (loc_x >= env->rows || loc_y >= env->cols || env->map[loc] == 1)
        return false;

    int loc2_x = loc2/env->cols;
    int loc2_y = loc2%env->cols;
    if (abs(loc_x-loc2_x) + abs(loc_y-loc2_y) > 1)
        return false;
    return true;

}


list<pair<int,int>> MAPFPlanner::getNeighbors(int location,int direction) {
    list<pair<int,int>> neighbors;
    //forward
    int candidates[4] = { location + 1,location + env->cols, location - 1, location - env->cols};
    int forward = candidates[direction];
    int new_direction = direction;
    if (forward>=0 && forward < env->map.size() && validateMove(forward,location))
        neighbors.emplace_back(make_pair(forward,new_direction));
    //turn left
    new_direction = direction-1;
    if (new_direction == -1)
        new_direction = 3;
    neighbors.emplace_back(make_pair(location,new_direction));
    //turn right
    new_direction = direction+1;
    if (new_direction == 4)
        new_direction = 0;
    neighbors.emplace_back(make_pair(location,new_direction));
    neighbors.emplace_back(make_pair(location,direction)); //wait
    return neighbors;
}

#endif