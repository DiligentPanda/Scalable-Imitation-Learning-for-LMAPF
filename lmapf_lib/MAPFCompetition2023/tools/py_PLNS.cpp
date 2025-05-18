#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <memory>
#include "LNS/Instance.h"
#include "LNS/Parallel/GlobalManager.h"
#include "util/HeuristicTable.h"
#include "util/Timer.h"
#include <iostream>
#include "LaCAM2/instance.hpp"

class PLNSSolver {
public:
    int num_of_agents;
    int window_size_for_PATH;
    
    std::shared_ptr<LNS::Parallel::GlobalManager> global_manager;
    std::shared_ptr<HeuristicTable> HT;
    std::shared_ptr<std::vector<float> > map_weights;
    std::shared_ptr<SharedEnvironment> env;
    std::shared_ptr<LNS::Instance> instance;
    std::shared_ptr<std::vector<LaCAM2::AgentInfo> > agent_infos;

    PLNSSolver(
        int rows,
        int cols,
        std::vector<int> & map,
        std::string & map_weights_path,
        int num_of_agents,
        int window_size_for_PATH,
        int num_threads,
        int max_iterations,
        bool verbose=true
    ): 
        num_of_agents(num_of_agents), 
        window_size_for_PATH(window_size_for_PATH) {

        // auto grid=std::make_shared<Grid>(map_path);
        env=std::make_shared<SharedEnvironment>();
        env->map=map;
        env->rows=rows;
        env->cols=cols;  
        env->num_of_agents=num_of_agents;

        // build fake start states and goals for now
        env->curr_states.resize(num_of_agents);
        env->goal_locations.resize(num_of_agents,vector<pair<int,int>>(1));

        // TODO: make the following configurable or passed in
        map_weights=std::make_shared<std::vector<float> >(env->rows*env->cols*5, 1.0);
        if (map_weights_path!=""){
            std::ifstream f(map_weights_path);
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
                std::cerr << "Failed to load " << map_weights_path << std::endl;
                std::cerr << "Message: " << error.what() << std::endl;
                exit(1);
            }
        }

        bool consider_rotation=false;
        // TODO: use env.get is dangerous! legacy code!
        HT=std::make_shared<HeuristicTable>(env.get(), map_weights, consider_rotation);
        HT->compute_weighted_heuristics();
        
        instance=std::make_shared<LNS::Instance>(*env);

        agent_infos=std::make_shared<std::vector<LaCAM2::AgentInfo> >();

        // TODO: agent_infos also maintains goal_location, etc.
        // a bad design! refactor it! the only thing we want is the disabled vector.
        for (int i=0; i<num_of_agents; ++i) {
            agent_infos->emplace_back();
            auto & agent_info=agent_infos->back();
            agent_info.id=i;
            // agent_info.disabled=false;
        }


        // TODO: make configurable
        bool async=true;
        // agent_infos
        int neighbor_size=8;
        LNS::Parallel::destroy_heuristic destroy_strategy=LNS::Parallel::destroy_heuristic::RANDOMWALK;
        bool ALNS=true;
        double decay_factor=0.01;
        double reaction_factor=0.1;
        std::string init_algo_name="LaCAM2";
        std::string replan_algo_name="PP";
        bool sipp=false;
        int window_size_for_CT=window_size_for_PATH;
        int window_size_for_CAT=window_size_for_PATH;
        int execution_window=1;
        // TODO: support disabled agents
        bool has_disabled_agents=false;
        bool fix_ng_bug=true;
        int screen=0;

        // TODO: agent_infos

        global_manager=std::make_shared<LNS::Parallel::GlobalManager>(
            async,
            *instance, HT, map_weights, agent_infos,
            neighbor_size, destroy_strategy,
            ALNS, decay_factor, reaction_factor,
            init_algo_name, replan_algo_name, sipp,
            window_size_for_CT, window_size_for_CAT, window_size_for_PATH, execution_window,
            has_disabled_agents,
            fix_ng_bug,
            screen,
            verbose,
            num_threads,
            max_iterations
        );


    }

    // starts and goals should 1 integer encoding the location of the agent
    // row*#cols + col
    // init_paths should be length of num_of_agents*(window_size_for_PATH+1), the 1 is for the start locations
    std::pair<std::vector<int>,float> solve(
        std::vector<int> & start_locations,
        std::vector<int> & goal_locations,
        std::vector<int> & init_paths,
        std::vector<int> & action_choices,
        double time_limit
    ) { 
        TimeLimiter time_limiter(time_limit);

        std::vector<State> starts;
        std::vector<State> goals;

        if (start_locations.size()!=num_of_agents) {
            std::cout<<"start locations size wrong!"<<std::endl;
            exit(-1);
        }

        if (goal_locations.size()!=num_of_agents) {
            std::cout<<"goal locations size wrong!"<<std::endl;
            exit(-1);
        }

        if (init_paths.size()!=num_of_agents*(window_size_for_PATH+1)) {
            std::cout<<"init paths size wrong!"<<std::endl;
            exit(-1);
        }

        for (int i=0; i<num_of_agents; ++i) {
            starts.emplace_back(start_locations[i], -1, -1);
            goals.emplace_back(goal_locations[i], -1, -1);
        }

        global_manager->reset();
        instance->set_starts_and_goals(starts, goals);

        // copy init paths
        for (int i=0; i<num_of_agents; ++i) {
            global_manager->agents[i].path.clear();
            for (int j=0; j<=window_size_for_PATH; ++j) {
                int location=init_paths[i*(window_size_for_PATH+1)+j];
                if (j==0) {
                    if (location!=start_locations[i]) {
                        std::cerr<<"start locations mismatch: "<<location<<" "<<start_locations[i]<<std::endl;
                        exit(-1);
                    }
                }
                global_manager->agents[i].path.nodes.emplace_back(location,-1);
            }
            // compute init costs
            global_manager->agents[i].path.path_cost=global_manager->agents[i].getEstimatedPathLength(
                global_manager->agents[i].path,goal_locations[i],HT
            );
        }

        global_manager->run(time_limiter);

        float total_delays=0;
        for (int i=0;i<num_of_agents;++i) {
            total_delays+=(HT->get(start_locations[i], goal_locations[i])-global_manager->agents[i].getEstimatedPathLength(global_manager->agents[i].path,goal_locations[i],HT));
        }

        // // TODO: this might be conflict!
        // for (int i=0;i<num_of_agents;++i) {
        //     if (global_manager->agents[i].path.nodes.size()<window_size_for_PATH+1) {
        //         global_manager->agents[i].path.nodes.resize(window_size_for_PATH+1,global_manager->agents[i].path.nodes.back());
        //     }
        // }

        // get actions from plan
        std::vector<int> actions;
        for (int i=0; i<num_of_agents; ++i) {
            // we get the next location at index 1
            int next_location = global_manager->agents[i].path.nodes[1].location;
            int y=start_locations[i]/env->cols;
            int x=start_locations[i]%env->cols;
            int ny=next_location/env->cols;
            int nx=next_location%env->cols;
            int dy=ny-y;
            int dx=nx-x;
            int j=0;
            for (;j<action_choices.size();++j) {
                if (dy==action_choices[j*2] && dx==action_choices[j*2+1]) {
                    actions.push_back(j);
                    break;
                }
            }
            if (j>=action_choices.size()) {
                std::cerr<<"a bug exists"<<std::endl;
                exit(-1);
            }
        }

        return {actions,total_delays};
    }

    std::string playground(){
        return "hello, test!";
    }

};

PYBIND11_MODULE(py_PLNS, m) {
	// optional module docstring
    // m.doc() = ;
    pybind11::class_<PLNSSolver>(m, "PLNSSolver")
        .def(pybind11::init<int, int, std::vector<int> &, string &, int, int, int, int, bool>())
        .def("solve", &PLNSSolver::solve)
        .def("playground", &PLNSSolver::playground);
}