#include <MAPFPlanner.h>
#include <random>
#include "pibt.hpp"
#include "flow.hpp"
#include "heuristics.hpp"
#include "my_pibt.h"
#include "LNS/Parallel/GlobalManager.h"
#include "LNS/Instance.h"

using namespace TrafficMAPF;

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



void MAPFPlanner::initialize(int preprocess_time_limit) {
    assert(env->num_of_agents != 0);
    p.resize(env->num_of_agents);
    decision.resize(env->map.size(), -1);
    prev_states.resize(env->num_of_agents);
    next_states.resize(env->num_of_agents);
    decided.resize(env->num_of_agents,DCR({-1,DONE::DONE}));
    occupied.resize(env->map.size(),false);
    checked.resize(env->num_of_agents,false);
    ids.resize(env->num_of_agents);
    task_change.resize(env->num_of_agents,false);
    for (int i = 0; i < ids.size();i++){
        ids[i] = i;
    }

    trajLNS = TrajLNS(env, seed);
    trajLNS.init_mem();

    env->init_neighbor();

    std::shuffle(ids.begin(), ids.end(), std::mt19937(0));
    for (int i = 0; i < ids.size();i++){
        p[ids[i]] = ((double)(ids.size() - i))/((double)(ids.size()+1));
    }
    p_copy = p;


    traffic.resize(env->map.size(),-1);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

    map_weights=std::make_shared<std::vector<float> >(env->rows*env->cols*5, 1.0);
}


// return next states for all agents
void MAPFPlanner::plan(int time_limit,vector<Action> & actions, bool get_actions) 
{
    // cout<<"---timestep,"<< env->curr_timestep<<endl;
    prev_decision.clear();
    prev_decision.resize(env->map.size(), -1);
    occupied.clear();
    occupied.resize(env->map.size(),false);

    int count = 0;
    
    for(int i=0; i<env->num_of_agents; i++)
    {

        for(int j=0; j<env->goal_locations[i].size(); j++)
        {
            int goal_loc = env->goal_locations[i][j].first;
            if (trajLNS.heuristics.at(goal_loc).empty()){
                init_heuristic(trajLNS.heuristics[goal_loc],env,goal_loc);
                count++;
                #ifndef GUIDANCE
                trajLNS.traj_inited++;
                #endif

            }
        }

        if ( (trajLNS.traj_inited < env->num_of_agents && count < RELAX) || (trajLNS.traj_inited >= env->num_of_agents)){
            for(int j=0; j<env->goal_locations[i].size(); j++)
            {
                int goal_loc = env->goal_locations[i][j].first;
                if (OBJECTIVE >= OBJ::SUI_TC){
                    int dist = get_heuristic(trajLNS.heuristics[goal_loc],env,traffic,trajLNS.flow,env->curr_states[i].location);
                    if ( dist > env->max_h)
                        env->max_h = dist;
                }
            }
        }

        assert(env->goal_locations[i].size()>0);
        task_change[i] =  env->goal_locations[i].front().first != trajLNS.tasks[i];
        trajLNS.tasks[i] = env->goal_locations[i].front().first;

        assert(env->curr_states[i].location >=0);
        prev_states[i] = env->curr_states[i];
        next_states[i] = State();
        prev_decision[env->curr_states[i].location] = i; 
        if (decided[i].loc == -1){
            decided[i].loc = env->curr_states[i].location;
            assert(decided[i].state == DONE::DONE);
        }
        if (prev_states[i].location == decided[i].loc){
            decided[i].state = DONE::DONE;
        }
        if (decided[i].state == DONE::NOT_DONE){
            occupied.at(decided[i].loc) = true;
            occupied.at(prev_states[i].location) = true;
        }

        if(task_change[i])
            p[i] = p_copy[i];
        else
            p[i] = p[i]+1;
        
    }

#ifdef GUIDANCE
    bool init_done = trajLNS.traj_inited == env->num_of_agents;
    TimePoint start_time = std::chrono::steady_clock::now();

    // cout<<"Check task updates"<<endl;
    // #ifndef FLOW_GUIDANCE
        for (int i = 0; i < env->num_of_agents;i++){
            if (task_change[i] && !trajLNS.trajs[i].empty()){

                remove_traj(trajLNS, i);
                update_traj(trajLNS, i, traffic);
            }
        }
    // #endif
    // cout << "---t-update," << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() << endl;


    if (trajLNS.traj_inited < env->num_of_agents){
        // cout << "init traj"<<endl;
#ifdef INIT_PP
        init_traj(trajLNS, traffic, RELAX);
#else
        init_traj_st(trajLNS, traffic);
#endif
        // exit(1);
    }
    // cout << "---t-init," << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_time).count() << endl;


    std::unordered_set<int> updated;

    #ifdef GUIDANCE_LNS
    if (init_done){
        // cout << "---op-flow,"<<trajLNS.op_flow << endl;
        // cout << "---vetex-flow,"<<trajLNS.vertex_flow << endl;
        // destory and improve traj
        destory_improve(trajLNS, traffic, updated, GUIDANCE_LNS, time_limit * 0.8);

        // cout << "---updated size:" << updated.size() << endl;
        // cout << "---op-flow,"<<trajLNS.op_flow << endl;
        // cout << "---vertex-flow,"<<trajLNS.vertex_flow << endl;

        
        #ifndef FLOW_GUIDANCE
            //use dist to path/trajectory
            // cout<<"update dist to path"<<endl;
            for (int i : updated){
                update_dist_2_path(trajLNS, i,traffic);
            }
            
        #endif
    }
    #endif

    // cout << "---t-lns," << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_time).count() << endl;


    #ifndef FLOW_GUIDANCE
        if (trajLNS.dist2path_inited < env->num_of_agents){
            // cout<<"Init dist to path"<<endl;
            init_dist_table(trajLNS,traffic, RELAX);
        }

        for (int i = 0; i < trajLNS.dist2path_inited;i++){
            if (task_change[i]&& updated.find(i) == updated.end()){
                update_dist_2_path(trajLNS, i,traffic);
            }
        }
        
    #else

        // cout<<"init flow guidance heuristic for all agents."<<endl;
        // if (env->curr_timestep%FLOW_GUIDANCE ==0)
        //     for (int i = 0; i < env->num_of_agents;i++){
        //         if (!trajLNS.trajs[i].empty() && trajLNS.trajs[i].back() == trajLNS.tasks[i])
        //             continue;
        //         remove_traj(trajLNS, i);
        //         update_traj(trajLNS, i, traffic);
        //     }
        // if (env->curr_timestep%FLOW_GUIDANCE ==0)
        //     fg_init_count = 0;
        count = 0;

        for (int i = 0; i < env->num_of_agents; i++){
            if (task_change[i]){
                init_flow_heuristic(trajLNS, traffic,i);
                count++;
            }
        }

        if (fg_init_count >= env->num_of_agents)
            fg_init_count = 0;

        for (int i= fg_init_count; i < env->num_of_agents; i++){
            if (count >= RELAX)
                break;
            if (!task_change[i]){
                init_flow_heuristic(trajLNS, traffic,i);
                count++;
            }
            fg_init_count++;
        }

    #endif

    // cout << "---t-done," << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_time).count() << endl;

#endif

    if (get_actions) {

        std::vector<bool> disabled;
        for (int i=0;i<ids.size();++i) {
            // int curr_loc=env->curr_states[i].location;
            int goal_loc = env->goal_locations[i][0].first;
            if (env->neighbors[goal_loc].size()==1) {
                disabled.push_back(true);
            } else {
                disabled.push_back(false);
            }
        }

        std::sort(ids.begin(), ids.end(), [&](int a, int b) {
                if (disabled[a]==disabled[b]) {
                    return p.at(a) > p.at(b);
                } else {
                    return disabled[a]<disabled[b];
                }
            }
        );


        int num_threads;
        char * num_threads_env = std::getenv("LNS_NUM_THREADS");
        if (num_threads_env!=nullptr) {
            num_threads=std::atoi(num_threads_env);
        } else {
            num_threads=omp_get_max_threads();
        }

        auto HT=std::make_shared<Dist2PathHeuristicTable>(trajLNS,traffic,map_weights,num_threads);
        LNS::Instance instance(*env);
        std::shared_ptr<std::vector<LNS::Parallel::AgentInfo> > agent_infos;

        agent_infos=std::make_shared<std::vector<LNS::Parallel::AgentInfo> > (env->num_of_agents);
        for (int i=0;i<env->num_of_agents;++i) {
            (*agent_infos)[i].id=i;
            (*agent_infos)[i].disabled=disabled[i];
        }

        bool use_lns=false;
        if (num_threads_env!=nullptr && num_threads!=0) {
            use_lns=true;
        }

        // prepare pibt solver
        auto pibt_solver=std::make_shared<PIBTSolver>();

        int planning_window;

        if (use_lns) {
            planning_window=15;
        } else {
            planning_window=1;
        }

        // call pibt multiple times
        std::vector<int> action_choices={0,1,1,0,0,-1,-1,0,0,0};
        std::vector<int> map_size={env->rows,env->cols};
        std::vector<float> priorities;
        std::vector<int> locations;

        std::vector<std::vector<int> > paths(env->num_of_agents);

        for (int agent_idx=0;agent_idx<env->num_of_agents;++agent_idx) {
            float priority=disabled[agent_idx]?-1:p[agent_idx];
            priorities.push_back(priority);
            int location=env->curr_states[agent_idx].location;
            locations.push_back(location/env->cols);
            locations.push_back(location%env->cols);
            paths[agent_idx].push_back(location);
        }

        for (int step=0;step<planning_window;++step) {
            std::vector<int> _actions=pibt_solver->solve(
                *HT,
                priorities,
                locations,
                action_choices,
                map_size,
                false
            );

            // take actions
            for (int agent_idx=0;agent_idx<env->num_of_agents;++agent_idx) {
                int y=locations[agent_idx*2];
                int x=locations[agent_idx*2+1];
                int _action=_actions[agent_idx];
                int dy=action_choices[_action*2];
                int dx=action_choices[_action*2+1];
                int ny=y+dy;
                int nx=x+dx;
                // update locations
                locations[agent_idx*2]=ny;
                locations[agent_idx*2+1]=nx;
                int next_location=ny*env->cols+nx;
                paths[agent_idx].push_back(next_location);
            }   
        }


        if (use_lns) {
            // prepare lns
            bool async=true;
            int neighborSize=8;
            int execution_window=1;
            auto lns=std::make_shared<LNS::Parallel::GlobalManager>(
                async,
                instance, // instance
                HT,
                map_weights,
                agent_infos, // agent_infos only used for disabled information now. maybe merged into instance?
                neighborSize,
                LNS::Parallel::destroy_heuristic::RANDOMWALK, // useless
                true, // always adaptive
                0.01, // decay factor
                0.01, // reaction factor
                "LaCAM2",
                "PP",
                false, // no sipp
                planning_window,
                planning_window,
                planning_window,
                execution_window, // execution window
                true, // has disabled agents
                true, // fix ng bug 
                0, // screen
                num_threads
            );

            lns->reset();
            // TODO(rivers): call instace->set_starts_and_goals if instance is not built each step.
            // instance->set_starts_and_goals(starts,goals);

            // copy pibt paths to lns paths
            for (int agent_idx=0;agent_idx<env->num_of_agents;++agent_idx) {
                auto & lns_path=lns->agents[agent_idx].path;
                auto & pibt_path=paths[agent_idx];
                lns_path.clear();
                for (int j=0;j<pibt_path.size();++j) {
                    lns_path.nodes.emplace_back(pibt_path[j],-1);
                }
                lns_path.path_cost=lns->agents[agent_idx].getEstimatedPathLength(lns_path,env->goal_locations[agent_idx][0].first,HT);
            }

            // TODO(rivers): should subtract other parts
            double _time_limit=time_limit;
            TimeLimiter time_limiter(_time_limit);
            bool succ=lns->run(time_limiter);

            // TODO(rivers): what is the following code for?
            // we cannot do this because it would make result invalid
            // deal with a special case when the goal and the start are the same.
            if (execution_window==1) {
                for (int i=0;i<lns->agents.size();++i) {
                    if (lns->agents[i].path.size()<planning_window+1) {
                        // in this case, actually the goal is the same as the start
                        lns->agents[i].path.nodes.resize(planning_window+1,lns->agents[i].path.back());
                    }
                }
            }

            // let's copy data back
            for (int agent_idx=0;agent_idx<env->num_of_agents;++agent_idx) {
                auto & lns_path=lns->agents[agent_idx].path;
                auto & path=paths[agent_idx];
                path.clear();
                for (int j=0;j<lns_path.size();++j) {
                    path.push_back(lns_path[j].location);
                }
            }
        }

        // get actions 
        actions.clear();
        for (int agent_idx=0;agent_idx<env->num_of_agents;++agent_idx) {
            auto & path=paths[agent_idx];//lns->agents[agent_idx].path.nodes;
            int location = path[0];//.location;
            int next_location = path[1];//.location;
            int y=location/env->cols;
            int x=location%env->cols;
            int ny=next_location/env->cols;
            int nx=next_location%env->cols;
            int dy=ny-y;
            int dx=nx-x;
            int j=0;
            for (;j<action_choices.size()/2;++j) {
                if (dy==action_choices[j*2] && dx==action_choices[j*2+1]) {
                    actions.push_back(Action(j));
                    break;
                }
            }
            if (j>=action_choices.size()/2) {
                std::cerr<<"MAPFPlanner: a bug exists in action selection"<<std::endl;
                exit(-1);
            }
        }

        // actions.clear();
        // for (auto _action: _actions) {
        //     actions.push_back(Action(_action));
        // }

        // we should first insert WPPL here.

        // for (int i : ids){
            
        //     if (decided[i].state == DONE::NOT_DONE){
        //         continue;
        //     }
        //     if (next_states[i].location==-1){
        //         assert(prev_states[i].location >=0 && prev_states[i].location < env->map.size());
        //         causalPIBT(i,-1,prev_states,next_states,
        //             prev_decision,decision,
        //             occupied, traffic, trajLNS);
        //     }
        // }
        
        // actions.resize(env->num_of_agents);
        // for (int id : ids){

        //     if (next_states.at(id).location!= -1)
        //         decision.at(next_states.at(id).location) = -1;
            
        //     assert(
        //         (next_states.at(id).location >=0 && decided.at(id).state == DONE::DONE)||
        //         (next_states.at(id).location == -1 && decided.at(id).state == DONE::NOT_DONE)
        //     );

        //     if (next_states.at(id).location >=0){
        //         decided.at(id) = DCR({next_states.at(id).location,DONE::NOT_DONE});
        //     }

            

        //     actions.at(id) = getAction(prev_states.at(id),decided.at(id).loc, env);
        //     checked.at(id) = false;
        //     #ifndef NDEBUG
        //         std::cout<<id <<":"<<actions.at(id)<<";"<<std::endl;
        //     #endif

        // }

    #ifdef MAPFT
        for (int id=0;id < env->num_of_agents ; id++){
            if (!checked.at(id) && actions.at(id) == Action::FW){
                moveCheck(id,checked,decided,actions,prev_decision);
            }
        }
    #endif



        // #ifndef NDEBUG
        //     for (auto d : decision){
        //         assert(d == -1);
        //     }
        // #endif

        // prev_states = next_states;
    }

    return;
}

std::vector<int> MAPFPlanner::query_pibt_actions() {
    int planning_window=1;

    std::vector<bool> disabled;
    for (int i=0;i<ids.size();++i) {
        // int curr_loc=env->curr_states[i].location;
        int goal_loc = env->goal_locations[i][0].first;
        if (env->neighbors[goal_loc].size()==1) {
            disabled.push_back(true);
        } else {
            disabled.push_back(false);
        }
    }

    // std::sort(ids.begin(), ids.end(), [&](int a, int b) {
    //         if (disabled[a]==disabled[b]) {
    //             return p.at(a) > p.at(b);
    //         } else {
    //             return disabled[a]<disabled[b];
    //         }
    //     }
    // );

    auto HT=std::make_shared<Dist2PathHeuristicTable>(trajLNS,traffic,map_weights,1);

    std::vector<int> action_choices={0,1,1,0,0,-1,-1,0,0,0};
    
    // prepare pibt solver
    auto pibt_solver=std::make_shared<PIBTSolver>();
    std::vector<int> map_size={env->rows,env->cols};
    std::vector<float> priorities;
    std::vector<int> locations;

    std::vector<std::vector<int> > paths(env->num_of_agents);

    for (int agent_idx=0;agent_idx<env->num_of_agents;++agent_idx) {
        float priority=disabled[agent_idx]?-1:p[agent_idx];
        priorities.push_back(priority);
        int location=env->curr_states[agent_idx].location;
        locations.push_back(location/env->cols);
        locations.push_back(location%env->cols);
        paths[agent_idx].push_back(location);
    }

    for (int step=0;step<planning_window;++step) {
        std::vector<int> _actions=pibt_solver->solve(
            *HT,
            priorities,
            locations,
            action_choices,
            map_size,
            false
        );

        // take actions
        for (int agent_idx=0;agent_idx<env->num_of_agents;++agent_idx) {
            int y=locations[agent_idx*2];
            int x=locations[agent_idx*2+1];
            int _action=_actions[agent_idx];
            int dy=action_choices[_action*2];
            int dx=action_choices[_action*2+1];
            int ny=y+dy;
            int nx=x+dx;
            // update locations
            locations[agent_idx*2]=ny;
            locations[agent_idx*2+1]=nx;
            int next_location=ny*env->cols+nx;
            paths[agent_idx].push_back(next_location);
        }   
    }
    
    std::vector<int> actions;
    for (int agent_idx=0;agent_idx<env->num_of_agents;++agent_idx) {
        auto & path=paths[agent_idx];//lns->agents[agent_idx].path.nodes;
        int location = path[0];//.location;
        int next_location = path[1];//.location;
        int y=location/env->cols;
        int x=location%env->cols;
        int ny=next_location/env->cols;
        int nx=next_location%env->cols;
        int dy=ny-y;
        int dx=nx-x;
        int j=0;
        for (;j<action_choices.size()/2;++j) {
            if (dy==action_choices[j*2] && dx==action_choices[j*2+1]) {
                actions.push_back(j);
                break;
            }
        }
        if (j>=action_choices.size()/2) {
            std::cerr<<"MAPFPlanner: a bug exists in action selection"<<std::endl;
            exit(-1);
        }
    }
    
    return actions;
}

std::vector<int> MAPFPlanner::query_lns_actions(std::vector<int> & init_plan, int planning_window, int num_theads, int max_iterations) {
    bool use_lns=true;

    std::vector<bool> disabled;
    for (int i=0;i<ids.size();++i) {
        // int curr_loc=env->curr_states[i].location;
        int goal_loc = env->goal_locations[i][0].first;
        if (env->neighbors[goal_loc].size()==1) {
            disabled.push_back(true);
        } else {
            disabled.push_back(false);
        }
    }

    // std::sort(ids.begin(), ids.end(), [&](int a, int b) {
    //         if (disabled[a]==disabled[b]) {
    //             return p.at(a) > p.at(b);
    //         } else {
    //             return disabled[a]<disabled[b];
    //         }
    //     }
    // );

    auto HT=std::make_shared<Dist2PathHeuristicTable>(trajLNS,traffic,map_weights,num_theads);
    LNS::Instance instance(*env);
    std::shared_ptr<std::vector<LNS::Parallel::AgentInfo> > agent_infos;

    agent_infos=std::make_shared<std::vector<LNS::Parallel::AgentInfo> > (env->num_of_agents);
    for (int i=0;i<env->num_of_agents;++i) {
        (*agent_infos)[i].id=i;
        (*agent_infos)[i].disabled=disabled[i];
    }

    // prepare pibt solver
    // auto pibt_solver=std::make_shared<PIBTSolver>();

    // int planning_window;

    // if (use_lns) {
    //     planning_window=15;
    // } else {
    //     planning_window=1;
    // }

    // // call pibt multiple times
    const std::vector<int> action_choices={0,1,1,0,0,-1,-1,0,0,0};
    // std::vector<int> map_size={env->rows,env->cols};
    // std::vector<float> priorities;
    // std::vector<int> locations;

    // std::vector<std::vector<int> > paths(env->num_of_agents);

    // for (int agent_idx=0;agent_idx<env->num_of_agents;++agent_idx) {
    //     float priority=disabled[agent_idx]?-1:p[agent_idx];
    //     priorities.push_back(priority);
    //     int location=env->curr_states[agent_idx].location;
    //     locations.push_back(location/env->cols);
    //     locations.push_back(location%env->cols);
    //     paths[agent_idx].push_back(location);
    // }

    // for (int step=0;step<planning_window;++step) {
    //     std::vector<int> _actions=pibt_solver->solve(
    //         *HT,
    //         priorities,
    //         locations,
    //         action_choices,
    //         map_size,
    //         false
    //     );

    //     // take actions
    //     for (int agent_idx=0;agent_idx<env->num_of_agents;++agent_idx) {
    //         int y=locations[agent_idx*2];
    //         int x=locations[agent_idx*2+1];
    //         int _action=_actions[agent_idx];
    //         int dy=action_choices[_action*2];
    //         int dx=action_choices[_action*2+1];
    //         int ny=y+dy;
    //         int nx=x+dx;
    //         // update locations
    //         locations[agent_idx*2]=ny;
    //         locations[agent_idx*2+1]=nx;
    //         int next_location=ny*env->cols+nx;
    //         paths[agent_idx].push_back(next_location);
    //     }   
    // }

    int path_len=planning_window+1;
    std::vector<std::vector<int> > paths(env->num_of_agents);
    if (((int)init_plan.size())!=env->num_of_agents*path_len) {
        std::cerr<<"init plan has wrong length: "<<init_plan.size()<<" vs "<<env->num_of_agents*planning_window<<std::endl;
        exit(1);
    }

    for (int agent_idx=0;agent_idx<env->num_of_agents;++agent_idx) {
        for (int path_idx=0;path_idx<path_len;++path_idx) {
            int idx=agent_idx*path_len+path_idx;
            paths[agent_idx].push_back(init_plan[idx]);
        }
    }

    if (use_lns) {
        // prepare lns
        bool async=true;
        int neighborSize=8;
        int execution_window=1;
        auto lns=std::make_shared<LNS::Parallel::GlobalManager>(
            async,
            instance, // instance
            HT,
            map_weights,
            agent_infos, // agent_infos only used for disabled information now. maybe merged into instance?
            neighborSize,
            LNS::Parallel::destroy_heuristic::RANDOMWALK, // useless
            true, // always adaptive
            0.01, // decay factor
            0.01, // reaction factor
            "LaCAM2",
            "PP",
            false, // no sipp
            planning_window,
            planning_window,
            planning_window,
            execution_window, // execution window
            true, // has disabled agents
            true, // fix ng bug 
            0, // screen
            false, // verbose
            num_theads, // #theads, we cannot use multithreading now, because the heuristic table will change and not protected
            max_iterations// #iterations
        );

        lns->reset();
        // TODO(rivers): call instace->set_starts_and_goals if instance is not built each step.
        // instance->set_starts_and_goals(starts,goals);

        // copy pibt paths to lns paths
        for (int agent_idx=0;agent_idx<env->num_of_agents;++agent_idx) {
            auto & lns_path=lns->agents[agent_idx].path;
            auto & pibt_path=paths[agent_idx];
            lns_path.clear();
            for (int j=0;j<pibt_path.size();++j) {
                lns_path.nodes.emplace_back(pibt_path[j],-1);
            }
            lns_path.path_cost=lns->agents[agent_idx].getEstimatedPathLength(lns_path,env->goal_locations[agent_idx][0].first,HT);
        }

        double _time_limit=1.0;
        TimeLimiter time_limiter(_time_limit);
        bool succ=lns->run(time_limiter);

        // TODO(rivers): what is the following code for?
        // we cannot do this because it would make result invalid
        // deal with a special case when the goal and the start are the same.
        if (execution_window==1) {
            for (int i=0;i<lns->agents.size();++i) {
                if (lns->agents[i].path.size()<planning_window+1) {
                    // in this case, actually the goal is the same as the start
                    lns->agents[i].path.nodes.resize(planning_window+1,lns->agents[i].path.back());
                }
            }
        }

        // let's copy data back
        for (int agent_idx=0;agent_idx<env->num_of_agents;++agent_idx) {
            auto & lns_path=lns->agents[agent_idx].path;
            auto & path=paths[agent_idx];
            path.clear();
            for (int j=0;j<lns_path.size();++j) {
                path.push_back(lns_path[j].location);
            }
        }
    }

    // get actions 
    std::vector<int> actions;
    for (int agent_idx=0;agent_idx<env->num_of_agents;++agent_idx) {
        auto & path=paths[agent_idx];//lns->agents[agent_idx].path.nodes;
        int location = path[0];//.location;
        int next_location = path[1];//.location;
        int y=location/env->cols;
        int x=location%env->cols;
        int ny=next_location/env->cols;
        int nx=next_location%env->cols;
        int dy=ny-y;
        int dx=nx-x;
        int j=0;
        for (;j<action_choices.size()/2;++j) {
            if (dy==action_choices[j*2] && dx==action_choices[j*2+1]) {
                actions.push_back(j);
                break;
            }
        }
        if (j>=action_choices.size()/2) {
            std::cerr<<"MAPFPlanner: a bug exists in action selection"<<std::endl;
            exit(-1);
        }
    }

    // float total_delays=0;
    // for (int i=0;i<num_of_agents;++i) {
    //     total_delays+=(HT->get(start_locations[i], goal_locations[i])-global_manager->agents[i].getEstimatedPathLength(global_manager->agents[i].path,goal_locations[i],HT));
    // }

    return actions;
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
