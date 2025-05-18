
#ifndef flow_hpp
#define flow_hpp
#include "Types.h"
#include "search.hpp"
#include "TrajLNS.h"
#include "heuristics.hpp"

#include <random>
#include <unordered_set>

namespace TrafficMAPF{


int get_op_flow(TrajLNS& lns, int prev_loc, int loc, int d);

int get_vertex_flow(TrajLNS& lns, int loc);

int get_all_op_flow(TrajLNS& lns);

//remove flow for each location's outgoing edge according to the traj
void remove_traj(TrajLNS& lns, int agent);

void add_traj(TrajLNS& lns, int agent);

void init_traj(TrajLNS& lns,std::vector<int>& traffic,int amount,bool remove_and_plan=false);

void init_traj_st(TrajLNS& lns,std::vector<int>& traffic);

void update_dist_2_path(TrajLNS& lns, int i, std::vector<int>& traffic);

//compute distance table for each traj
void init_dist_table(TrajLNS& lns,std::vector<int>& traffic, int amount);

//update traj and distance table for agent i
void update_traj(TrajLNS& lns, int i, std::vector<int>& traffic);




bool terminate(TrajLNS& lns,int max_ite, double time_limit, int ite, TimePoint start_time, int stop);

void random_select(TrajLNS& lns, std::vector<int> & agents);

void select_most_expensive(TrajLNS& lns, std::vector<int> & agents);


int select_method(TrajLNS& lns);


//neighborhood search
void destory_improve(TrajLNS& lns, std::vector<int>& traffic,std::unordered_set<int>& updated, int max_ite, double time_limit);

}
#endif