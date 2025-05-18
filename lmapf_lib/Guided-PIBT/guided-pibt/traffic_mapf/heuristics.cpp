
#include "heuristics.hpp"
#include <queue>

namespace TrafficMAPF{

void init_heuristic(HeuristicTable& ht, SharedEnvironment* env, int goal_location){
	// initialize my_heuristic, but have error on malloc: Region cookie corrupted for region
	ht.htable.clear();
	ht.htable.resize(env->map.size(),MAX_TIMESTEP);
	ht.open.clear();
	// generate a open that can save nodes (and a open_handle)
	HNode root(goal_location,0, 0);
	ht.htable[goal_location] = 0;
	ht.open.push_back(root);  // add root to open
}


int get_heuristic(HeuristicTable& ht, SharedEnvironment* env, std::vector<int>& traffic, std::vector<Int4>& flow, int source){
		if (ht.htable[source] < MAX_TIMESTEP) return ht.htable[source];

		std::vector<int> neighbors;
		int cost, diff, reverse_d, op_flow, vertex_flow;
		while (!ht.open.empty())
		{
			HNode curr = ht.open.front();
			ht.open.pop_front();

			
			getNeighborLocs(env,neighbors,curr.location);

			
			for (int next : neighbors)
			{
				cost = curr.value + 1;
				diff = curr.location - next;
				
				assert(next >= 0 && next < env->map.size());
				//set current cost for reversed direction

				if (cost >= ht.htable[next] )
					continue;

				ht.open.emplace_back(next,0, cost);
				ht.htable[next] = cost;
				
			}

			if (source == curr.location)
				return curr.value;
		}


		return MAX_TIMESTEP;
}



s_node* get_flow_heuristic(FlowHeuristic& ht, SharedEnvironment* env, std::vector<int>& traffic, std::vector<Int4>& flow, int source){
  		if (ht.mem.has_node(source) && ht.mem.is_closed(source))
			return ht.mem.get_node(source);
		re_of re;
		std::vector<int> neighbors;
		int cost, diff, reverse_d, op_flow, vertex_flow,h, depth, tie_breaker;
		while (ht.open.size()>0)
		{
			s_node* curr = ht.open.pop();
			curr->close();


			
			getNeighborLocs(env,neighbors,curr->id);

			
			for (int next : neighbors)
			{
				cost = curr->g+1;
				diff = curr->id - next;
				

				reverse_d = get_d(diff,env); //the direction of moving from next to current
				op_flow = 0;

				vertex_flow = 0;
				for (int i = 0; i < 4; i++)
				{
					vertex_flow += flow[curr->id].d[i];
				}
				
				cost += (vertex_flow-1)/2;
				

				h=get_heuristic(*(ht.h), env, traffic, flow,next);
				depth = curr->depth +1;

				s_node temp_node(next,cost,h,op_flow, depth);

				if (!ht.mem.has_node(next)){
					s_node* next_node = ht.mem.generate_node(next,cost,h,op_flow, depth);
					next_node->parent = curr;
					ht.open.push(next_node);
				}
				else{ 
					s_node* existing = ht.mem.get_node(next);

					if (!existing->is_closed()){
						if (re(temp_node,*existing) ){
							existing->g = cost;
							existing->parent = curr;
							existing->depth = depth;
							existing->op_flow = op_flow;
							ht.open.decrease_key(existing);
						}
					}
					else if (re(temp_node,*existing) ){ //cmp has a random tie breaker, cause problem when equeal, thus a not equal checking

						// raise error as optimal search with admissible heuristic
						// has no re expansion
						std::cout<<next<<","<<ht.origin<<","<<source<<","<< existing->g <<","<< existing->h<<","<< cost <<","<< h<<std::endl;
						std::cout << "error in getFlowH: re-expansion" << std::endl;
						assert(false);
						exit(1);
					}
				}
	


			}
			if (source == curr->id)
				return curr;
			

		}


		return nullptr;
}

void init_flow_heuristic(TrajLNS& lns, std::vector<int>& traffic, int i){
	int start_loc = lns.env->curr_states[i].location;
	int goal_location = lns.tasks[i];
	FlowHeuristic& ht = lns.flow_heuristics[i];
	
	ht.h = &lns.heuristics[start_loc];

	if (ht.h->empty()){
		init_heuristic(*ht.h, lns.env, start_loc);
	}
	
	// initialize my_heuristic, but have error on malloc: Region cookie corrupted for region
	ht.reset();
	// ht.op_flows.resize(env->map.size(),MAX_TIMESTEP);
	if (!ht.mem.is_ready()){
		ht.mem.init(lns.env->map.size());
	}
	ht.target = goal_location;
	ht.origin = start_loc;
	s_node* root = ht.mem.generate_node(goal_location,0, get_heuristic(*(ht.h), lns.env,traffic,lns.flow,start_loc),0,0);
	
	ht.open.push(root);  // add root to open

	get_flow_heuristic(ht, lns.env, traffic, lns.flow, start_loc);
}

void init_flow_heuristic(TrajLNS& lns, std::vector<int>& traffic, FlowHeuristic& ht, int i, int goal_loc){
	int start_loc = lns.env->curr_states[i].location;
	int goal_location = goal_loc;	
	ht.h = &lns.heuristics[start_loc];

	if (ht.h->empty()){
		init_heuristic(*ht.h, lns.env, start_loc);
	}
	
	// initialize my_heuristic, but have error on malloc: Region cookie corrupted for region
	ht.reset();
	// ht.op_flows.resize(env->map.size(),MAX_TIMESTEP);
	if (!ht.mem.is_ready()){
		ht.mem.init(lns.env->map.size());
	}
	ht.target = goal_location;
	ht.origin = start_loc;
	s_node* root = ht.mem.generate_node(goal_location,0, get_heuristic(*(ht.h), lns.env,traffic,lns.flow,start_loc),0,0);
	
	ht.open.push(root);  // add root to open

	get_flow_heuristic(ht, lns.env, traffic, lns.flow, start_loc);
}





void compute_dist_2_path(std::vector<int>& my_heuristic, SharedEnvironment* env, Traj& path)
{
	my_heuristic.clear();
	my_heuristic.resize(env->map.size(), MAX_TIMESTEP);

    std::queue<std::pair<int,int>> open;

    for(auto& p : path){
        if (0 < my_heuristic.at(p)){
            my_heuristic.at(p) = 0;
            open.emplace(p,0);
        }
    }
    int neighbors[4];
	while (!open.empty())
	{
		std::pair<int,int> curr = open.front();
		open.pop();
		
        getNeighborLocs(env,neighbors,curr.first);
		for (int next_location : neighbors)
		{
			if (next_location == -1)
				continue;
			if (my_heuristic.at(next_location) > curr.second + 1)
			{
				my_heuristic.at(next_location) = curr.second + 1;
				open.emplace(next_location, curr.second + 1);
			}
		}
	}
}

void init_dist_2_path(Dist2Path& dp, SharedEnvironment* env, Traj& path, std::vector<int>& traffic){
	if (dp.dist2path.empty())
		dp.dist2path.resize(env->map.size(), d2p(0,-1,MAX_TIMESTEP,MAX_TIMESTEP));
	
	// why don't clear dist2path and reset? after traj changes.
	dp.open.clear();
	dp.label++;

    int togo = 0;
    for(int i = path.size()-1; i>=0; i--){
        auto p = path[i];
		assert(dp.dist2path[p].label != dp.label || dp.dist2path[p].cost == MAX_TIMESTEP);
		dp.open.emplace_back(dp.label,p,0,togo);
		dp.dist2path[p] = {dp.label,p,0,togo};
		togo++;
    }

}

int get_dist_2_path(Dist2Path& dp, SharedEnvironment* env, std::vector<int>& traffic, int source)
{
	if (dp.dist2path[source].label == dp.label && dp.dist2path[source].cost < MAX_TIMESTEP){
		// std::cout<<dp.dist2path[source].first<<" "<<dp.dist2path[source].second<<std::endl;

		return dp.dist2path[source].cost + dp.dist2path[source].togo;
	}

	
	std::vector<int> neighbors;
	int cost;

	while (!dp.open.empty())
	{
		d2p curr = dp.open.front();
		dp.open.pop_front();



		getNeighborLocs(env,neighbors,curr.id);

		for (int next_location : neighbors)
		{

			cost = curr.cost + 1;

			// BUG (rivers): interesting. this implementation is different from what I expected. 
			// I thought we should keep the smallest overall distance to the goal, namely cost+to_go.
			// BUT I think I'm wrong, because we want to go back to the reference path as soon as possible in general.
			// but sitll we should compare when the cost is the same, if to go is smaller, then we should update?
			// this may make it not BFS? or may this situation won't happen because of the init BFS order.
			if (dp.dist2path[next_location].label == dp.label && cost >= dp.dist2path[next_location].cost )
				continue;
			dp.open.emplace_back(dp.label,next_location,cost,curr.togo);
			dp.dist2path[next_location] = {dp.label,next_location,cost,curr.togo};
			
		}
		if (source == curr.id){
			// std::cout<<curr.second.first<<" "<<curr.second.second<<std::endl;
			return curr.cost + curr.togo;
		}
	}

	return MAX_TIMESTEP;
}



}
