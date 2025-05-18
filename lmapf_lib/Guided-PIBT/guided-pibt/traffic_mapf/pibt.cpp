


#include "pibt.hpp"





namespace TrafficMAPF{

template<typename T, typename F>
void quickSort(T& agent_order, int low, int high, F compare)
{
    if (low >= high)
        return;
    auto pivot = agent_order[high];    // pivot
    int i = low;  // Index of smaller element
    for (int j = low; j <= high - 1; j++)
    {
        // If true, current element is smaller than or equal to pivot
        // If true, a have higher priority
        if (compare(agent_order[j], pivot))
        {
            std::swap(agent_order[i], agent_order[j]);
            i++;    // increment index of smaller element
        }
    }
    std::swap(agent_order[i], agent_order[high]);

    quickSort(agent_order, low, i - 1, compare);  // Before i
    quickSort(agent_order, i + 1, high, compare); // After i
}


bool causalPIBT(int curr_id, int higher_id,std::vector<State>& prev_states,
	 std::vector<State>& next_states,
      std::vector<int>& prev_decision, std::vector<int>& decision, 
	  std::vector<bool>& occupied, std::vector<int>& traffic,TrajLNS& lns
	  ){
	// The PIBT works like a causal PIBT when using MAPF-T model. But a normal PIBT when using MAPF model.
	
	//assert next states of curr_id is not decided
	assert(next_states[curr_id].location == -1);
    int prev_loc = prev_states[curr_id].location;
	int prev_orientation = prev_states[curr_id].orientation;
	int next[4] = { prev_loc + 1,prev_loc + lns.env->cols, prev_loc - 1, prev_loc - lns.env->cols};
#ifdef MAPFT
	int orien_next = next[prev_orientation];
#endif

	assert(prev_loc >= 0 && prev_loc < lns.env->map.size());
	// assert(prev_orientation >= 0 && prev_orientation < 4);

	int target = lns.tasks.at(curr_id);

	// for each neighbor of (prev_loc,prev_direction), and a wait copy of current location, generate a successor
	std::vector<int> neighbors;
	std::vector<State> successors;
	getNeighborLocs(lns.env,neighbors,prev_loc);
	for (auto& neighbor: neighbors){

		//check if prev_loc -> neighbor violoate the traffic rule on neighbor
		//violate if  traffic[neighbor] leads to prev_loc

		if (traffic[neighbor] != -1 ){
			int candidates[4] = { neighbor + 1,neighbor + lns.env->cols, neighbor - 1, neighbor - lns.env->cols};
			if (prev_loc  == candidates[traffic[neighbor]])
				continue;
		}
		assert(validateMove(prev_loc, neighbor, lns.env));

		//get ,min(heuristics[tasks[curr_id]][neighbor].d)
		int min_heuristic;
#ifdef FLOW_GUIDANCE
		if (!lns.flow_heuristics.empty() && !lns.flow_heuristics[curr_id].empty()){
			s_node* s= get_flow_heuristic(lns.flow_heuristics[curr_id], lns.env, traffic, lns.flow, neighbor);
			min_heuristic = s==nullptr? MAX_TIMESTEP : s->get_g();
		}
#else

		if (!lns.traj_dists.empty() && !lns.traj_dists[curr_id].empty())
			min_heuristic = get_dist_2_path(lns.traj_dists[curr_id], lns.env, traffic, neighbor);
	
#endif
		else if (!lns.heuristics[lns.tasks.at(curr_id)].empty())
			min_heuristic = get_heuristic(lns.heuristics[lns.tasks.at(curr_id)], lns.env, traffic, lns.flow, neighbor);
			// min_heuristic = heuristics.at(tasks.at(curr_id)).at(neighbor);
		// else if (!dh.empty())
		// 	min_heuristic = dh.get_heuristic(neighbor, tasks.at(curr_id));
		else
			min_heuristic = manhattanDistance(neighbor,lns.tasks.at(curr_id),lns.env);

		// for (int d = 0; d < 4; d++){
		// 	if (min_heuristic > heuristics[tasks[curr_id]][neighbor].d[d]){
		// 		min_heuristic = heuristics[tasks[curr_id]][neighbor].d[d];
		// 	}
		// }
		successors.emplace_back(neighbor,min_heuristic,-1);
	}

	int wait_heuristic;
#ifdef FLOW_GUIDANCE
	if (!lns.flow_heuristics.empty() && !lns.flow_heuristics[curr_id].empty()){
		s_node* s= get_flow_heuristic(lns.flow_heuristics[curr_id], lns.env, traffic, lns.flow, prev_loc);
		wait_heuristic = s==nullptr? MAX_TIMESTEP : s->get_g();
	}
#else
	if (!lns.traj_dists.empty() && !lns.traj_dists[curr_id].empty())
		wait_heuristic = get_dist_2_path(lns.traj_dists[curr_id], lns.env, traffic, prev_loc);
#endif
	else if (!lns.heuristics.at(lns.tasks.at(curr_id)).empty())
		wait_heuristic = get_heuristic(lns.heuristics[lns.tasks.at(curr_id)], lns.env, traffic, lns.flow, prev_loc);
	else
		wait_heuristic = manhattanDistance(prev_loc,lns.tasks.at(curr_id),lns.env);

	successors.emplace_back(prev_loc, wait_heuristic,	-1);

	// std::sort(successors.begin(), successors.end(), 
	quickSort(successors,0, successors.size()-1, 
		[&](State& a, State& b)
		{
			int diff[4] = {1,lns.env->cols,-1,-lns.env->cols};
			if (a.timestep == b.timestep){
#ifdef MAPFT
					if (a.orientation==orien_next && b.orientation!=orien_next)
						return true;
					if (a.orientation!=orien_next && b.orientation==orien_next)
						return false;
#endif
					return rand()%2==1;

			}
			return a.timestep < b.timestep; 
		});
#ifndef NDEBUG
	std::cout<<"curr_id: "<<curr_id<<" prev_loc: "<<prev_loc<<" prev_orientation: "<<prev_orientation<<","<< prev_states[curr_id].timestep<<","<<lns.tasks[curr_id]<<std::endl;
	for (auto& next: successors){
		std::cout<<curr_id <<" next: "<<next.location<<" "<<next.timestep<<" "<< next.tie_breaker<<" "<<next.orientation<<std::endl;
	}
	if (!lns.heuristics.at(lns.tasks.at(curr_id)).empty() && lns.tasks.at(curr_id) != prev_loc)
		assert(successors.front().location!=prev_loc);
#endif

    for (auto& next: successors){
		if(occupied[next.location])
			continue;
		assert(validateMove(prev_loc, next.location, lns.env));
		
		if (next.location == -1)
			continue;
		if (decision[next.location] != -1){
			continue;
		}
		if (higher_id != -1 && prev_decision[next.location] == higher_id){
			continue;
		}
		next_states.at(curr_id) = next;
		decision.at(next.location) = curr_id;

        if (prev_decision.at(next.location) != -1 && 
			next_states.at(prev_decision.at(next.location)).location == -1){
            int lower_id = prev_decision.at(next.location);
            if (!causalPIBT(lower_id,curr_id,prev_states,next_states, prev_decision,decision, occupied,traffic,lns)){
				continue;
            }
        }

		#ifndef NDEBUG
		std::cout<<"true: "<< next.location<<","<<next.orientation <<std::endl;
		#endif
        return true;
    }

    next_states.at(curr_id) = State(prev_loc,-1 ,-1);;
    decision.at(prev_loc) = curr_id;     

	#ifndef NDEBUG
		std::cout<<"false: "<< next_states[curr_id].location<<","<<next_states[curr_id].orientation <<std::endl;
	#endif   

    return false;
}

#ifdef MAPFT

Action getAction(State& prev, State& next){
	if (prev.location == next.location && prev.orientation == next.orientation){
		return Action::W;
	}
	if (prev.location != next.location && prev.orientation == next.orientation){
		return Action::FW;
	}
	if (next.orientation  == (prev.orientation+1)%4){
		return Action::CR;
	}
	if (next.orientation  == (prev.orientation+3)%4){
		return Action::CCR;
	}
	assert(false);
	return Action::W;
}

Action getAction(State& prev, int next_loc, SharedEnvironment* env){
	if (prev.location == next_loc){
		return Action::W;
	}
	int diff = next_loc -prev.location;
	int orientation;
	if (diff == 1){
		orientation = 0;
	}
	if (diff == -1){
		orientation = 2;
	}
	if (diff == env->cols){
		orientation = 1;
	}
	if (diff == -env->cols){
		orientation = 3;
	}
	if (orientation == prev.orientation){
		return Action::FW;
	}
	if (orientation  == (prev.orientation+1)%4){
		return Action::CR;
	}
	if (orientation  == (prev.orientation+3)%4){
		return Action::CCR;
	}
	if (orientation  == (prev.orientation+2)%4){
		return Action::CR;
	}
	assert(false);



}

bool moveCheck(int id, std::vector<bool>& checked,
		std::vector<DCR>& decided, std::vector<Action>& actions, std::vector<int>& prev_decision){
	if (checked.at(id) && actions.at(id) == Action::FW)
		return true;
	checked.at(id) = true;

	if (actions.at(id) != Action::FW)
		return false;

	//move forward
	int target = decided.at(id).loc;
	assert(target != -1);

	int na = prev_decision[target];
	if (na == -1)
		return true;

	if (moveCheck(na,checked,decided,actions,prev_decision))
		return true;
	actions.at(id) = Action::W;
	return false;
	

	

	
	
}
#else

Action getAction(State& prev, State& next){
	if (prev.location == next.location ){
		return Action::W;
	}
	if (next.location-prev.location ==1)
		return Action::R;
	if (next.location-prev.location ==-1)
		return Action::L;
	if (next.location-prev.location > 1)
		return Action::D;
	if (next.location-prev.location < -1)
		return Action::U;

	return Action::W;
}

Action getAction(State& prev, int next_loc, SharedEnvironment* env){
	if (prev.location == next_loc){
		return Action::W;
	}
	if (next_loc-prev.location ==1)
		return Action::R;
	if (next_loc-prev.location ==-1)
		return Action::L;
	if (next_loc-prev.location == env->cols)
		return Action::D;
	if (next_loc-prev.location == -env->cols)
		return Action::U;

	return Action::W;



}

bool moveCheck(int id, std::vector<bool>& checked,
		std::vector<DCR>& decided, std::vector<Action>& actions, std::vector<int>& prev_decision){
	return true;
}

#endif
}