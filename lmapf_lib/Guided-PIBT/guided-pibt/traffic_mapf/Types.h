#pragma once
#include <limits.h>


#include <vector>
#include <iostream>
#include <deque>
#include <regex>
#include <fstream>
#include <cassert>


#define MAX_TIMESTEP INT_MAX/2

#ifdef ROBOT_RUNNERS

#include "SharedEnv.h"
#include "ActionModel.h"
#endif

namespace TrafficMAPF{

	enum OBJ{
		NONE,
		O_VC,
		VC,
		SUM_OVC,
		SUI_TC,
		SUI_TG
	};
	
	enum DONE{
		NOT_DONE,
		DONE
	};

	struct Int4{
		int d[4];
	};

	struct Int2{
		int d[4];
	};

	struct DCR{
		int loc;
		int state;
	};

	typedef std::vector<int> Traj;

	// typedef std::vector<int> HeuristicTable;


	struct HNode
		{
			int label;
			int location;
			int direction;
			int value;
			int other;

			unsigned int priority;
			unsigned int get_priority() const { return priority; }
    		void set_priority(unsigned int p) { priority = p; }


			HNode() = default;
			HNode(int location,int direction, int value) : location(location), direction(direction), value(value) {}
			// the following is used to compare nodes in the OPEN list
			struct compare_node
			{
				bool operator()(const HNode& n1, const HNode& n2) const
				{
					return n1.value < n2.value;
				}
			};  // used by OPEN (open) to compare nodes (top of the open has min f-val, and then highest g-val)
		};

	struct HeuristicTable{
		std::vector<int> htable;
		std::deque<HNode> open;
		
		bool empty(){
			return htable.empty();
		}
	};



	struct d2p{
		int label = 0;
		int id = -1;
		int cost = -1;
		int togo = -1;

		d2p(int label, int id, int cost, int togo):label(label), id(id), cost(cost), togo(togo){};
	};

	struct Dist2Path{
		int label = 0;
		std::vector<d2p> dist2path;
		std::deque<d2p> open;
		bool empty(){
			return dist2path.empty();
		}

	};

	typedef std::chrono::steady_clock::time_point TimePoint;

#ifndef ROBOT_RUNNERS

#ifdef MAPFT
	enum Action {FW, CR, CCR, W, NA};
#else
	enum Action {R, D, L, U, W, NA};
#endif
	  // to load graph
	static const std::regex r_height = std::regex(R"(height\s(\d+))");
	static const std::regex r_width = std::regex(R"(width\s(\d+))");
	static const std::regex r_map = std::regex(R"(map)");

	struct State{
		int location;
		int orientation;
		int timestep;
		int tie_breaker;
		State(): location(-1), timestep(-1), orientation(-1) {};
		State(int location, int timestep, int orientation): location(location), timestep(timestep), orientation(orientation){};
	};



	class SharedEnvironment {
	public:
		int num_of_agents;
		int rows;
		int cols;
		std::vector<int> map;

		std::vector<std::vector<int>> neighbors;

		std::vector<State> curr_states;

		int max_h;


		// goal locations for each agent
		// each task is a pair of <goal_loc, reveal_time>
		SharedEnvironment(){}



		void load_map(std::string fname){
				std::ifstream file(fname);
				if (!file) {
					std::cout << "error: file " << fname << " is not found." << std::endl;
					exit(1);
				}
				std::string line;
				std::smatch results;

				// read fundamental graph parameters
				while (getline(file, line)) {
					// for CRLF coding
					if (*(line.end() - 1) == 0x0d) line.pop_back();

					if (std::regex_match(line, results, r_height)) {
					rows = std::stoi(results[1].str());
					}
					if (std::regex_match(line, results, r_width)) {
					cols = std::stoi(results[1].str());
					}
					if (std::regex_match(line, results, r_map)) break;
				}
				
				map.resize(rows * cols, 0);
				neighbors.resize(rows * cols);

				// create vertices
				uint y = 0;
				while (getline(file, line)) {
					// for CRLF coding
					if (*(line.end() - 1) == 0x0d) line.pop_back();
					for (uint x = 0; x < cols; ++x) {
						char s = line[x];
						auto index = cols * y + x;

						if (s == 'T' or s == '@'){
							map[index] = 1;  // obstacle
						};  // object
					}
					++y;
				}
				file.close();

				for (int row=0; row<rows; row++){
					for (int col=0; col<cols; col++){
						int loc = row*cols+col;
						if (map[loc]==0){
							if (row>0 && map[loc-cols]==0){
								neighbors[loc].push_back(loc-cols);
							}
							if (row<rows-1 && map[loc+cols]==0){
								neighbors[loc].push_back(loc+cols);
							}
							if (col>0 && map[loc-1]==0){
								neighbors[loc].push_back(loc-1);
							}
							if (col<cols-1 && map[loc+1]==0){
								neighbors[loc].push_back(loc+1);
							}
						}
					}
				} 
		}

	};
#endif
}

// struct DH{
// //differential heuristic

// 	std::vector<HeuristicTable> heuristics_backwards;
// 	std::vector<HeuristicTable> heuristics_forwards;

// 	std::vector<int> pivot;
//     const SharedEnvironment* env;

// 	bool empty()
// 	{
// 		return pivot.empty();
// 	}

// 	int get_heuristic(int source, int target)
// 	{
//         assert(!pivot.empty());
//         if (pivot.empty()){
//             cout<<"error: pivot is empty"<<endl;
//             exit(1);
//         }
//         int h=0;
// 		// int h = manhattanDistance(source,target,env);
// 		for (int i=0; i<pivot.size();i++){
// 			h = max(heuristics_backwards[i][source] - heuristics_backwards[i][target], h);
// 			h = max(heuristics_forwards[i][target] - heuristics_forwards[i][source], h);
// 		}
// 		return h;
// 	}
// };






