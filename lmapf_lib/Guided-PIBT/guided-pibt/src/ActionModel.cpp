#include "ActionModel.h"

std::ostream& operator<<(std::ostream &stream, const Action &action){
#ifdef MAPFT
    if (action == Action::FW){
        stream << "F";
    } else if (action == Action::CR){
        stream << "R";
    } else if (action == Action::CCR){
        stream << "C";
    } else {
        stream << "W";
    }
#else
    if (action == Action::R){
        stream << "R";
    } else if (action == Action::D){
        stream << "D";
    } else if (action == Action::L){
        stream << "L";
    } else {
        stream << "U";
    }
#endif

    return stream;
}

bool ActionModelWithRotate::is_valid(const vector<State>& prev, const vector<Action> & actions){
    if (prev.size() != actions.size()) {
        errors.push_back(make_tuple("incorrect vector size",-1,-1,prev[0].timestep+1));
        return false;
    }

    vector<State> next = result_states(prev, actions);
    unordered_map<int, int> occupied;
    unordered_map<std::pair<int,int>, int> occupied_edge;

    for (int i = 0; i < prev.size(); i ++) {
        /*
          if (prev[i].location == next[i].location) {
          // check if the rotation is not larger than 90 degree
          if (abs(prev[i].orientation - next[i].orientation) == 2){
          cout << "ERROR: agent " << i << " over-rotates. " << endl;
          errors.push_back(make_tuple("over-rotate",i,-1,next[i].timestep));
          return false;
          }
          } else {
          if (prev[i].orientation != next[i].orientation){
          cout << "ERROR: agent " << i << " moves and rotates at the same time. " << endl;
          errors.push_back(make_tuple("unallowed move",i,-1,next[i].timestep));
          return false;
          }
          if (next[i].location - prev[i].location != moves[prev[i].orientation]){
          cout << "ERROR: agent " << i << " moves in a wrong direction. " << endl;
          errors.push_back(make_tuple("unallowed move",i,-1,next[i].timestep));
          return false;
          }

          if (abs(next[i].location / cols - prev[i].location/cols) + abs(next[i].location % cols - prev[i].location %cols) > 1  ){
          cout << "ERROR: agent " << i << " moves more than 1 steps. " << endl;
          errors.push_back(make_tuple("unallowed move",i,-1,next[i].timestep));
          return false;
          }
          }
        */
       std::pair<int,int> edge = make_pair(next[i].location,prev[i].location);
        if (grid.map[next[i].location] == 1) {
            cout << "ERROR: agent " << i << " moves to an obstacle. " << endl;
            errors.push_back(make_tuple("unallowed move",i,-1,next[i].timestep));
            return false;
        }

        if (occupied.find(next[i].location) != occupied.end()) {
            cout << "ERROR: agents " << i << " and " << occupied[next[i].location] << " have a vertex conflict. "<<next[i].location << endl;
            errors.push_back(make_tuple("vertex conflict",i,occupied[next[i].location],next[i].timestep));
            return false;
        }
        // int edge_idx = (prev[i].location + 1) * rows * cols +  next[i].location;
        if (occupied_edge.find(edge) != occupied_edge.end()) {
            cout << "ERROR: agents " << i << " and " << occupied_edge[edge] << " have an edge conflict. " << endl;
            errors.push_back(make_tuple("edge conflict",i,occupied_edge[edge],next[i].timestep));
            return false;
        }

        occupied[next[i].location] = i;
        // int r_edge_idx = (next[i].location + 1) * rows * cols +  prev[i].location;
        occupied_edge[edge] = i;
    }

    return true;
}
