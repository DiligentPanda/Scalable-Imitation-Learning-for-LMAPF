/*
 * lacam-star
 */

#pragma once

#include "LaCAM2/graph.hpp"
#include "LaCAM2/instance.hpp"
#include "LaCAM2/utils.hpp"
#include "util/HeuristicTable.h"
#include <memory>
#include "LaCAM2/executor.hpp"

namespace LaCAM2 {

// objective function
enum Objective { OBJ_NONE, OBJ_MAKESPAN, OBJ_SUM_OF_LOSS };
std::ostream& operator<<(std::ostream& os, const Objective objective);

// PIBT agent
struct Agent {
  const uint id;
  Vertex* v_now;   // current location
  Vertex* v_next;  // next location
  int o_next;
  Agent(uint _id) : id(_id), v_now(nullptr), v_next(nullptr), o_next(-1) {}
};
using Agents = std::vector<Agent*>;

// low-level node
struct LNode {
  std::vector<uint> who;
  std::vector<std::tuple<Vertex*,int > > where; // vertex and orientation
  const uint depth;
  LNode(LNode* parent, uint i, const std::tuple<Vertex*,int > & t);  // who and where
  LNode(): who(), where(), depth(0) {}
};

// high-level node
struct HNode {
  static uint HNODE_CNT;  // count #(high-level node)
  const Config C;

  // tree
  HNode* parent;
  std::set<HNode*> neighbor;

  uint d;        // depth (might be updated, it might be different from g)

  // costs
  float g;        // g-value (might be updated)
  float h;  // h-value
  float f;        // g + h (might be updated)

  // for low-level search
  std::vector<float> priorities;
  std::vector<uint> order;
  std::queue<LNode*> search_tree;
  
  int order_strategy;

  HNode(const Config& _C, const std::shared_ptr<HeuristicTable> & HT, Instance * ins, HNode* _parent, float _g,
        float _h, const uint _d, int order_strategy, bool disable_agent_goals);
  ~HNode();
};
using HNodes = std::vector<HNode*>;

struct Planner {
  Instance* ins;
  const Deadline* deadline;
  std::mt19937* MT;
  const int verbose;
  bool use_swap;  // use swap operation or not
  bool use_orient_in_heuristic;  // use orientation in heuristic or not
  bool use_external_executor;  // use external executor or not
  int MC_idx;

  // hyper parameters
  const Objective objective;
  const float RESTART_RATE;  // random restart

  // solver utils
  const uint N;       // number of agents
  const uint V_size;  // number o vertices
  std::shared_ptr<HeuristicTable> HT;
  std::shared_ptr<std::vector<float> > map_weights; // map weights
  uint loop_cnt;      // auxiliary

  // used in PIBT
  std::vector<std::array<Vertex*, 5> > C_next;  // next locations, used in PIBT
  std::vector<float> tie_breakers;              // random values, used in PIBT
  Agents A;
  Agents occupied_now;                          // for quick collision checking
  Agents occupied_next;                         // for quick collision checking

  bool disable_agent_goals;

  float get_cost_move(int pst,int ped);

  Planner(Instance* _ins, const std::shared_ptr<HeuristicTable> & HT, const std::shared_ptr<std::vector<float> > & map_weights, const Deadline* _deadline, std::mt19937* _MT,
          const int _verbose = 0,
          // other parameters
          const Objective _objective = OBJ_NONE,
          const float _restart_rate = 0.001f,
          bool use_swap=false,
          bool use_orient_in_heuristic=false,
          bool use_external_executor=false,
          bool disable_agent_goals=true);
  ~Planner();

  Executor executor;

  Solution solve(std::string& additional_info, int order_strategy);

  std::vector<std::tuple<Vertex *,int> > get_successors(Vertex *v, int orient);

  void expand_lowlevel_tree(HNode* H, LNode* L);
  void rewrite(HNode* H_from, HNode* T, HNode* H_goal,
               std::stack<HNode*>& OPEN);
  float get_edge_cost(const Config& C1, const Config& C2);
  float get_edge_cost(HNode* H_from, HNode* H_to);
  float get_h_value(const Config& C);
  bool get_new_config(HNode* H, LNode* L);
  bool funcPIBT(Agent* ai, HNode * H);

  // swap operation
  Agent* swap_possible_and_required(Agent* ai);
  bool is_swap_required(const uint pusher, const uint puller,
                        Vertex* v_pusher_origin, Vertex* v_puller_origin);
  bool is_swap_possible(Vertex* v_pusher_origin, Vertex* v_puller_origin);

  // utilities
  template <typename... Body>
  void solver_info(const int level, Body&&... body)
  {
    if (verbose < level) return;
    std::cout << "elapsed:" << std::setw(6) << elapsed_ms(deadline) << "ms"
              << "  loop_cnt:" << std::setw(8) << loop_cnt
              << "  node_cnt:" << std::setw(8) << HNode::HNODE_CNT << "\t";
    info(level, verbose, (body)...);
  }
};

}  // namespace LaCAM2