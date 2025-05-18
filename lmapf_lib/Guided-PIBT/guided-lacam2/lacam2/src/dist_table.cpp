#include "../include/dist_table.hpp"
#include <heuristics.hpp>
#include <flow.hpp>

DistTable::DistTable(const Instance& ins)
    : V_size(ins.G.V.size()), table(ins.N, std::vector<uint>(V_size, V_size))
{
  setup(&ins);
}

DistTable::DistTable(const Instance* ins)
    : V_size(ins->G.V.size()), table(ins->N, std::vector<uint>(V_size, V_size))
{
  setup(ins);
}

void DistTable::setup(const Instance* ins)
{


  for (size_t i = 0; i < ins->N; ++i) {
    OPEN.push_back(std::queue<Vertex*>());
    auto n = ins->goals[i];
    OPEN[i].push(n);
    table[i][n->id] = 0;
  }


}

void DistTable::setup_guidance(const Instance* ins, int t_ms)
{
  std::srand(0);
  guidance_ready = true;

  env.load_map(ins->G.filename);
  env.num_of_agents = ins->N;
  auto start_time = std::chrono::steady_clock::now();


  lns = TrafficMAPF::TrajLNS(&env);
  lns.init_mem();
  lns.start_time = start_time;
  lns.t_ms = GUID_T*1000 - 1*(GUID_T/0.005);


  traffic.resize(env.map.size(),-1);

  env.curr_states.resize(ins->N);
  for (size_t i = 0; i < ins->N; ++i) {
    lns.tasks[i] = ins->goals[i]->index;
    env.curr_states[i].location = ins->starts[i]->index;
    init_heuristic(lns.heuristics[ins->goals[i]->index],&env,ins->goals[i]->index);

  }

#ifdef GUIDANCE
#ifdef INIT_PP
        init_traj(lns, traffic, MAX_TIMESTEP);
#else
        init_traj_st(lns, traffic);
  #ifndef FLOW_GUIDANCE
        init_traj(lns, traffic, MAX_TIMESTEP, true);
  #endif
#endif
  std::cout << "---init-t," << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_time).count() <<std::endl;
  std::unordered_set<int> updated;

#ifdef GUIDANCE_LNS
  std::cout << "---op-flow,"<<lns.op_flow <<std::endl;
  std::cout << "---vertex-flow,"<<lns.vertex_flow <<std::endl;
  // destory and improve traj
  destory_improve(lns, traffic, updated, GUIDANCE_LNS, GUIDANCE_LNS);



#endif
  std::cout << "---lns-t," << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_time).count() <<std::endl;


  std::cout << "---op-flow,"<<lns.op_flow <<std::endl;
  std::cout << "---vertex-flow"<<lns.vertex_flow <<std::endl;

#ifndef FLOW_GUIDANCE
  // std::cout<<"Init dist to path"<<std::endl;
  init_dist_table(lns,traffic, MAX_TIMESTEP);

#else

  for (int i = 0; i < env.num_of_agents; i++){
    if (lns.t_ms !=0 && i%100 ==0 && std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - lns.start_time).count() >lns.t_ms){
            break;
    }
    int goal_loc = lns.tasks[i];
    init_flow_heuristic(lns, traffic, i);

}
#endif

  std::cout << "---setup-t," << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_time).count() <<std::endl;

#endif

}

uint DistTable::get(uint i, uint v_id)
{
  if (table[i][v_id] < V_size) return table[i][v_id];

  /*
   * BFS with lazy evaluation
   * c.f., Reverse Resumable A*
   * https://www.aaai.org/Papers/AIIDE/2005/AIIDE05-020.pdf
   *
   * sidenote:
   * tested RRA* but lazy BFS was much better in performance
   */

  while (!OPEN[i].empty()) {
    auto&& n = OPEN[i].front();
    OPEN[i].pop();
    const int d_n = table[i][n->id];
    for (auto&& m : n->neighbor) {
      const int d_m = table[i][m->id];
      if (d_n + 1 >= d_m) continue;
      table[i][m->id] = d_n + 1;
      OPEN[i].push(m);
    }
    if (n->id == v_id) return d_n;
  }
  return V_size;
}

uint DistTable::get(uint i, Vertex* v) {
#ifdef Guidance
  return  TrafficMAPF::get_heuristic(lns.heuristics[lns.tasks[i]], lns.env, traffic,lns.flow, v->index);
#else
  return get(i, v->id); 
#endif

  }

uint DistTable::get_g(uint i, Vertex* v) {
#ifdef GUIDANCE
  #ifndef FLOW_GUIDANCE
    if (lns.traj_dists[i].empty()){
      // return get(i, v->id); 
      return  TrafficMAPF::get_heuristic(lns.heuristics[lns.tasks[i]], lns.env, traffic,lns.flow, v->index);
    }
    return TrafficMAPF::get_dist_2_path(lns.traj_dists[i], lns.env, traffic, v->index);
  #else
  if (lns.flow_heuristics[i].empty()){
    // return get(i, v->id); 
      return  TrafficMAPF::get_heuristic(lns.heuristics[lns.tasks[i]], lns.env, traffic,lns.flow, v->index);
    }
  TrafficMAPF::s_node* s =  TrafficMAPF::get_flow_heuristic(lns.flow_heuristics[i], lns.env, traffic, lns.flow, v->index);
  return s->get_g();
  #endif
#else

  return get(i, v->id); 
#endif
  }

uint DistTable::get_gd(uint i, Vertex* v) {
#ifdef GUIDANCE
  #ifndef FLOW_GUIDANCE
    if (lns.traj_dists[i].empty()){
      // return get(i, v->id); 
      return  TrafficMAPF::get_heuristic(lns.heuristics[lns.tasks[i]], lns.env, traffic,lns.flow, v->index);
    }
    return TrafficMAPF::get_dist_2_path(lns.traj_dists[i], lns.env, traffic, v->index);
  #else
  if (lns.flow_heuristics[i].empty()){
    // return get(i, v->id); 
      return  TrafficMAPF::get_heuristic(lns.heuristics[lns.tasks[i]], lns.env, traffic,lns.flow, v->index);
    }
  TrafficMAPF::s_node* s =  TrafficMAPF::get_flow_heuristic(lns.flow_heuristics[i], lns.env, traffic, lns.flow, v->index);
  return s->depth;
#endif
#else

  return get(i, v->id); 
#endif
  }

