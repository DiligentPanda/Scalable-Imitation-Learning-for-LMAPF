/*
 * distance table with lazy evaluation, using BFS
 */
#pragma once

#include "graph.hpp"
#include "instance.hpp"
#include "utils.hpp"
#include <vector>
#include <Types.h>
#include <TrajLNS.h>



struct DistTable {
  const uint V_size;  // number of vertices
  std::vector<std::vector<uint> >
      table;          // distance table, index: agent-id & vertex-id
  std::vector<std::queue<Vertex*> > OPEN;  // search queue

  inline uint get(uint i, uint v_id);      // agent, vertex-id
  uint get(uint i, Vertex* v);             // agent, vertex
  uint get_g(uint i, Vertex* v);             // agent, vertex
  uint get_gd(uint i, Vertex* v);             // agent, vertex

  DistTable(const Instance& ins);
  DistTable(const Instance* ins);

  TrafficMAPF::SharedEnvironment env;

  TrafficMAPF::TrajLNS lns;

  std::vector<int> traffic;

  bool guidance_ready = false;

  void setup(const Instance* ins);  // initialization
  void setup_guidance(const Instance* ins, int t_ms);
};
