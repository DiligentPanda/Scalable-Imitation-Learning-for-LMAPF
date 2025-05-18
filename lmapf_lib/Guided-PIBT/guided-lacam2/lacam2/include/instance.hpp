/*
 * instance definition
 */
#pragma once
#include <random>
#include <iostream>
#include <fstream>

#include "graph.hpp"
#include "utils.hpp"

struct Instance {
  const Graph G;
  Config starts;
  Config goals;
  const uint N;  // number of agents
  std::mt19937* MT;
  // for testing
  Instance(const std::string& map_filename,
           const std::vector<uint>& start_indexes,
           const std::vector<uint>& goal_indexes);
  // for MAPF benchmark
  Instance(const std::string& scen_filename, const std::string& map_filename,std::mt19937* MT,
           const uint _N = 1);
  // random instance generation
  Instance(const std::string& map_filename, std::mt19937* MT,
           const uint _N = 1);
  ~Instance() {}

  // simple feasibility check of instance
  bool is_valid(const int verbose = 0) const;

  void random_agents();

};

// solution: a sequence of configurations
using Solution = std::vector<Config>;
std::ostream& operator<<(std::ostream& os, const Solution& solution);
