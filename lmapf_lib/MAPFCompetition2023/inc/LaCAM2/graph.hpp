/*
 * graph definition
 */
#pragma once
#include "LaCAM2/utils.hpp"
#include "SharedEnv.h"

namespace LaCAM2 {

struct Vertex {
  const uint id;     // index for V in Graph
  const uint index;  // index for U, width * y + x, in Graph
  std::vector<Vertex*> neighbor;

  Vertex(uint _id, uint _index);
};
using Vertices = std::vector<Vertex*>;
// using Config = std::vector<Vertex*>;  // a set of locations for all agents

struct Config {
  std::vector<Vertex*> locs;
  std::vector<int> orients;
  std::vector<bool> arrivals;

  Config(int N) {
    locs.resize(N, nullptr);
    orients.resize(N, -1);
    arrivals.resize(N, false);
  }

  inline size_t size() const {
    return locs.size();
  };

  
  inline bool all_arrived() const {
    for (int i=0;i<arrivals.size();++i) {
      if (!arrivals[i]) return false;
    }
    return true;
  }

  struct ConfigEqual {
    bool operator()(const Config& C1, const Config& C2) const {
        const auto N = C1.size();
        for (size_t i = 0; i < N; ++i) {
          if (C1.locs[i]->id != C2.locs[i]->id) return false;
          if (C1.orients[i] != C2.orients[i]) return false;
          if (C1.arrivals[i] != C2.arrivals[i]) return false;
        }
        return true;
    }
  };


};

struct Graph {
  Vertices V;                          // without nullptr
  Vertices U;                          // with nullptr
  uint width;                          // grid width
  uint height;                         // grid height
  Graph();
  Graph(const std::string& filename);  // taking map filename
  // for competition
  Graph(const SharedEnvironment & env);
  ~Graph();

  uint size() const;  // the number of vertices
};

bool is_same_config(
    const Config& C1,
    const Config& C2);  // check equivalence of two configurations

// hash function of configuration
// c.f.
// https://stackoverflow.com/questions/10405030/c-unordered-map-fail-when-used-with-a-vector-as-key
struct ConfigHasher {
  uint operator()(const Config& C) const;
};

std::ostream& operator<<(std::ostream& os, const Vertex* v);
std::ostream& operator<<(std::ostream& os, const Config& config);

}  // namespace LaCAM2