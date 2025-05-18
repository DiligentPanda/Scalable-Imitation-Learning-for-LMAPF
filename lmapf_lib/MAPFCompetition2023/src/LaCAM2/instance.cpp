#include "LaCAM2/instance.hpp"

namespace LaCAM2 {

Instance::Instance(
    const Graph & G,
    const std::vector<std::pair<uint,int> >& start_indexes,
    const std::vector<std::pair<uint,int> >& goal_indexes,
    std::vector<AgentInfo>& agent_infos,
    int planning_window,
    std::vector<::Path> * _precomputed_paths
    ):
      G(G),
      starts(agent_infos.size()),
      goals(agent_infos.size()),
      N(agent_infos.size()),
      agent_infos(agent_infos),
      planning_window(planning_window), 
      precomputed_paths(_precomputed_paths) {
  // for (auto k : start_indexes) starts.push_back(G.U[k]);
  // for (auto k : goal_indexes) goals.push_back(G.U[k]);

  for (int i=0;i<N;++i) {
    starts.locs[i]=G.U[start_indexes[i].first];
    starts.orients[i]=start_indexes[i].second;
    goals.locs[i]=G.U[goal_indexes[i].first];
    goals.orients[i]=goal_indexes[i].second;
  }

}

void Instance::set_starts_and_goals(std::vector<::State> * starts_ptr, std::vector<::State> * goals_ptr) {
  auto & starts=*starts_ptr;
  auto & goals=*goals_ptr;

  for (int i=0;i<N;++i) {
    this->starts.locs[i]=G.U[starts[i].location];
    this->starts.orients[i]=starts[i].orientation;
    this->goals.locs[i]=G.U[goals[i].location];
    this->goals.orients[i]=goals[i].orientation;
  }
}

// // for load instance
// static const std::regex r_instance =
//     std::regex(R"(\d+\t.+\.map\t\d+\t\d+\t(\d+)\t(\d+)\t(\d+)\t(\d+)\t.+)");

// Instance::Instance(const std::string& scen_filename,
//                    const std::string& map_filename, const uint _N)
//     : G(Graph(map_filename)), starts(Config()), goals(Config()), N(_N)
// {
//   // load start-goal pairs
//   std::ifstream file(scen_filename);
//   if (!file) {
//     info(0, 0, scen_filename, " is not found");
//     return;
//   }
//   std::string line;
//   std::smatch results;

//   while (getline(file, line)) {
//     // for CRLF coding
//     if (*(line.end() - 1) == 0x0d) line.pop_back();

//     if (std::regex_match(line, results, r_instance)) {
//       uint x_s = std::stoi(results[1].str());
//       uint y_s = std::stoi(results[2].str());
//       uint x_g = std::stoi(results[3].str());
//       uint y_g = std::stoi(results[4].str());
//       if (x_s < 0 || G.width <= x_s || x_g < 0 || G.width <= x_g) continue;
//       if (y_s < 0 || G.height <= y_s || y_g < 0 || G.height <= y_g) continue;
//       auto s = G.U[G.width * y_s + x_s];
//       auto g = G.U[G.width * y_g + x_g];
//       if (s == nullptr || g == nullptr) continue;
//       starts.push_back(s);
//       goals.push_back(g);
//     }

//     if (starts.size() == N) break;
//   }
// }

// Instance::Instance(const std::string& map_filename, std::mt19937* MT,
//                    const uint _N)
//     : G(Graph(map_filename)), starts(Config()), goals(Config()), N(_N)
// {
//   // random assignment
//   const auto V_size = G.size();

//   // set starts
//   auto s_indexes = std::vector<uint>(V_size);
//   std::iota(s_indexes.begin(), s_indexes.end(), 0);
//   std::shuffle(s_indexes.begin(), s_indexes.end(), *MT);
//   size_t i = 0;
//   while (true) {
//     if (i >= V_size) return;
//     starts.push_back(G.V[s_indexes[i]]);
//     if (starts.size() == N) break;
//     ++i;
//   }

//   // set goals
//   auto g_indexes = std::vector<uint>(V_size);
//   std::iota(g_indexes.begin(), g_indexes.end(), 0);
//   std::shuffle(g_indexes.begin(), g_indexes.end(), *MT);
//   size_t j = 0;
//   while (true) {
//     if (j >= V_size) return;
//     goals.push_back(G.V[g_indexes[j]]);
//     if (goals.size() == N) break;
//     ++j;
//   }
// }

bool Instance::is_valid(const int verbose) const
{
  if (N != starts.size() || N != goals.size()) {
    info(1, verbose, "invalid N, check instance");
    return false;
  }
  return true;
}

std::ostream& operator<<(std::ostream& os, const Solution& solution)
{
  auto N = solution.front().size();
  for (size_t i = 0; i < N; ++i) {
    os << std::setw(5) << i << ":";
    for (size_t k = 0; k < solution[i].size(); ++k) {
      if (k > 0) os << "->";
      os << "(" << std::setw(5) << solution[i].locs[k]->index <<"," << solution[i].orients[k] << ")"; 
    }
    os << std::endl;
  }
  return os;
}

}  // namespace LaCAM2