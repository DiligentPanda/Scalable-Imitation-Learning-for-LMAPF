#include "LaCAM2/post_processing.hpp"

namespace LaCAM2 {

bool is_feasible_solution(const Instance& ins, const Solution& solution,
                          const int verbose)
{
  // if (solution.empty()) return true;

  // // check start locations
  // if (!is_same_config(solution.front(), ins.starts)) {
  //   info(1, verbose, "invalid starts");
  //   return false;
  // }

  // // check goal locations
  // if (!is_same_config(solution.back(), ins.goals)) {
  //   info(1, verbose, "invalid goals");
  //   return false;
  // }

  // for (size_t t = 1; t < solution.size(); ++t) {
  //   for (size_t i = 0; i < ins.N; ++i) {
  //     auto v_i_from = solution[t - 1][i];
  //     auto v_i_to = solution[t][i];
  //     // check connectivity
  //     if (v_i_from != v_i_to &&
  //         std::find(v_i_to->neighbor.begin(), v_i_to->neighbor.end(),
  //                   v_i_from) == v_i_to->neighbor.end()) {
  //       info(1, verbose, "invalid move");
  //       return false;
  //     }

  //     // check conflicts
  //     for (size_t j = i + 1; j < ins.N; ++j) {
  //       auto v_j_from = solution[t - 1][j];
  //       auto v_j_to = solution[t][j];
  //       // vertex conflicts
  //       if (v_j_to == v_i_to) {
  //         info(1, verbose, "vertex conflict");
  //         return false;
  //       }
  //       // swap conflicts
  //       if (v_j_to == v_i_from && v_j_from == v_i_to) {
  //         info(1, verbose, "edge conflict");
  //         return false;
  //       }
  //     }
  //   }
  // }

  return true;
}

float get_makespan(const Instance& ins, const std::shared_ptr<HeuristicTable>& HT, const Solution& solution)
{
  if (solution.empty()) return 0;
  float c = 0;
  for (auto i=0;i<ins.N;++i) {
    c = std::max(c,(float)(solution.size()-1) + HT->get(solution.back().locs[i]->index,solution.back().orients[i],ins.goals.locs[i]->index));
  }
  return c;
}

int get_path_cost(const Solution& solution, uint i)
{
  const auto makespan = solution.size();
  const auto g = solution.back().locs[i];
  auto c = makespan;
  while (c > 0 && solution[c - 1].locs[i] == g) --c;
  return c;
}

int get_sum_of_costs(const Solution& solution)
{
  if (solution.empty()) return 0;
  int c = 0;
  const auto N = solution.front().size();
  for (size_t i = 0; i < N; ++i) c += get_path_cost(solution, i);
  return c;
}

float get_sum_of_loss(const Instance& ins, const std::shared_ptr<HeuristicTable>& HT, const Solution& solution)
{
  if (solution.empty()) return 0;
  float c = 0;
  const auto N = solution.front().size();
  const auto T = solution.size();
  for (size_t i = 0; i < N; ++i) {
    c+= (float)(solution.size()-1) + HT->get(solution.back().locs[i]->index,solution.back().orients[i],ins.goals.locs[i]->index);
  }
  return c;
}

float get_makespan_lower_bound(const Instance& ins, const std::shared_ptr<HeuristicTable>& HT)
{
  float c = 0;
  for (size_t i = 0; i < ins.N; ++i) {
    c = std::max(c, HT->get(ins.starts.locs[i]->index,ins.goals.locs[i]->index));
  }
  return c;
}

float get_sum_of_costs_lower_bound(const Instance& ins, const std::shared_ptr<HeuristicTable>& HT)
{
  float c = 0;
  for (size_t i = 0; i < ins.N; ++i) {
    c += HT->get(ins.starts.locs[i]->index,ins.goals.locs[i]->index);
  }
  return c;
}

void print_stats(const int verbose, const Instance& ins, const std::shared_ptr<HeuristicTable>& HT,
                 const Solution& solution, const double comp_time_ms)
{
  auto ceil = [](float x) { return std::ceil(x * 100) / 100; };
  const auto makespan = get_makespan(ins,HT,solution);
  const auto makespan_lb = get_makespan_lower_bound(ins, HT);
  // const auto sum_of_costs = get_sum_of_costs(solution);
  const auto sum_of_costs_lb = get_sum_of_costs_lower_bound(ins, HT);
  const auto sum_of_loss = get_sum_of_loss(ins,HT,solution);
  g_timer.record_p("info_s");
  // TODO(rivers): not sure why it is so slow...
  // ONLYDEV(
  //   g_logger.info("solved: {} ms makespan: {} (lb={}, ratio={}) sum_of_loss: {} (lb={}, ratio={})",
  //   comp_time_ms, makespan, makespan_lb, ceil((float)makespan / makespan_lb),
  //   sum_of_loss, sum_of_costs_lb, ceil((float)sum_of_loss / sum_of_costs_lb));
  // )
  info(1, verbose, "solved: ", comp_time_ms, "ms", "\tmakespan: ", makespan,
       " (lb=", makespan_lb, ", ratio=", ceil((float)makespan / makespan_lb), ")",
      //  "\tsum_of_costs: ", sum_of_costs, " (lb=", sum_of_costs_lb,
      //  ", ub=", ceil((float)sum_of_costs / sum_of_costs_lb), ")",
       "\tsum_of_loss: ", sum_of_loss, " (lb=", sum_of_costs_lb,
       ", ratio=", ceil((float)sum_of_loss / sum_of_costs_lb), ")");
  g_timer.record_d("info_s", "info");
}

// for log of map_name
static const std::regex r_map_name = std::regex(R"(.+/(.+))");

void make_log(const Instance& ins, const std::shared_ptr<HeuristicTable>& HT, const Solution& solution,
              const std::string& output_name, const double comp_time_ms,
              const std::string& map_name, const int seed,
              const std::string& additional_info, const bool log_short)
{
  // map name
  std::smatch results;
  const auto map_recorded_name =
      (std::regex_match(map_name, results, r_map_name)) ? results[1].str()
                                                        : map_name;

  // log for visualizer
  auto get_x = [&](int k) { return k % ins.G.width; };
  auto get_y = [&](int k) { return k / ins.G.width; };
  std::ofstream log;
  log.open(output_name, std::ios::out);
  log << "agents=" << ins.N << "\n";
  log << "map_file=" << map_recorded_name << "\n";
  log << "solver=planner\n";
  log << "solved=" << !solution.empty() << "\n";
  // log << "soc=" << get_sum_of_costs(solution) << "\n";
  // log << "soc_lb=" << get_sum_of_costs_lower_bound(ins, HT) << "\n";
  log << "makespan=" << get_makespan(ins,HT,solution) << "\n";
  log << "makespan_lb=" << get_makespan_lower_bound(ins, HT) << "\n";
  log << "sum_of_loss=" << get_sum_of_loss(ins,HT,solution) << "\n";
  log << "sum_of_loss_lb=" << get_sum_of_costs_lower_bound(ins, HT)
      << "\n";
  log << "comp_time=" << comp_time_ms << "\n";
  log << "seed=" << seed << "\n";
  log << additional_info;
  if (log_short) return;
  log << "starts=";
  for (size_t i = 0; i < ins.N; ++i) {
    auto k = ins.starts.locs[i]->index;
    log << "(" << get_x(k) << "," << get_y(k) << "),";
  }
  log << "\ngoals=";
  for (size_t i = 0; i < ins.N; ++i) {
    auto k = ins.goals.locs[i]->index;
    log << "(" << get_x(k) << "," << get_y(k) << "),";
  }
  log << "\nsolution=\n";
  for (size_t t = 0; t < solution.size(); ++t) {
    log << t << ":";
    auto C = solution[t];
    for (auto v : C.locs) {
      log << "(" << get_x(v->index) << "," << get_y(v->index) << "),";
    }
    log << "\n";
  }
  log.close();
}

}  // namespace LaCAM2