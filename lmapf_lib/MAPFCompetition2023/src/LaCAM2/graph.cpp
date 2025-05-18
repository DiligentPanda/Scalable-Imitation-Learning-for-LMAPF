#include "LaCAM2/graph.hpp"

namespace LaCAM2 {

Vertex::Vertex(uint _id, uint _index)
    : id(_id), index(_index), neighbor(Vertices())
{
}

Graph::Graph() : V(Vertices()), width(0), height(0) {}
Graph::~Graph()
{
  for (auto& v : V)
    if (v != nullptr) delete v;
  V.clear();
}


Graph::Graph(const SharedEnvironment & env): V(Vertices()), width(env.cols), height(env.rows) {
  U = Vertices(width * height, nullptr);
  cout<<"graph size "<<U.size()<<endl;

  // create vertices
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int idx=i*width+j;
      int loc = env.map[idx];
      if (loc==1) continue;  // object
      auto v = new Vertex(V.size(), idx);
      V.push_back(v);
      U[idx] = v;
      // cout<<"vertex id "<<U[idx]->id<<" "<<U[idx]->index<<endl;
    }
  }

  // create edges
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      auto v = U[width * y + x];
      if (v == nullptr) continue;
      // left
      if (x > 0) {
        auto u = U[width * y + (x - 1)];
        if (u != nullptr) v->neighbor.push_back(u);
      }
      // right
      if (x < width - 1) {
        auto u = U[width * y + (x + 1)];
        if (u != nullptr) v->neighbor.push_back(u);
      }
      // up
      if (y < height - 1) {
        auto u = U[width * (y + 1) + x];
        if (u != nullptr) v->neighbor.push_back(u);
      }
      // down
      if (y > 0) {
        auto u = U[width * (y - 1) + x];
        if (u != nullptr) v->neighbor.push_back(u);
      }
    }
  }

}



// to load graph
static const std::regex r_height = std::regex(R"(height\s(\d+))");
static const std::regex r_width = std::regex(R"(width\s(\d+))");
static const std::regex r_map = std::regex(R"(map)");

Graph::Graph(const std::string& filename) : V(Vertices()), width(0), height(0)
{
  std::ifstream file(filename);
  if (!file) {
    std::cout << "file " << filename << " is not found." << std::endl;
    return;
  }
  std::string line;
  std::smatch results;

  // read fundamental graph parameters
  while (getline(file, line)) {
    // for CRLF coding
    if (*(line.end() - 1) == 0x0d) line.pop_back();

    if (std::regex_match(line, results, r_height)) {
      height = std::stoi(results[1].str());
    }
    if (std::regex_match(line, results, r_width)) {
      width = std::stoi(results[1].str());
    }
    if (std::regex_match(line, results, r_map)) break;
  }

  U = Vertices(width * height, nullptr);

  // create vertices
  uint y = 0;
  while (getline(file, line)) {
    // for CRLF coding
    if (*(line.end() - 1) == 0x0d) line.pop_back();
    for (uint x = 0; x < width; ++x) {
      char s = line[x];
      if (s == 'T' or s == '@') continue;  // object
      auto index = width * y + x;
      auto v = new Vertex(V.size(), index);
      V.push_back(v);
      U[index] = v;
    }
    ++y;
  }
  file.close();

  // create edges
  for (uint y = 0; y < height; ++y) {
    for (uint x = 0; x < width; ++x) {
      auto v = U[width * y + x];
      if (v == nullptr) continue;
      // left
      if (x > 0) {
        auto u = U[width * y + (x - 1)];
        if (u != nullptr) v->neighbor.push_back(u);
      }
      // right
      if (x < width - 1) {
        auto u = U[width * y + (x + 1)];
        if (u != nullptr) v->neighbor.push_back(u);
      }
      // up
      if (y < height - 1) {
        auto u = U[width * (y + 1) + x];
        if (u != nullptr) v->neighbor.push_back(u);
      }
      // down
      if (y > 0) {
        auto u = U[width * (y - 1) + x];
        if (u != nullptr) v->neighbor.push_back(u);
      }
    }
  }
}

uint Graph::size() const { return V.size(); }

bool is_same_config(const Config& C1, const Config& C2)
{
  const auto N = C1.size();
  for (size_t i = 0; i < N; ++i) {
    if (C1.locs[i]->id != C2.locs[i]->id) return false;
    if (C1.orients[i] != C2.orients[i]) return false;
    if (C1.arrivals[i] != C2.arrivals[i]) return false;
  }
  return true;
}

uint ConfigHasher::operator()(const Config& C) const
{
  // TODO: rivers: has is not very necessary in our case, because we don't use LaCAM to expand a large search tree.
  // we will just do it simple here and revisit it later.
  uint hash = C.size();
  for (auto i=0;i<C.size();++i) {
    hash ^= C.locs[i]->id<<3 + C.orients[i]<<1 + C.arrivals[i] + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  }
  return hash;
}

std::ostream& operator<<(std::ostream& os, const Vertex* v)
{
  os << v->index;
  return os;
}

std::ostream& operator<<(std::ostream& os, const Config& config)
{
  os << "<";
  const auto N = config.size();
  for (size_t i = 0; i < N; ++i) {
    if (i > 0) os << ",";
    os << "(" <<std::setw(5) << config.locs[i]<<","<< config.orients[i]<<","<<config.arrivals[i]<<")";
  }
  os << ">";
  return os;
}

}  // namespace LaCAM2