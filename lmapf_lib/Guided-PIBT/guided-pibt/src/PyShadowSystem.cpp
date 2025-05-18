#include "PyShadowSystem.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

PyShadowSystem::PyShadowSystem(
    const string & map_fp
) {
    logger=new Logger();
    grid=new Grid(map_fp);
    model=new ActionModelWithRotate(*grid);
    model->set_logger(logger);
    logger->set_level(boost::log::trivial::info);
}

PyShadowSystem::PyShadowSystem(
    const std::string & map_name,
    const std::vector<int> & map,
    int h,
    int w
) {
    logger=new Logger();
    grid=new Grid(map_name, map, h, w);
    model=new ActionModelWithRotate(*grid);
    model->set_logger(logger);
    logger->set_level(boost::log::trivial::error);
}



void PyShadowSystem::reset(
    std::vector<int> & _agents,
    std::vector<int> & _tasks,
    int _seed
) {
    agents=_agents;
    tasks=_tasks;

    if (planner!=nullptr) {
        delete planner;
    }
    if (system!=nullptr) {
        delete system;
    }
    planner= new MAPFPlanner(_seed);
    system= new InfAssignSystem(*grid, planner, agents, tasks, model);
    system->set_logger(logger);
    system->set_plan_time_limit(60); //s
    system->set_preprocess_time_limit(1800); //s
    system->set_num_tasks_reveal(1);
    system->reset();
}

void PyShadowSystem::step(
    std::vector<int> & actions
) {
    system->step(actions);
}

std::vector<int> PyShadowSystem::query() {
    return system->query();
}

void PyShadowSystem::sync(
    int timestep,
    std::vector<int> & curr_locations,
    std::vector<int> & goal_locations
) {
    system->sync(timestep, curr_locations, goal_locations);
}

std::vector<int> PyShadowSystem::query_locations() {
    return system->query_locations();
}

std::vector<int> PyShadowSystem::query_goals() {
    return system->query_goals();
}

std::vector<float> PyShadowSystem::query_heuristics(
    std::vector<int> & locs, 
    std::vector<int> & view_y,
    std::vector<int> & view_x 
) {
    return system->query_heuristics(locs, view_y, view_x);
}

std::vector<int> PyShadowSystem::query_pibt_actions() {
    return system->query_pibt_actions();
}

std::vector<int> PyShadowSystem::query_lns_actions(std::vector<int> & init_plan, int planning_window, int num_theads, int max_iterations) {
    return system->query_lns_actions(init_plan, planning_window, num_theads, max_iterations);
}

void PyShadowSystem::seed(uint _seed) {
    system->planner->trajLNS.seed(_seed);
}

PyShadowSystem::~PyShadowSystem() {
    if (grid!=nullptr) {
        delete grid;
    }
    if (model!=nullptr) {
        delete model;
    }
    if (planner!=nullptr) {
        delete planner;
    }
    if (system!=nullptr) {
        delete system;
    }
    if (logger!=nullptr) {
        delete logger;
    }
}


PYBIND11_MODULE(py_shadow_system, m) {
    pybind11::class_<PyShadowSystem>(m, "PyShadowSystem")
        .def(pybind11::init<const std::string &>())
        .def(pybind11::init<const std::string &, const std::vector<int> &, int, int>())
        .def("reset",&PyShadowSystem::reset)
        .def("step",&PyShadowSystem::step)
        .def("query",&PyShadowSystem::query)
        .def("sync",&PyShadowSystem::sync)
        .def("query_locations",&PyShadowSystem::query_locations)
        .def("query_goals",&PyShadowSystem::query_goals)
        .def("query_heuristics",&PyShadowSystem::query_heuristics)
        .def("query_pibt_actions",&PyShadowSystem::query_pibt_actions)
        .def("query_lns_actions",&PyShadowSystem::query_lns_actions)
        .def("seed",&PyShadowSystem::seed)
    ;
}