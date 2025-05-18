#include "Evaluation.h"
#include <iostream>

#ifndef NO_ROT

void DummyPlanner::load_plans(std::string fname){
    std::ifstream ifs(fname);
    auto jf = nlohmann::json::parse(ifs);
    if (!jf.contains("actualPaths") || !jf["actualPaths"].is_array()){
        return;
    }

    for  (auto it = jf["actualPaths"].begin(); it != jf["actualPaths"].end(); ++it)
    {
        if (!it->is_string())
        {
            agent_plans.clear();
            return;
        }
        agent_plans.emplace_back();
        for (auto& ch: it->get<std::string>())
        {
            if (ch=='W')
            {
                agent_plans.back().push_back(Action::W);
            }
            else if (ch=='C')
            {
                agent_plans.back().push_back(Action::CCR);
            }
            else if (ch=='R')
            {
                agent_plans.back().push_back(Action::CR);
            }
            else if (ch=='F')
            {
                agent_plans.back().push_back(Action::FW);
            }
        }
        std::cout<<"load "<<agent_plans.size()<<"actions from "<<fname<<std::endl;
    }
}


void DummyPlanner::plan(int time_limit, std::vector<Action> & actions)
{
    actions.clear();
    for (auto & dq: agent_plans)
    {
        if (!dq.empty())
        {
            actions.push_back(dq.front());
            dq.pop_front();
        } else {
            std::cerr<<"the number of actions and simulation steps don't match!"<<std::endl;
            exit(-1);
        }
    }
}

#else

void DummyPlanner::load_plans(std::string fname){
    std::ifstream ifs(fname);
    auto jf = nlohmann::json::parse(ifs);
    if (!jf.contains("actualPaths") || !jf["actualPaths"].is_array()){
        return;
    }

    for  (auto it = jf["actualPaths"].begin(); it != jf["actualPaths"].end(); ++it)
    {
        if (!it->is_string())
        {
            agent_plans.clear();
            return;
        }
        agent_plans.emplace_back();
        for (auto& ch: it->get<std::string>())
        {
            if (ch=='R')
            {
                agent_plans.back().push_back(Action::R);
            }
            else if (ch=='D')
            {
                agent_plans.back().push_back(Action::D);
            }
            else if (ch=='L')
            {
                agent_plans.back().push_back(Action::L);
            }
            else if (ch=='U')
            {
                agent_plans.back().push_back(Action::U);
            }
            else if (ch=='W')
            {
                agent_plans.back().push_back(Action::W);
            } else if (ch=='X')
            {
                agent_plans.back().push_back(Action::NA);
            }
        }
    }
    std::cout<<"load "<<agent_plans.size()<<"actions from "<<fname<<std::endl;
}


void DummyPlanner::plan(int time_limit, std::vector<Action> & actions)
{
    actions.clear();
    for (auto & dq: agent_plans)
    {
        if (!dq.empty())
        {
            actions.push_back(dq.front());
            dq.pop_front();
        } else {
            std::cerr<<"the number of actions and simulation steps don't match!"<<std::endl;
            exit(-1);
        }
    }
}

#endif