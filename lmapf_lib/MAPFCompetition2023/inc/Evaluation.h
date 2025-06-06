#pragma once
#include "MAPFPlanner.h"

class DummyPlanner: public MAPFPlanner
{
private:
    std::vector<std::deque<Action>> agent_plans;
public:

    DummyPlanner(): MAPFPlanner() {};
    DummyPlanner(std::string fname): MAPFPlanner()
    {
        load_plans(fname);
    };
	~DummyPlanner(){}

    void load_plans(std::string fname);

    virtual void plan(int time_limit, std::vector<Action> & plan);

};
