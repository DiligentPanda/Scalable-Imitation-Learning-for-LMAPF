#pragma once
#include "common.h"
#include "nlohmann/json.hpp"

nlohmann::json read_conditional_value(const nlohmann::json & config, const string & key, int n_agents) {
    nlohmann::json values=config.at(key);
    for (auto & value: values) {
        if (value.at("n_agents")>=n_agents) {
            return value.at("value");
        }
    }
    std::cerr<<"n_agents "<<n_agents<<" satisfies no condition. please check!"<<endl;
    exit(-1);
}