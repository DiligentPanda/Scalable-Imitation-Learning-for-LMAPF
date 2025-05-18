#pragma once
#include "common.h"
#include "nlohmann/json.hpp"

int get_orient_idx(string o);
void move(int & x,int & y,int & o, char action);
nlohmann::json analyze_result_json(const nlohmann::json & result, int h, int w);