#include "common.h"
#include "nlohmann/json.hpp"

int get_orient_idx(string o) {
    if (o=="E") return 0;
    if (o=="S") return 1;
    if (o=="W") return 2;
    if (o=="N") return 3;
    std::cerr<<"unknown orientation: "<<o<<std::endl;
    exit(-1);
}

void move(int & x,int & y,int & o, char action) {
    if (action=='F') {
        if (o==0) {
            x+=1;
        } else if (o==1) {
            y+=1;
        } else if (o==2) {
            x-=1;
        } else if (o==3) {
            y-=1;
        } else {
            std::cerr<<"unknown orientaiton: "<<o<<std::endl;
            exit(-1);
        }
    } else if (action=='R') {
        o=(o+1)%4;
    } else if (action=='C') {
        o=(o+3)%4;
    } else if (action=='W') {
        // do nothing
    } else {
        std::cerr<<"unknown action: "<<action<<std::endl;
        exit(-1);
    }

}

nlohmann::json analyze_result_json(const nlohmann::json & result, int h, int w) {
    // objective: throughput
    double throughput=result["numTaskFinished"].get<double>();
    std::cerr<<"throughput: "<<throughput<<std::endl;

    // statistics:
    // 1. vertex usage
    // 2. edge usage
    int map_size=h*w;
    std::vector<double> vertex_usage(map_size,0);
    std::vector<std::vector<double> > edge_usage(map_size, std::vector<double>(map_size,0));

    int team_size=result["teamSize"].get<int>();
    std::cerr<<"team size: "<<team_size<<std::endl;

    auto actions_str = result["actualPaths"][0].get<std::string>();
    int T=actions_str.size()/2+1;
    std::cerr<<"T: "<<T<<std::endl;

    for (int aid=0;aid<team_size;++aid) {
        auto actions_str = result["actualPaths"][aid].get<std::string>();
        // std::cerr<<"agent "<<aid<<" actions: "<<actions_str<<std::endl;

        auto start_y = result["start"][aid][0].get<int>();
        auto start_x = result["start"][aid][1].get<int>();
        auto start_orient = get_orient_idx(result["start"][aid][2].get<std::string>());
        // std::cerr<<"agent "<<aid<<" start: "<<start_y<<","<<start_x<<","<<start_orient<<std::endl;

        auto pos=start_y*w+start_x;
        // update vertex usage
        vertex_usage[pos]+=1;

        // simulate
        auto prev_x=start_x;
        auto prev_y=start_y;
        auto prev_orient=start_orient;
        for (int i=0;i<actions_str.size();i=i+2) {
            // std::cerr<<"iter "<<i<<std::endl;
            char action=actions_str[i];
            auto curr_x=prev_x;
            auto curr_y=prev_y;
            auto curr_orient=prev_orient;
            move(curr_x,curr_y,curr_orient,action);

            if (curr_x<0 || curr_x>=w || curr_y<0 || curr_y>=h) {
                std::cerr<<"agent "<<aid<<" out of bound"<<std::endl;
                exit(-1);
            }

            auto prev_pos=prev_y*w+prev_x;
            auto curr_pos=curr_y*w+curr_x;
            // update vertex usage
            vertex_usage[curr_pos]+=1;
            // update edge usage
            edge_usage[prev_pos][curr_pos]+=1;

            prev_x=curr_x;
            prev_y=curr_y;
            prev_orient=curr_orient;
        }
    }

    nlohmann::json analysis;
    analysis = {
        {"throughput", throughput},
        {"vertexUsage", vertex_usage},
        {"edgeUsage", edge_usage}
    };
    return analysis;
}