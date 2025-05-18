#include "util/Analyzer.h"

void Analyzer::snapshot(
    const string & fp_prefix,
    int timestep,
    const std::vector<::Path> & paths
) {

    string fp=fp_prefix+"_"+std::to_string(timestep)+".ppath";

    auto path=boost::filesystem::path(fp);
    auto parent_path=path.parent_path();
    if (!boost::filesystem::exists(parent_path)) {
        boost::filesystem::create_directories(parent_path);
    }

    int height=data.at("height").get<int>();
    int width=data.at("width").get<int>();

    std::ofstream out(fp);

    out<<data.at("instance").get<string>()<<endl;
    out<<height<<" "<<width<<endl;
    
    for (int a=0;a<paths.size();++a) {
        auto & path = paths[a];
        out<<a<<endl;
        for (int t=timestep;t<path.size();++t){
            out<<path[t].location/width<<","<<path[t].location%width;
            if (t!=path.size()-1) {
                out<<" ";
            } else {
                out<<endl;
            }
        }
    }

}

Analyzer analyzer;