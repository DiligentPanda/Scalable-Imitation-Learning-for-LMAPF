#include "util/analyze.h"
#include <string>

int main(int argc, char ** argv){

    if (argc!=5) {
        std::cerr<<"Usage: "<<argv[0]<<" <result.json> <height> <width> <analysis.json>"<<std::endl;
        exit(1);
    }

    std::string path=argv[1];
    int h = atoi(argv[2]);
    int w = atoi(argv[3]);
    std::string out_path=argv[4];

    std::ifstream fin(path);
    nlohmann::json result;
    try {
        result = nlohmann::json::parse(fin);
    } catch (nlohmann::json::parse_error & error) {
        std::cerr << "Failed to load " << path << std::endl;
        std::cerr << "Message: " << error.what() << std::endl;
        exit(1);
    }

    auto analysis=analyze_result_json(result, h, w);

    std::ofstream f(out_path,std::ios_base::trunc |std::ios_base::out);
    f << std::setw(4) << analysis;


    // std::cout<<analysis["throughput"]<<std::endl;
    // std::cout<<analysis.dump(4)<<std::endl;

    return 0;
}