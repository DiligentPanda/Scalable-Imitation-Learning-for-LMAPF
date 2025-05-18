#pragma once
#include <iostream>
#include <random>
#include <cstring>

namespace SUO2 {

namespace Spatial {

class CostMap {
public:
    int max_x;
    int max_y;

    float * data;

    CostMap(int max_x, int max_y): max_x(max_x), max_y(max_y) {
        data = new float[max_x*max_y];
        clear();
    }

    ~CostMap() {
        delete [] data;
    }

    inline float & operator[](int index) {
        return data[index];
    }

    void clear() {
        memset(data, 0, sizeof(float)*max_x*max_y);
    }

    void copy(const CostMap & other) {
        // TODO(rivers): this is a large array, should not be copied often
        if (max_x!=other.max_x || max_y!=other.max_y) {
            std::cerr<<"Error: copying cost map with different shape"<<std::endl;
            exit(-1);
        }

        memcpy(data, other.data, sizeof(float)*max_x*max_y);
    }

    void update(const std::vector<std::pair<int,float> > & deltas) {
        for (const auto & delta: deltas) {
            int index = delta.first;
            float cost = delta.second;
            data[index] += cost;
        }
    }

};






}

}