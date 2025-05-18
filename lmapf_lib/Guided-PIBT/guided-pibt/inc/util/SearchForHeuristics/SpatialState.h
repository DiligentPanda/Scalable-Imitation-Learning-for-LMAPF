#pragma once
#include <cstdlib>
#include <utility>
#include "boost/heap/pairing_heap.hpp"

namespace RIVERS {

namespace SPATIAL {

struct State {
    int pos;
    int orient;
    float g;
    float h;
    float f;
    State * prev;

    State(): pos(-1), orient(-1), g(-1), h(0), f(-1), prev(nullptr), closed(false) {};
    // State(int pos, int g, int h, State * prev): pos(pos), orient(-1), g(g), h(h), f(g+h), prev(prev), closed(false) {};
    State(int pos, int orient, float g, float h, State * prev): pos(pos), orient(orient), g(g), h(h), f(g+h), prev(prev), closed(false) {};

    void copy(const State * s) {
        pos = s->pos;
        orient = s->orient;
        g = s->g;
        h = s->h;
        f = s->f;
        prev = s->prev;
    }

    struct StateCompare {
        bool operator()(const State * s1, const State * s2) const {
            if (s1->f == s2->f){
                // if (s1->h == s2->h) {
                //     return rand()%2==0;
                // }
                return s1->h > s2->h;
            }
            return s1->f > s2->f;
        }
    };

    struct StateHash {
        std::size_t operator()(const State * s) const {
            size_t loc_hash = std::hash<int>()(s->pos*4+s->orient);
            return loc_hash;
        }
    };

    struct StateEqual {
        bool operator()(const State * s1, const State * s2) const {
            return s1->pos == s2->pos && s1->orient==s2->orient;
        }
    };

    bool closed;
    boost::heap::pairing_heap<State*, boost::heap::compare<State::StateCompare> >::handle_type open_list_handle;
    int heap_index;

};

}

}