#pragma once
#include <cstdlib>
#include <utility>
#include "boost/heap/pairing_heap.hpp"

namespace LNS {

namespace Parallel {

class TimeSpaceAStarState {
public:
    // static const int MAX_STATE=100000;
    // static const int MAX_ORIENT=4;

    int pos;
    int orient;
    int t;
    float g;
    float h;
    float f;
    int num_of_conflicts = 0;
    bool arrived;
    TimeSpaceAStarState * prev;

    TimeSpaceAStarState(): pos(-1), orient(-1), t(-1), g(-1), h(0), f(-1), num_of_conflicts(0), prev(nullptr), closed(false), arrived(false) {};
    TimeSpaceAStarState(int pos, int orient, int t, float g, float h, int num_of_conflicts, bool arrived, TimeSpaceAStarState * prev):
        pos(pos), orient(orient), t(t), g(g), h(h), f(g+h), num_of_conflicts(num_of_conflicts), prev(prev), closed(false), arrived(arrived) {};

    void copy(const TimeSpaceAStarState * s) {
        pos = s->pos;
        orient = s->orient;
        t = s->t;
        g = s->g;
        h = s->h;
        f = s->f;
        num_of_conflicts = s->num_of_conflicts;
        arrived = s->arrived;
        prev = s->prev;
    }

    struct Compare {
        bool operator()(const TimeSpaceAStarState *s1, const TimeSpaceAStarState *s2) const {
            if (s1->num_of_conflicts == s2->num_of_conflicts) {
                if (s1->arrived==s2->arrived) {
                    if (s1->f == s2->f){
                        // if (s1->h == s2->h) {
                        //     return rand()%2==0;
                        // }
                        return s1->h > s2->h;
                        // not arrived is worse.
                    }
                    return s1->f > s2->f;
                }
                return s1->arrived < s2->arrived;
            }
            return s1->num_of_conflicts > s2->num_of_conflicts;
        }
    };
    
    struct Hash {
        std::size_t operator()(const TimeSpaceAStarState *s) const {
            // TODO(rivers): we may replace the hash here!
            size_t h = std::hash<int>()(s->t<<21+s->pos<<5+s->orient<<1+s->arrived);
            return h;        
        }
    };
    
    struct Equal {
        bool operator()(const TimeSpaceAStarState *s1, const TimeSpaceAStarState *s2) const {
            return s1->pos == s2->pos && s1->orient==s2->orient && s1->t==s2->t && s1->arrived==s2->arrived;
        }
    };

    bool closed=false;
    boost::heap::pairing_heap<TimeSpaceAStarState*, boost::heap::compare<TimeSpaceAStarState::Compare>>::handle_type open_list_handle;

};

}

}