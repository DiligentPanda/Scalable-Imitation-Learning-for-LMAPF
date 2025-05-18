#include "util/SearchForHeuristics/DataStructure.h"
#include <vector>
#include <algorithm>
#include <random>

using RIVERS::SPATIAL::OpenList;
using RIVERS::SPATIAL::State;

int main() {

    OpenList openlist(10);

    std::vector<State *> states;
    for (int i=0;i<10;++i) {
        State * s = new State(-1, -1, i, 0, NULL);
        states.push_back(s);
    }

    std::shuffle(states.begin(), states.end(), std::default_random_engine(120));

    for (auto & s : states) {
        openlist.push(s);
    }

    printf("the size is %d\n", openlist.size);
    printf("the last element's f value is %d\n", openlist.heap[openlist.size-1]->f);
    openlist.heap[openlist.size-1]->g = 3;
    openlist.heap[openlist.size-1]->f = 3;
    openlist.increase(openlist.heap[openlist.size-1]);

    printf("the last element's f value is %d\n", openlist.heap[openlist.size-1]->f);
    openlist.heap[openlist.size-1]->g = 2;
    openlist.heap[openlist.size-1]->f = 2;
    openlist.increase(openlist.heap[openlist.size-1]);

    printf("the last element's f value is %d\n", openlist.heap[openlist.size-1]->f);
    openlist.heap[openlist.size-1]->g = 1;
    openlist.heap[openlist.size-1]->f = 1;
    openlist.increase(openlist.heap[openlist.size-1]);


    while (!openlist.empty()) {
        State * s = openlist.pop();

        printf("%d\n", s->f);
        printf("remained size: %d\n", openlist.size);

        if (openlist.size==6) {
            printf("the [3] element's f value is %d\n", openlist.heap[openlist.size-1]->f);
            openlist.heap[3]->g = 4;
            openlist.heap[3]->f = 4;
            openlist.increase(openlist.heap[3]);
        }
    }    

    return 0;
}