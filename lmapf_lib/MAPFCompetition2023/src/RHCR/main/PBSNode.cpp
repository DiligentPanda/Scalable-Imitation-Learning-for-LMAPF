#include "RHCR/main/PBSNode.h"
#include <iostream>

namespace RHCR {

void PBSNode::clear()
{
    conflicts.clear();
    priorities.clear();
}


void PBSNode::print_priorities() const
{
    cout << "Priorities: ";
    for (auto row : priorities.G)
    {
        cout << row.first << " < (";
        for (auto a : row.second)
            std::cout << a << ", ";
        cout << "); ";
    }
    cout << endl;
}
}
