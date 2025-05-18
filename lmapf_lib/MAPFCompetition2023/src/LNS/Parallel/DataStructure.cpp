#include "LNS/Parallel/DataStructure.h"

namespace LNS {

namespace Parallel {

std::ostream & operator << (std::ostream &out, const PathEntry &pe)
{
    out << pe.location << "," << pe.orientation;
    return out;
}

std::ostream & operator << (std::ostream &out, const Path &path)
{
    for (auto & pe : path.nodes)
    {
        out << "(" << pe.location << "," << pe.orientation << ")->";
    }
    out << std::endl;
    return out;
}

}

}