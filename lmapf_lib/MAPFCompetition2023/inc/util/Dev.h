#pragma once

#ifdef DEV
    #define ONLYDEV(code) code 
#else
    #define ONLYDEV(code)
#endif