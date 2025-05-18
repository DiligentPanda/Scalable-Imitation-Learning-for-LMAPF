cmake_minimum_required(VERSION 3.16)
project(lifelong)

set(CMAKE_CXX_STANDARD 17)

# Set the default value of PYTHON to false
option(PYTHON "Enable Python" OFF)


# Print the value of PYTHON
message(STATUS "PYTHON: ${PYTHON}")


if(GUIDANCE)
    message(STATUS "Guidance is enabled ")
    add_definitions(-DGUIDANCE)
endif()

if(RELAX)
    message(STATUS "RELAX is enabled: ${RELAX} traj/dist/heu table initialization per timestep")
    add_definitions(-DRELAX=${RELAX})
else()
    message(STATUS "RELAX is disabled default 100000 initialization per timestep")
    add_definitions(-DRELAX=100000)
endif()

if(FLOW_GUIDANCE)
    message(STATUS "FLOW_GUIDANCE is enabled, re init every: ${FLOW_GUIDANCE}")
    add_definitions(-DFLOW_GUIDANCE=${FLOW_GUIDANCE})
endif()

if(GUIDANCE_LNS)
    message(STATUS "Guidance LNS is enabled: ${GUIDANCE_LNS}")
    add_definitions(-DGUIDANCE_LNS=${GUIDANCE_LNS})
endif()

if(LNS_GROUP_SIZE)
    message(STATUS "LNS_GROUP_SIZE is set to: ${LNS_GROUP_SIZE}")
    add_definitions(-DLNS_GROUP_SIZE=${LNS_GROUP_SIZE})
else()
    message(STATUS "LNS_GROUP_SIZE is set to: 10")
    add_definitions(-DLNS_GROUP_SIZE=10)
endif()

if(LNS_EXPOUT)
    message(STATUS "Print lns exp data")
    add_definitions(-DLNS_EXPOUT)
endif()

if (LNS_DES_METHOD)
    message(STATUS "LNS_DES_METHOD is enabled: ${LNS_DES_METHOD}")
    add_definitions(-DLNS_DES_METHOD=${LNS_DES_METHOD})
endif()

if(INIT_PP)
    message(STATUS "INIT_PP is enabled")
    add_definitions(-DINIT_PP)
endif()

if(MAPFT)
    message(STATUS "MAPFT is enabled")
    add_definitions(-DMAPFT)
endif()

if(OBJECTIVE)
    message(STATUS "OBJECTIVE is enabled: ${OBJECTIVE}")
    add_definitions(-DOBJECTIVE=${OBJECTIVE})
else()
    message(STATUS "OBJECTIVE is default none")

    add_definitions(-DOBJECTIVE=0)
endif()


if(FOCAL_SEARCH)
    message(STATUS "FOCAL_SEARCH is enabled: ${FOCAL_SEARCH}")
    add_definitions(-DFOCAL_SEARCH=${FOCAL_SEARCH})
endif()

add_definitions(-DROBOT_RUNNERS)

add_definitions(-DNO_ROT)

IF(NOT CMAKE_BUILD_TYPE)
    # SET(CMAKE_BUILD_TYPE "DEBUG")
    SET(CMAKE_BUILD_TYPE "RELEASE")
    #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
    #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()

include_directories("inc")
include_directories("traffic_mapf")

# Find Boost
ADD_DEFINITIONS(-DBOOST_LOG_DYN_LINK)
#SET(Boost_USE_STATIC_LIBS ON)
find_package( Boost 1.49.0 REQUIRED COMPONENTS program_options system filesystem log log_setup)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

include_directories( ${Boost_INCLUDE_DIRS} )

add_subdirectory("pybind11")


IF(PYTHON)
    message(STATUS "Python support is enabled")
    find_package(Python3 3.7 REQUIRED COMPONENTS Interpreter Development)
    find_package(PythonInterp 3.7  REQUIRED)
    find_package(pybind11 REQUIRED)
    set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")


    pybind11_add_module(MAPF "./src/Grid.cpp"   "./src/States.cpp" "./python/MAPFbinding.cpp")
    target_link_libraries(MAPF PRIVATE ) 

    file(GLOB SOURCES "src/*.cpp")
    list(FILTER SOURCES EXCLUDE REGEX ".*src/driver\.cpp")

    file(GLOB PY_SOURCES "python/*.cpp")
    list(FILTER PY_SOURCES EXCLUDE REGEX ".*python/MAPFbinding\.cpp")

    add_executable(lifelong  ${PY_SOURCES} ${SOURCES})
    target_link_libraries(lifelong PRIVATE pybind11::embed ${Boost_LIBRARIES})

ELSE()
    message(STATUS "Python support is disabled")
    

    file(GLOB SOURCES "src/*.cpp", "src/LNS/*.cpp" "src/LNS/Parallel/*.cpp", "src/util/*.cpp")
    file(GLOB TRAFFIC_SOURCES "traffic_mapf/*.cpp")
    list(FILTER SOURCES EXCLUDE REGEX "(src/driver.cpp)|(src/PyShadowSystem.cpp)")

    pybind11_add_module(py_shadow_system ${SOURCES} ${TRAFFIC_SOURCES} "src/PyShadowSystem.cpp")
    target_link_libraries(py_shadow_system PRIVATE ${Boost_LIBRARIES} OpenMP::OpenMP_CXX)

    add_executable(lifelong ${SOURCES} ${TRAFFIC_SOURCES} "src/driver.cpp")
    # pybind11_add_module(lifelong ${SOURCES} ${TRAFFIC_SOURCES})
    target_link_libraries(lifelong PRIVATE ${Boost_LIBRARIES} OpenMP::OpenMP_CXX)
ENDIF()


add_custom_target(clean_all
    COMMAND ${CMAKE_BUILD_TOOL} clean
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/lifelong
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/build
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/*.so
)


set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES
    "${CMAKE_BINARY_DIR}/lifelong;
     ${CMAKE_BINARY_DIR}/build;
     ${CMAKE_BINARY_DIR}/*.so"
)



