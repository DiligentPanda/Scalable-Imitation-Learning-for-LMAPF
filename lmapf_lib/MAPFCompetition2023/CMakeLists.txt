cmake_minimum_required(VERSION 3.16)

project(lifelong)

set(CMAKE_CXX_STANDARD 17)


# set(MAP_OPT ON)

# if set, use dev mode, which has a lot of logging and timing.
# set(DEV ON)
# if set, use action model with no rotation.
set(NO_ROT ON)

# Set the default value of PYTHON to false
option(PYTHON "Enable Python" OFF)

if(PYTHON)
    set(PYTHON_FLAG 1)
else()
    set(PYTHON_FLAG 0)
endif()


# Print the value of PYTHON
message(STATUS "PYTHON: ${PYTHON}")

IF(NOT CMAKE_BUILD_TYPE)
    #SET(CMAKE_BUILD_TYPE "DEBUG")
    SET(CMAKE_BUILD_TYPE "RELEASE")
    #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
    #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()

include_directories("inc")

# Find Boost
ADD_DEFINITIONS(-DBOOST_LOG_DYN_LINK)
#SET(Boost_USE_STATIC_LIBS ON)
find_package( Boost 1.49.0 REQUIRED COMPONENTS program_options system filesystem log log_setup iostreams)

include_directories( ${Boost_INCLUDE_DIRS} )

find_package(OpenMP REQUIRED)
find_package(spdlog REQUIRED)
find_package(Threads REQUIRED)
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)


add_subdirectory(pybind11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion")

# IF(PYTHON)
#     include_directories("python")
#     message(STATUS "Python support is enabled")

#     find_package(pybind11 REQUIRED)
#     set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")


#     pybind11_add_module(MAPF "./src/Grid.cpp"   "./src/States.cpp" "./python/MAPFbinding.cpp")
#     target_link_libraries(MAPF PRIVATE ) 

#     file(GLOB SOURCES "src/*.cpp")

#     file(GLOB PY_SOURCES "python/*.cpp")
#     list(FILTER PY_SOURCES EXCLUDE REGEX ".*python/MAPFbinding\.cpp")

#     add_executable(lifelong  ${PY_SOURCES} ${SOURCES})
#     target_link_libraries(lifelong PRIVATE pybind11::embed ${Boost_LIBRARIES})
#     target_compile_definitions(lifelong PRIVATE PYTHON=${PYTHON_FLAG})

#     if (COPY_PY_PATH_CONFIG)
#         file(COPY "config.json" DESTINATION ${CMAKE_BINARY_DIR})
#     endif()


# ELSE()
# message(STATUS "Python support is disabled")

IF(NO_ROT)
    message(STATUS "Use action model with no rotation")
    add_definitions(-DNO_ROT)
    file(GLOB OFFICIAL_SOURCES "src/**.cpp")
    file(GLOB_RECURSE MY_SOURCES "src/LNS/**.cpp" "src/LaCAM2/**.cpp" "src/util/**.cpp")
ELSE()
    message(STATUS "Use action model with rotation")
    file(GLOB OFFICIAL_SOURCES "src/**.cpp")
    file(GLOB_RECURSE MY_SOURCES "src/LNS/**.cpp" "src/RHCR/**.cpp" "src/LaCAM2/**.cpp" "src/util/**.cpp")
ENDIF()

list(FILTER OFFICIAL_SOURCES EXCLUDE REGEX ".*analyze_results\.cpp")
list(FILTER OFFICIAL_SOURCES EXCLUDE REGEX ".*py_driver\.cpp")
set(SOURCES ${OFFICIAL_SOURCES} ${MY_SOURCES})

IF(DEV)
    message(STATUS "Dev mode is enabled")
    add_definitions(-DDEV)
ENDIF()

IF (MAP_OPT)
    message(STATUS "Use map optimization")
    add_definitions(-DMAP_OPT)
ENDIF()

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wfatal-errors")

add_executable(lifelong ${SOURCES} )
target_link_libraries(lifelong ${Boost_LIBRARIES})
target_link_libraries(lifelong OpenMP::OpenMP_CXX)
target_link_libraries(lifelong spdlog::spdlog)
target_compile_definitions(lifelong PRIVATE PYTHON=${PYTHON_FLAG})

list(FILTER SOURCES EXCLUDE REGEX ".*driver\.cpp")
add_executable(compute_heuristics ${SOURCES} "tools/compute_heuristics.cpp")
target_link_libraries(compute_heuristics ${Boost_LIBRARIES})
target_link_libraries(compute_heuristics OpenMP::OpenMP_CXX)
target_link_libraries(compute_heuristics spdlog::spdlog)

pybind11_add_module(py_compute_heuristics ${SOURCES} "tools/py_compute_heuristics.cpp")
target_link_libraries(py_compute_heuristics PRIVATE ${Boost_LIBRARIES})
target_link_libraries(py_compute_heuristics PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(py_compute_heuristics PRIVATE spdlog::spdlog)

pybind11_add_module(py_PIBT "tools/py_PIBT.cpp")

pybind11_add_module(py_STOP "tools/py_STOP.cpp")

pybind11_add_module(py_PLNS ${SOURCES} "tools/py_PLNS.cpp")
target_link_libraries(py_PLNS PRIVATE ${Boost_LIBRARIES})
target_link_libraries(py_PLNS PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(py_PLNS PRIVATE spdlog::spdlog)

IF (MAP_OPT)
    list(FILTER SOURCES EXCLUDE REGEX "src/driver\.cpp")
    pybind11_add_module(py_driver "src/py_driver.cpp" ${SOURCES})
    target_link_libraries(py_driver PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(py_driver PRIVATE spdlog::spdlog)
    target_link_libraries(py_driver PRIVATE ${Boost_LIBRARIES})
ENDIF()

# ENDIF()

# add_executable(test_log "test/my_logger.cpp" "src/util/MyLogger.cpp")
# target_link_libraries(test_log spdlog::spdlog)

# add_executable(test_openlist "test/my_openlist.cpp")
# add_executable(test_thread_pool "test/thread_pool.cpp")
# target_link_libraries(test_thread_pool PRIVATE Threads::Threads)

add_custom_target(clean_all
    COMMAND ${CMAKE_BUILD_TOOL} clean
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/lifelong
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/compute_heuristics
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/py_compute_heuristics
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/py_PIBT
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/build
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/*.so
)


set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES
    "${CMAKE_BINARY_DIR}/lifelong;
     ${CMAKE_BINARY_DIR}/compute_heuristics;
     ${CMAKE_BINARY_DIR}/py_compute_heuristics;
     ${CMAKE_BINARY_DIR}/py_PIBT;
     ${CMAKE_BINARY_DIR}/build;
     ${CMAKE_BINARY_DIR}/*.so"
)
