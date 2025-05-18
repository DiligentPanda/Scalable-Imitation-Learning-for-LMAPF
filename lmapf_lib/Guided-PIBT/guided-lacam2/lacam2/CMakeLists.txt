cmake_minimum_required(VERSION 3.16)
file(GLOB SRCS "./src/*.cpp")
project(lacam2)
add_library(${PROJECT_NAME} STATIC ${SRCS})
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../guided-pibt/traffic_mapf ${CMAKE_CURRENT_BINARY_DIR}/traffic_mapf)


target_compile_options(${PROJECT_NAME} PUBLIC -O3 -Wall)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME} INTERFACE ./include)
target_link_libraries(${PROJECT_NAME} PUBLIC traffic_mapf)
