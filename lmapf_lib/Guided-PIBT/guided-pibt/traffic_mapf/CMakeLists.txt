cmake_minimum_required(VERSION 3.16)
file(GLOB SRCS "./*.cpp")

project(traffic_mapf)





add_library(${PROJECT_NAME} STATIC ${SRCS})

if(GUIDANCE)
    message(STATUS "Guidance is enabled ")
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DGUIDANCE)
endif()

if(FLOW_GUIDANCE)
    message(STATUS "FLOW_GUIDANCE is enabled")
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DFLOW_GUIDANCE)
endif()

if(GUIDANCE_LNS)
    message(STATUS "Guidance LNS is enabled: ${GUIDANCE_LNS}")
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DGUIDANCE_LNS=${GUIDANCE_LNS})
endif()

if(LNS_GROUP_SIZE)
    message(STATUS "LNS_GROUP_SIZE is set to: ${LNS_GROUP_SIZE}")
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DLNS_GROUP_SIZE=${LNS_GROUP_SIZE})
else()
    message(STATUS "LNS_GROUP_SIZE is set to: 1")
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DLNS_GROUP_SIZE=10)
endif()

if(LNS_EXPOUT)
    message(STATUS "Print lns exp data")
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DLNS_EXPOUT)
endif()

if (LNS_DES_METHOD)
    message(STATUS "LNS_DES_METHOD is enabled: ${LNS_DES_METHOD}")
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DLNS_DES_METHOD=${LNS_DES_METHOD})
endif()

if(FLOW_PENALTY)
    message(STATUS "FLOW PENALTY is enabled")
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DFLOW_PENALTY)
endif()

if(INIT_PP)
    message(STATUS "INIT_PP is enabled")
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DINIT_PP)
endif()

if(GUID_T)
    message(STATUS "GUID_T is enabled: ${GUID_T}")
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DGUID_T=${GUID_T})
else()
    message(STATUS "GUID_T is default 30")
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DGUID_T=30)
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

target_compile_options(${PROJECT_NAME} PUBLIC -O3 -Wall)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})