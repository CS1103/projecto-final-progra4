cmake_minimum_required(VERSION 3.10.2)
project(prog3_nn_final_project_v2025_01)

set(CMAKE_CXX_STANDARD 20)
if(UNIX AND NOT APPLE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()



if(MINGW OR CYGWIN)
    add_definitions(-O3)
endif()

include_directories(./src)
file(GLOB source_files
        main.cpp
        src/*.cpp)

add_executable(${PROJECT_NAME}
        ${source_files}
    )

#if(UNIX AND NOT APPLE)
#    find_package(TBB)
#    target_link_libraries(${PROJECT_NAME} TBB::tbb)
#endif()

#add_subdirectory(autograder)

