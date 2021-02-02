cmake_minimum_required (VERSION 3.0)
project (myproject)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MAKE_BUILD_TYPE "Release")

find_package (Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(Boost 1.36.0 COMPONENTS iostreams)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OPENMP_INCLUDES})
link_directories(${OPENMP_LIBRARIES})

set(BINF simple.cpp simple_mab.cpp multirep.cpp test_corral.cpp hls.cpp vardim.cpp mixing.cpp)
foreach(name ${BINF})
    get_filename_component(base_name ${name} NAME_WE)
    message(STATUS "bulding ${base_name}")
    add_executable(${base_name} ${name})
    if(OpenMP_CXX_FOUND)
        target_link_libraries (${base_name} Eigen3::Eigen OpenMP::OpenMP_CXX nlohmann_json::nlohmann_json ${Boost_LIBRARIES})
    else()
        target_link_libraries (${base_name} Eigen3::Eigen OpenMP::OpenMP_CXX nlohmann_json::nlohmann_json ${Boost_LIBRARIES})
    endif()
endforeach()


