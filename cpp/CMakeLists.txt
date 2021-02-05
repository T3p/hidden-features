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

# For CMake < 3.9, we need to make the target ourselves
if(NOT TARGET OpenMP::OpenMP_CXX)
    find_package(Threads REQUIRED)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX
                 PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
    # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
    set_property(TARGET OpenMP::OpenMP_CXX
                 PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)

endif()

add_subdirectory(cnpy)
include_directories(${CMAKE_SOURCE_DIR}/cnpy/)

set(BINF simple_continuous.cpp simple_mab.cpp simple_linucb.cpp simple_leader.cpp multirep.cpp test_corral.cpp hls.cpp vardim.cpp mixing.cpp leader_jester.cpp oful_jester.cpp all_jester.cpp)
foreach(name ${BINF})
    get_filename_component(base_name ${name} NAME_WE)
    message(STATUS "bulding ${base_name}")
    add_executable(${base_name} ${name})
    target_link_libraries(${base_name} cnpy)
    if(OpenMP_CXX_FOUND)
        target_link_libraries (${base_name} Eigen3::Eigen OpenMP::OpenMP_CXX nlohmann_json::nlohmann_json ${Boost_LIBRARIES})
    else()
        target_link_libraries (${base_name} Eigen3::Eigen nlohmann_json::nlohmann_json ${Boost_LIBRARIES})
    endif()
endforeach()


