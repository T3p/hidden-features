# Required libraries
Eigen (http://eigen.tuxfamily.org/)
Boost (https://www.boost.org/)
nlohmann_json (https://github.com/nlohmann/json)
### Optional
OpenMP

## For Ubuntu
libeigen3-dev
libboost-all-dev
libomp-dev
nlohmann-json-dev


## FOR MAC
- install Eigen (http://eigen.tuxfamily.org/index.php?title=Main_Page)
Remember to set -DCMAKE_INSTALL_PREFIX=
    After installation, update the CMakeLists.txt with the install path and eigen
    You can also install eigen using homebrew
- install nlohmann_json using homebrew
- install boost using homebrew
- install libomp using homebrew

# Compilation

mkdir build
cd build
cmake ../
make


