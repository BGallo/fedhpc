cmake_minimum_required(VERSION 3.15)
project(fedhpc_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

find_package(OpenMP)

pybind11_add_module(_moea src/fedhpc/_ext/moea.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_moea PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS _moea DESTINATION fedhpc)
