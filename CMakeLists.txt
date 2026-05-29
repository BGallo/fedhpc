cmake_minimum_required(VERSION 3.21)
project(fedhpc_ext LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)

python_add_library(_moea MODULE src/fedhpc/_ext/moea.cpp WITH_SOABI)
target_link_libraries(_moea PRIVATE pybind11::headers)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_moea PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_features(_moea PRIVATE cxx_std_17)

install(TARGETS _moea DESTINATION fedhpc)
