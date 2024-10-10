cmake_minimum_required(VERSION 3.25)

project(cpp_under_sea LANGUAGES CXX)

# compile with ISO C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)


# find DPC++ compiler
find_package(IntelDPCPP CONFIG REQUIRED)


set(SOURCES src/main.cpp)

# Add the executable
add_executable(cpp_under_sea ${SOURCES})
target_compile_options(cpp_under_sea PRIVATE -O3)




