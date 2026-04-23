cmake_minimum_required(VERSION 3.21)
 
project(BurnoutML
    VERSION 1.0
    LANGUAGES CXX
)
 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(main
    src/main.cpp
    src/sequential_trainer.cpp
    src/data_loader.cpp
)
