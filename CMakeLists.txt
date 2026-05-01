cmake_minimum_required(VERSION 3.21)
 
project(BurnoutML
    VERSION 1.0
    LANGUAGES CXX
)
 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

find_package(OpenMP REQUIRED)

add_executable(main
    src/main.cpp
    src/sequential_trainer.cpp
    src/data_loader.cpp
    src/parallel_trainer.cpp
    src/strategy/stress_test_strategy.cpp
    src/strategy/prediction_strategy.cpp
)

target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)

# GOOGLE TEST:
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(runTests tests/test_main.cpp
    src/sequential_trainer.cpp
    src/parallel_trainer.cpp
    src/data_loader.cpp
)

target_link_libraries(runTests PRIVATE
    OpenMP::OpenMP_CXX
    gtest_main
)

target_include_directories(runTests PRIVATE include)

add_test(NAME runTests COMMAND runTests)