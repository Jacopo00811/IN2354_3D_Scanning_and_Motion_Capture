cmake_minimum_required(VERSION 3.10)

# Project name
project(exercise_0)

# Set some build variables which appear in the Cmake GUI (you can write the build logic with these options)
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs CACHE PATH "Path to lib folder")
set(MAGIC_TOGGLE True CACHE BOOL "Some toggable")

# Set C++ standards
set(CMAKE_CXX_STANDARD 14)

# This message will be display during the configuration of the project
if(${MAGIC_TOGGLE})
    message(STATUS "Magic Toggle is on.")
else()
    message(STATUS "Magic Toggle is off.")
endif()


# LIBRARIES

# Let's have a look how the libraries are set up
# First, create a list variable with all the files used to compile the library
# Then, create a build target named "lib" that will create the library from the the files in the list
# ${variable} is used to access the value of the variable
LIST(APPEND LIB_FILES "lib/algo.h" "lib/algo.cpp")
add_library(lib STATIC ${LIB_FILES})


# EXECUTABLES

# Create the list variables HEADER_FILES and SOURCE_FILES, assign them the following values
LIST(APPEND HEADER_FILES "util.h" "constants.h")
LIST(APPEND SOURCE_FILES "main.cpp")


# Add a path to the installed Eigen library for find_package command https://cmake.org/cmake/help/latest/command/find_package.html
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to the folder holding Eigen3Config.cmake file in the Eigen installation folder")
# finds and loads settings from the installed Eigen package
find_package(Eigen3 REQUIRED)


# Create a build target with a name "app" that will create the executable from the the files in the list
add_executable(app ${SOURCE_FILES} ${HEADER_FILES})

# Add the lib folder to the list of the include directories (directories where the header files are searched)
# So that you can #include the necessary symbols in your project
target_include_directories(app PUBLIC "lib")  

# Link Eigen and "lib" library to our executable
target_link_libraries(app Eigen3::Eigen lib)
