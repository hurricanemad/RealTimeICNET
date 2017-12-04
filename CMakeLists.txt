cmake_minimum_required(VERSION 2.6)
add_definitions(-std=c++11)
project(videocapture)


add_subdirectory(src bin)
add_subdirectory(lib lib)

install(DIRECTORY model prototxt lib video DESTINATION install)

