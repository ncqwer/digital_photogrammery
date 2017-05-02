cmake_minimum_required(VERSION 2.6)
project("digtal_photo_photogrammery")

# include_directories(/usr/local/include/opencv2)
# include_directories(/usr/local/include)

SET(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -pthread -std=c++11 -O0 -g -ggdb")

set(OpenCV_DIR1 /usr/local/alternative/OpenCV/OpenCV2.0)
message(${OpenCV_DIR1})
# Find OpenCV, you may need to set OpenCV_DIR {variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED
			 PATHS ${OpenCV_DIR1})
# include(${OpenCV_DIR1}/share/OpenCV/OpenCVConfig.cmake)
message(${OpenCV_DIR})
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    install path: ${OpenCV_INSTALL_PATH}")
message(STATUS "    config path: ${OpenCV_CONFIG_PATH}")
message(STATUS "    cmake current path: ${CMAKE_CURRENT_LIST_DIR}")
message(STATUS "    root_dir:${OpenCV_DIR1}")


# Add OpenCV headers location to your include paths
include_directories(${OpenCV_INCLUDE_DIRS})

# Add Eigen headers location to your include paths
include_directories(/usr/local/include/eigen3)
enable_testing()
add_subdirectory(test)
add_subdirectory(src lib)
add_subdirectory(work)

install(FILES README DESTINATION doc)
install(DIRECTORY res/ DESTINATION res)