cmake_minimum_required( VERSION 2.8 )
project( g2o_test )

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -Wall -O2 -march=native" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( G2O REQUIRED )
find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )
find_package( Cholmod )

include_directories( ${EIGEN3_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR} )

add_executable( g2o_test g2o_test.cpp )
target_link_libraries( g2o_test 
    ${OpenCV_LIBS}
    g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension g2o_types_sba
    ${CHOLMOD_LIBRARIES}
    )
