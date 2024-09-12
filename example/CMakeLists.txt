add_executable(closestPair closestPair.cpp)
target_link_libraries(closestPair closestPairLib pargeoLib)

add_executable(graphGenerator graphGenerator.cpp)
target_link_libraries(graphGenerator
  pargeoLib
  delaunay2dLib
  emstLib
  spatialGraphLib)

set(EMST_INCLUDE_DIRECTORIES 
  /usr/include/eigen3
  /scratch1/rkwitt/spanning_trees_code/external/cnpy-install/include)
add_executable(emstLength emstLength.cpp)
target_include_directories(emstLength PRIVATE ${EMST_INCLUDE_DIRECTORIES})
target_link_libraries(emstLength
  pargeoLib
  delaunay2dLib
  emstLib
  spatialGraphLib
  boost_program_options
  sqlite3 
  /scratch1/rkwitt/spanning_trees_code/external/cnpy-install/lib/libcnpy.so)

add_executable(kNearestNeighbor kNearestNeighbor.cpp)
target_link_libraries(kNearestNeighbor pargeoLib)

add_executable(bichromaticClosestPair bichromaticClosestPair.cpp)
target_link_libraries(bichromaticClosestPair pargeoLib)

add_executable(rangeSearch rangeSearch.cpp)
target_link_libraries(rangeSearch pargeoLib)

add_executable(dynamicKdTreeKNN dynamicKdTreeKNN.cpp)
target_link_libraries(dynamicKdTreeKNN pargeoLib)

add_executable(batchKdTree batchKdTree.cpp)
target_link_libraries(batchKdTree pargeoLib)

add_executable(wellSeparatedPair wellSeparatedPair.cpp)
target_link_libraries(wellSeparatedPair pargeoLib)

add_executable(mortonSort mortonSort.cpp)
target_link_libraries(mortonSort pargeoLib)

add_executable(uniformDataGenerator uniformDataGenerator.cpp)
target_link_libraries(uniformDataGenerator pargeoLib)

add_executable(seedSpreader seedSpreaderGenerator.cpp)
target_link_libraries(seedSpreader
  datasetLib
  pargeoLib)

add_executable(convexHull convexHull.cpp)
target_link_libraries(convexHull pargeoLib
  divideHull2dLib
  pseudoHull3dLib
  )

add_executable(smallestEnclosingBall smallestEnclosingBall.cpp)
target_link_libraries(smallestEnclosingBall pargeoLib
  samplingSebLib
  # welzlSebLib
  # scanSebLib
  )
