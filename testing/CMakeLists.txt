#add_subdirectory(glfw)
#add_subdirectory(learning)
#add_subdirectory(test_particles)

# the x3d scene graph, need to rework this to use magnum 
#add_subdirectory(scenegraph)

#add_subdirectory(voronoi)
#add_subdirectory(blob1)
#add_subdirectory(particles2)
#add_subdirectory(test_magnum)
#add_subdirectory(test1)
#add_subdirectory(CylinderComponent)
#add_subdirectory(qtest)
#add_subdirectory(twave)
#add_subdirectory(fluid-test3)
#add_subdirectory(octree)

if(MX_APPLE)
# add_subdirectory(MeshTest)
#  add_subdirectory(CylinderTest)
# add_subdirectory(CylinderCocoa)
endif()

add_subdirectory(WindowlessTestApp)
#add_subdirectory(ApplicationTest)
