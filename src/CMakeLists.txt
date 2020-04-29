# LLVM compile options
#add_definitions(${LLVM_LDFLAGS})

add_compile_options(-DMX_STRICT)

add_subdirectory(mdcore)

# quiet linker warnings on Apple
# set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# Make sure the compiler can find include files for our Hello library
# when other libraries or executables link to Hello
include_directories(PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${GLM_INCLUDE_DIRS})

message(STATUS "numpy: ${NUMPY_INCLUDE_DIR}")


file(GLOB AST_SRC
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "ast/*.cpp"
  )

#file(GLOB X3D_SRC
#  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
#  "x3d/*.cpp"
#  )

# Resources for the MxMeshShaderProgram
corrade_add_resource(MxMeshShaderProgramRes MxMeshShaderProgram.res)

set(SRC
  MObject.cpp
  moduledef.cpp
  mx_callable.cpp
  mx_float.cpp
  mx_import.cpp
  mx_port.cpp
  mx_abstract.cpp
  mx_eval.cpp
  mx_floatobject.cpp
  mx_int.cpp
  mx_number.cpp
  mx_runtime.cpp
  mx_ast.cpp
  mx_function.cpp
  MxList.cpp
  mx_stmt.cpp
  mx_tupleobject.cpp
  MxMeshCore.cpp
  MxMesh.cpp
  MxPolygon.cpp
  MxModule.cpp
  MxSimulator.cpp
  MxTestCube.cpp
  MxTestParticles.cpp
  MxPropagator.cpp
  MxTestMDCoreParticles.cpp
  MxModule.cpp
  MxModel.cpp
  MxSystem.cpp
  MxPropagator.cpp
  MxCompositeParticle.cpp
  MxController.cpp
  MxView.cpp
  MxLattice.cpp
  MxLatticeCluster.cpp
  MxCell.cpp
  MeshPropagator.cpp
  MxMeshPottsPropagator.cpp
  MxExecutionContext.cpp
  MxForceFunction.cpp
  LangevinPropagator.cpp
  ScalarField.cpp
  MeshRelationships.cpp
  MeshOperations.cpp
  stochastic_rk.cpp
  MxEdge.cpp
  MeshIO.cpp
  MxTesselator.cpp
  MxColumnarCell.cpp
  FlipEdge.cpp
  CollapsePolygon.cpp
  SplitPolygon.cpp
  SplitCell.cpp
  SplitEdge.cpp
  FlipEdgeToPolygon.cpp
  FlipPolygonToEdge.cpp
  MxConstraints.cpp
  MxForces.cpp
  MxPolygonSurfaceTensionForce.cpp
  MxPolygonAreaConstraint.cpp
  MxCellVolumeConstraint.cpp
  MxSurfaceSimulator.cpp
  MxCylinderModel.cpp
  MxUniverse.cpp

  MxPyTest.cpp

  rendering/ArcBallInteractor.cpp
  rendering/MxGlfwApplication.cpp
  rendering/MxWindowlessApplication.cpp
  rendering/MxApplication.cpp
  rendering/MxWindowView.cpp
  rendering/MxWindow.cpp
  rendering/MxWindowProxy.cpp
  rendering/MxWindowHost.cpp
  rendering/MxGlfwWindow.cpp
  rendering/MxRenderer.cpp
  rendering/MxUniverseRendererOld.cpp
  rendering/MxUniverseRenderer.cpp

  rendering/MxMeshRenderer.cpp
  rendering/MxMeshShaderProgram.cpp
  rendering/MxRenderTarget.cpp
  rendering/MxImageView.cpp
  rendering/MxUI.cpp
  rendering/MxTestView.cpp
  rendering/MxImageConverters.cpp

  shaders/ParticleSphereShader.cpp

  ${AST_SRC}
#  ${X3D_SRC}
  )


set(HEADERS
  ${MECHANICA_PUBLIC_HEADERS}
  MxDebug.h
  mechanica_llvm.h
  mechanica_private.h
  MxMeshCore.h
  MxMesh.h
  MxPolygon.h
  MxModule.h
  MxSimulator.h
  MxTestCube.h
  MxTestParticles.h
  MxTestMDCoreParticles.h
  MxModule.h
  MxModel.h
  MxSystem.h
  MxPropagator.h

  MxCompositeParticle.h
  MxController.h
  MxView.h
  MxLattice.h
  MxLatticeCluster.h
  MxCell.h

  MeshPropagator.h
  MxMeshPottsPropagator.h
  MxExecutionContext.h
  MxForceFunction.h
  LangevinPropagator.h
  ScalarField.h
  MeshRelationships.h
  MeshOperations.h
  stochastic_rk.h
  MxEdge.h
  MeshIO.h
  MxTesselator.h
  MxColumnarCell.h
  SymmetricEigen.h
  GteSymmetricEigensolver3x3.h
  MxConstraints.h
  MxForces.h
  MxPolygonSurfaceTensionForce.h
  MxPolygonAreaConstraint.h
  MxCellVolumeConstraint.h
  MxSurfaceSimulator.h
  MxCylinderModel.h
  MxUniverse.h

  MxPyTest.h

  rendering/ArcBallInteractor.h
  rendering/MxGlfwApplication.h
  rendering/MxWindowlessApplication.h
  rendering/MxWindowless.h
  rendering/MxApplication.h
  rendering/MxWindowView.h
  rendering/MxWindow.h
  rendering/MxWindowProxy.h
  rendering/MxWindowHost.h
  rendering/MxGlfwWindow.h
  rendering/MxRenderer.h
  rendering/MxUniverseRendererOld.h
  rendering/MxUniverseRenderer.h
  rendering/WireframeObjects.h
  rendering/FlatShadeObject.h
  rendering/MxMeshRenderer.h
  rendering/MxMeshShaderProgram.h
  rendering/MxRenderTarget.h
  rendering/MxImageView.h
  rendering/MxUI.h
  rendering/MxTestView.h
  rendering/MxImageConverters.h

  shaders/ParticleSphereShader.h
  )




# For development, for the time being, we won't build the object library
# because it takes longer to build if we're only building one library,
# and won't build the py version for a while.

# build all the source as a CMake 'OBJECT' lib, this gets
# linked into the real libraries below.
#add_library(mechanica_OBJECTS OBJECT ${SRC} ${HEADERS})

# get the include directories of the dependencies that the mechanica objects use
# really goofy design decision in design of CMake OBJECT libraries -- no transitive includes
#target_include_directories(mechanica_OBJECTS PRIVATE $<TARGET_PROPERTY:mdcore_single,INTERFACE_INCLUDE_DIRECTORIES>)
#target_include_directories(mechanica_OBJECTS PRIVATE $<TARGET_PROPERTY:voro++,INTERFACE_INCLUDE_DIRECTORIES> )
#target_include_directories(mechanica_OBJECTS PRIVATE $<TARGET_PROPERTY:Magnum::Magnum,INTERFACE_INCLUDE_DIRECTORIES>)
#target_include_directories(mechanica_OBJECTS PRIVATE "${PROJECT_SOURCE_DIR}/include")
#target_include_directories(mechanica_OBJECTS PRIVATE ${NUMPY_INCLUDE_DIR})



# this is the "object library" target: compiles the sources only once
# need at least one src file in a project for xcode to build it, xcode
# won't work if any product *only* has an object src
add_library(mechanica_obj
  OBJECT
  ${SRC}
  ${PROJECT_BINARY_DIR}/include/mx_config.h
  ${HEADERS}
  ${MxMeshShaderProgramRes}
  )

message("Shader Program Res: ${MxMeshShaderProgramRes}")


target_include_directories(mechanica_obj PRIVATE ${PYBIND11_INCLUDE_DIR})

# set_target_properties(mechanica PROPERTIES POSITION_INDEPENDENT_CODE ON)

# temporarily add add these includes until we get a stable public API
target_include_directories(mechanica_obj PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  #${PYTHON_INCLUDE_DIRS}
  #${NUMPY_INCLUDE_DIR}
  #$<INSTALL_INTERFACE:include/mylib>  # <prefix>/include/mylib
)


message("Carbon Includes:")
get_target_property(dirs Carbon::Static INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message("carbon include='${dir}'")
endforeach()


message("Carbon Interface Includes:")

get_target_property(dirs Carbon::Static INTERFACE_INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message( "carbon interface include='${dir}'")
endforeach()




target_link_libraries(mechanica_obj
  Carbon::Static
  DirectXMath
  Python::Python
  Python::NumPy
  glfw
  Assimp::Assimp
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders
  JPEG::JPEG
  voro++
  sundials_cvode_static
  ${OPENGL_LIBRARIES}
  mdcore_single
  )


source_group("Public Header Files" FILES ${MECHANICA_PUBLIC_HEADERS})
source_group("Rendering" REGULAR_EXPRESSION "rendering/.*")
source_group("Shaders" REGULAR_EXPRESSION "shaders/.*")


#message("MECHANICA_PUBLIC_HEADERS: ${MECHANICA_PUBLIC_HEADERS}")

#message("MECHANICA_SRC: ${SRC}")


# Link against LLVM libraries
#link_directories(${LLVM_LIBRARY_DIRS})
#target_link_libraries(mechanica ${LLVM_LIBRARIES})


# shared libraries need PIC
set_property(TARGET mechanica_obj PROPERTY POSITION_INDEPENDENT_CODE 1)


# shared and static libraries built from the same object files
add_library(mechanica_shared SHARED
  $<TARGET_OBJECTS:mechanica_obj>
  mechanica.cpp
  )

set_target_properties(mechanica_shared PROPERTIES OUTPUT_NAME mechanica)

target_include_directories(mechanica_shared PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  #${PYTHON_INCLUDE_DIRS}
  #${NUMPY_INCLUDE_DIR}
  #$<INSTALL_INTERFACE:include/mylib>  # <prefix>/include/mylib
)

target_include_directories(mechanica_shared PRIVATE ${PYBIND11_INCLUDE_DIR})



# shared and static libraries built from the same object files
add_library(mechanica_py SHARED
  $<TARGET_OBJECTS:mechanica_obj>
  mechanica.cpp
  )

target_include_directories(mechanica_py PRIVATE
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  #${PYTHON_INCLUDE_DIRS}
  #${NUMPY_INCLUDE_DIR}
  #$<INSTALL_INTERFACE:include/mylib>  # <prefix>/include/mylib
)

target_include_directories(mechanica_py PRIVATE ${PYBIND11_INCLUDE_DIR})



target_link_libraries(mechanica_py
  Carbon::Static
  Python::Python
  Python::NumPy
  mdcore_single
  glfw
  Assimp::Assimp
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders
  JPEG::JPEG
  voro++
  sundials_cvode_static
  MagnumGlfwApplication
  Magnum_PyBind
  )

if(MX_APPLE)
  target_link_libraries(mechanica_py MagnumWindowlessCglApplication)
elseif(MX_LINUX)
  target_link_libraries(mechanica_py MagnumWindowlessEglApplication)
elseif(MX_WINDOWS)
  target_link_libraries(mechanica_py MagnumWindowlessWglApplication)
endif()


set_target_properties(mechanica_py PROPERTIES PREFIX "")
set_target_properties(mechanica_py PROPERTIES OUTPUT_NAME "_mechanica")
set_target_properties(mechanica_py PROPERTIES SUFFIX ".so")


#add_custom_command(
#  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/mechanica.py"
#  POST_BUILD
#  COMMAND ${CMAKE_COMMAND} -E copy
#  ${CMAKE_CURRENT_SRC_DIR}/mechanica.py
#  ${CMAKE_CURRENT_BINARY_DIR}/mechanica.py
#  DEPENDS mechanica.py)


#add_custom_command(
# TARGET ${destination} POST_BUILD
# COMMAND ${CMAKE_COMMAND} -E copy ${source} ${destination}
# DEPENDS ${source}
# COMMENT "symbolic link resources folder from ${source} => ${destination}"
# )

#add_custom_target(mechanica.py
#  DEPENDS mechanica.py)

add_custom_command(
  TARGET mechanica_py
  DEPENDS mechanica.py
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy  ${CMAKE_CURRENT_SOURCE_DIR}/mechanica.py ${CMAKE_CURRENT_BINARY_DIR}/mechanica.py
  )

target_link_libraries(mechanica_shared
  Carbon::Static
  Python::Python
  Python::NumPy
  mdcore_single
  glfw
  Assimp::Assimp
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders
  JPEG::JPEG
  voro++
  sundials_cvode_static
  MagnumGlfwApplication
  Magnum_PyBind
  )

if(MX_APPLE)
  target_link_libraries(mechanica_shared MagnumWindowlessCglApplication)
elseif(MX_LINUX)
  target_link_libraries(mechanica_shared MagnumWindowlessEglApplication)
elseif(MX_WINDOWS)
  target_link_libraries(mechanica_shared MagnumWindowlessWglApplication)
endif()



#add_library(mechanica_static STATIC
#  $<TARGET_OBJECTS:mechanica_obj> mechanica.cpp)

#set_target_properties(mechanica_static PROPERTIES OUTPUT_NAME mechanica)

#target_link_libraries(mechanica_static
#  Python::Python
#  glfw
#  Assimp::Assimp
#  Magnum::Application
#  Magnum::Magnum
#  Magnum::Shaders
#  voro++
#  sundials_cvode_static
#  ${OPENGL_LIBRARIES}
#  )

#if(MX_APPLE)
#  target_link_libraries(mechanica_static MagnumWindowlessCglApplication)
#elseif(MX_LINUX)
#  target_link_libraries(mechanica_static MagnumWindowlessEglApplication)
#elseif(MX_WINDOWS)
#  target_link_libraries(mechanica_static MagnumWindowlessWglApplication)
#endif()


add_library(mechanica ALIAS mechanica_shared)
add_library(Mechanica::Mechanica ALIAS mechanica_shared)
#add_library(Mechanica::Static ALIAS mechanica_static)
add_library(Mechanica::Shared ALIAS mechanica_shared)
add_library(Mechanica::PythonLib ALIAS mechanica_py)
