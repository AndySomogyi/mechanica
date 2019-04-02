
# LLVM compile options
#add_definitions(${LLVM_LDFLAGS})

add_compile_options(-DMX_STRICT)

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
  mx_error.cpp
  moduledef.cpp
  mx_callable.cpp
  mx_float.cpp
  mx_import.cpp
  mx_port.cpp
  mx_string.cpp
  mx_abstract.cpp
  mx_eval.cpp
  mx_floatobject.cpp
  mx_int.cpp
  mx_number.cpp
  mx_runtime.cpp
  mx_ast.cpp
  mx_expr.cpp
  mx_function.cpp
  MxList.cpp
  MxObject.cpp
  mx_stmt.cpp
  mx_tupleobject.cpp
  MxMeshCore.cpp
  MxMesh.cpp
  MxPolygon.cpp
  MxModule.cpp
  MxSimulator.cpp
  MxTestCube.cpp
  MxTestParticles.cpp
  MxType.cpp
  MxPropagator.cpp
  MxTestMDCoreParticles.cpp
  MxModule.cpp
  MxModel.cpp
  MxSystem.cpp
  MxSymbol.cpp
  MxPropagator.cpp
  MxParticle.cpp
  MxCompositeParticle.cpp
  MxController.cpp
  MxView.cpp
  MxLattice.cpp
  MxLatticeCluster.cpp
  MxCell.cpp
  MxMeshRenderer.cpp
  MxMeshShaderProgram.cpp
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
  MxExpression.cpp
  MeshIO.cpp
  MxTesselator.cpp
  MxColumnarCell.cpp
  T1Transition.cpp
  T2Transition.cpp
  T3Transition.cpp
  HITransition.cpp
  MxConstraints.cpp
  MxForces.cpp
  MxPolygonSurfaceTensionForce.cpp
  MxPolygonAreaConstraint.cpp
  MxCellVolumeConstraint.cpp
  MxRenderTarget.cpp
  MxWindowView.cpp
  MxImageView.cpp
  ArcBallInteractor.cpp
  MxUI.cpp
  MxTestView.cpp
  MxApplication.cpp
  MxGlfwApplication.cpp
  MxWindowlessApplication.cpp
  MxSurfaceSimulator.cpp
  ${AST_SRC}
#  ${X3D_SRC}
  )


set(HEADERS
  ${MECHANICA_PUBLIC_HEADERS}
  mx_error.h
  MxDebug.h
  mechanica_llvm.h
  mechanica_private.h
  MxObject.h
  MxMeshCore.h
  MxMesh.h
  MxPolygon.h
  MxModule.h
  MxSimulator.h
  MxTestCube.h
  MxTestParticles.h
  MxType.h
  MxTestMDCoreParticles.h
  MxModule.h
  MxModel.h
  MxSystem.h
  MxSymbol.h
  MxPropagator.h
  MxParticle.h
  MxCompositeParticle.h
  MxController.h
  MxView.h
  MxLattice.h
  MxLatticeCluster.h
  MxCell.h
  MxMeshRenderer.h
  MxMeshShaderProgram.h
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
  MxExpression.h
  MeshIO.h
  MxTesselator.h
  MxColumnarCell.h
  T1Transition.h
  T2Transition.h
  T3Transition.h
  HITransition.h
  SymmetricEigen.h
  GteSymmetricEigensolver3x3.h
  MxConstraints.h
  MxForces.h
  MxPolygonSurfaceTensionForce.h
  MxPolygonAreaConstraint.h
  MxCellVolumeConstraint.h
  MxRenderTarget.h
  MxWindowView.h
  MxImageView.h
  ArcBallInteractor.h
  MxUI.h
  MxTestView.h
  MxWindowless.h
  MxApplication.h
  MxGlfwApplication.h
  MxWindowlessApplication.h
  MxSurfaceSimulator.h
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


target_link_libraries(mechanica_obj
  Python::Python
  glfw
  Assimp::Assimp
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders
  TgaImageConverter
  voro++
  sundials_cvode_static
  ${OPENGL_LIBRARIES}
  )


source_group("Public Header Files" FILES ${MECHANICA_PUBLIC_HEADERS})

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



target_link_libraries(mechanica_py
  Python::Python
  glfw
  Assimp::Assimp
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders
  TgaImageConverter
  voro++
  sundials_cvode_static
  ${OPENGL_LIBRARIES}
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
  Python::Python
  glfw
  Assimp::Assimp
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders
  TgaImageConverter
  voro++
  sundials_cvode_static
  ${OPENGL_LIBRARIES}
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
