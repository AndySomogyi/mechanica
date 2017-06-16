
# LLVM compile options
add_definitions(${LLVM_LDFLAGS})

add_compile_options(-DMX_STRICT)


set(SRC
  SOIL.h
  image_DXT.h
  image_helper.h
  stb_image_aug.h
  stbi_DDS_aug_c.h
  SOIL.c
  image_DXT.c
  image_helper.c
  stb_image_aug.c
  stbi_DDS_aug.h
  )



add_library(soil STATIC ${SRC})

target_include_directories(soil PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )

target_link_libraries(soil PUBLIC ${OPENGL_LIBRARIES})


message("SOIL_SRC: ${SRC}")


