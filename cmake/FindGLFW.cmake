# Find GLFW
# ---------
#
# Finds the LOCAL GLFW library. This module defines:
#
#  GLFW_FOUND               - True if GLFW library is found
#  GLFW::GLFW               - GLFW imported target
#
# Additionally these variables are defined for internal usage:
#
#  GLFW_LIBRARY             - GLFW library
#  GLFW_INCLUDE_DIR         - Root include dir
#
# Designed to tell sub-projects to use the GLFW we build here locally,
# instead of the system GLFW. 

# Include dir

message("Using top level find glfw...")


set(GLFW_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/glfw/include/GLFW)
set(GLFW_FOUND TRUE)
set(GLFW_LIBRARY $<TARGET_FILE:glfw>)
add_library(GLFW::GLFW ALIAS glfw)



message("GLFW_INCLUDE_DIR: ${GLFW_INCLUDE_DIR}")
message("GLFW_LIBRARY: ${GLFW_LIBRARY}")
message("GLFW_FOUND: ${GLFW_FOUND}")

