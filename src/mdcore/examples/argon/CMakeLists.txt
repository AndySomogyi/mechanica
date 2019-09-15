add_executable(argon argon.c)
target_compile_definitions(argon PUBLIC ENGINE_FLAGS=engine_flag_none FPTYPE_SINGLE)
target_link_libraries(argon mdcore_single)

  
