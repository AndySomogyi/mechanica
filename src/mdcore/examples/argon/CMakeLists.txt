add_executable(argon argon.cpp)

target_compile_definitions(argon PUBLIC ENGINE_FLAGS=engine_flag_none FPTYPE_SINGLE)

find_package(Threads REQUIRED)

target_link_libraries(argon
	Threads::Threads
	Mechanica::Mechanica)
