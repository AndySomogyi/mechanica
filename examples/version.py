import mechanica as m
print("Mechanica Version:")
print("version: ",  m.version.version)
print("build data: ", m.version.build_date)
print("compiler: ", m.version.compiler)
print("compiler_version: ", m.version.compiler_version)
print("system_version: ", m.version.system_version)

for k, v in m.version.cpuinfo().items():
    print("cpuinfo[", k, "]: ",  v)

for k, v in m.version.compile_flags().items():
    print("compile_flags[", k, "]: ",  v)
