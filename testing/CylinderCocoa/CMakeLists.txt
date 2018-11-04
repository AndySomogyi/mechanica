add_executable(
  CylinderCocoa
  MACOSX_BUNDLE
  AppDelegate.h
  AppDelegate.m
  main.m
  MainMenu.xib
  )



target_link_libraries(CylinderCocoa
  CylinderComponent
  )


set_source_files_properties(
    MainMenu.xib
    PROPERTIES
    MACOSX_PACKAGE_LOCATION
    Resources
)

set_target_properties(
    CylinderCocoa
    PROPERTIES
    MACOSX_BUNDLE_INFO_PLIST
    ${CMAKE_CURRENT_LIST_DIR}/plist.in
)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -framework AppKit")
