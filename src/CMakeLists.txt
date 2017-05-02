add_library(help_func  STATIC help_func.cpp)
target_link_libraries(help_func ${OpenCV_LIBS})

install(TARGETS help_func
		RUNTIME DESTINATION bin
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib/static)