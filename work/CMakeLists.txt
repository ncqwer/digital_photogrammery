add_executable(forstner forstner.cpp)
target_link_libraries(forstner help_func ${OpenCV_LIBS})

install(TARGETS forstner
		DESTINATION bin)

add_executable(harris harris.cpp)
target_link_libraries(harris help_func ${OpenCV_LIBS})

install(TARGETS harris
		DESTINATION bin)

add_executable(correlation Correlation.cpp)
target_link_libraries(correlation ${OpenCV_LIBS})
install(TARGETS correlation
		DESTINATION bin)