# get img location
set(img_location ${CMAKE_SOURCE_DIR}/res/234.jpg)

create_test_sourcelist(
	tests
	common_test.cpp
	sub_img_iter_try.cpp)

add_executable(common_test ${tests})
message(${tests})
target_link_libraries(common_test help_func ${OpenCV_LIBS})

set(tests_to_run ${tests})
remove(tests_to_run common_test.cpp)

foreach(test ${tests_to_run})
	get_filename_component(test_name ${test} NAME_WE)
	add_test(${test_name} common_test ${test_name} ${img_location})
endforeach()
