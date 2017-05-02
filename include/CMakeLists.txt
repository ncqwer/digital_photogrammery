file(GLOB header_file "*.h*")

install(FILES ${header_file}
		DESTINATION inc)