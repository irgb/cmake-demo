INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/math)

ADD_EXECUTABLE(raw_test raw_test.cpp)
TARGET_LINK_LIBRARIES(raw_test math)

add_test(NAME raw_test_1 COMMAND $<TARGET_FILE:raw_test> 1 2)
set_tests_properties(raw_test_1 PROPERTIES PASS_REGULAR_EXPRESSION "3")

add_test(NAME raw_test_2 COMMAND $<TARGET_FILE:raw_test> 2 1)
set_tests_properties(raw_test_2 PROPERTIES PASS_REGULAR_EXPRESSION "3")

macro (do_raw_test arg1 arg2 result)
  add_test (NAME raw_test_${arg1}_${arg2} COMMAND $<TARGET_FILE:raw_test> ${arg1} ${arg2})
  set_tests_properties (raw_test_${arg1}_${arg2} PROPERTIES PASS_REGULAR_EXPRESSION "${result}")
endmacro (do_raw_test)

do_raw_test(1 2 "3")
do_raw_test(2 1 "3")
