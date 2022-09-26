CC = cc

FLAGS = -Wall -Werror -Wextra -pedantic
DEBUG = -g

MATH = 		src/math/tensor.c

TEST = 		test/run_tests.c

HELP =		src/octopy_helper.c

test: $(OBJECTS)
	$(CC) $(TEST) $(MATH) $(HELP) -o test.o $(DEBUG)
