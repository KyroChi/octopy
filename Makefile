CC = cc

FLAGS = -Wall -Werror -Wextra -pedantic
DEBUG = -g

MATH = 		src/math/tensor.c

NN = 		src/nn/sequential.c

BENCH =		benchmarks/run_benchmarks.c

TEST = 		test/run_tests.c

HELP =		src/octopy_helper.c \
		src/threading.c

SRC_TARGETS = 	$(MATH) $(NN) $(HELP)

MULTITHREAD =	-DMULTI_THREADING

test: $(OBJECTS)
	$(CC) $(TEST) $(SRC_TARGETS) -o test.o $(DEBUG)

test_mt: $(OBJECTS)
	$(CC) $(TEST) $(MATH) $(NN) $(HELP) -o test_mt.o $(DEBUG) $(MULTITHREAD)

bench: $(OBJECTS)
	$(CC) $(BENCH) $(MATH) $(HELP) -o bench.o 

bench_mt: $(OBJECTS)
	$(CC) $(BENCH) $(MATH) $(HELP) -o bench_mt.o $(MULTITHREAD)
