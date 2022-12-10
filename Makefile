CC = cc

FLAGS = -Wall -Werror -Wextra -pedantic
DEBUG = -g

MATH = 		src/math/initializers.c \
		src/math/random.c \
		src/math/tensor.c \
		src/math/mtwister/mtwister.c

NN = 		src/nn/activation.c \
		src/nn/losses.c \
		src/nn/optimizers.c \
		src/nn/sequential.c

BENCH =		benchmarks/run_benchmarks.c

TEST = 		test/run_tests.c \
		test/nn_optimizers_test.c

HELP =		src/octopy_helper.c \
		src/threading.c

SRC_TARGETS = 	$(MATH) $(NN) $(HELP)

MULTITHREAD =	-DMULTI_THREADING

test: $(OBJECTS)
	$(CC) $(TEST) $(SRC_TARGETS) -o test.o $(DEBUG)

test_mt: $(OBJECTS)
	$(CC) $(TEST) $(MATH) $(NN) $(HELP) -o test_mt.o $(DEBUG) $(MULTITHREAD)

sequential: $(OBJECTS)
	$(CC) test/sequential_api.c $(MATH) $(NN) $(HELP) -o sequential.o

bench: $(OBJECTS)
	$(CC) $(BENCH) $(MATH) $(HELP) -o bench.o 

bench_mt: $(OBJECTS)
	$(CC) $(BENCH) $(MATH) $(HELP) -o bench_mt.o $(MULTITHREAD)

module:
	python setup.py build; python setup.py install
