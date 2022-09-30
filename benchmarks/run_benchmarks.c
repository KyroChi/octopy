#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../src/math/tensor.h"

void
run_tensor_matmul_bench ()
{
	// Takes about 3500 in second dimension for the parallel
	// implementation to be faster.
	// For 5000, the parallel is about twice as fast
	// At 10k, we have linear: 5.1 seconds, parallel: 2.5
	// At 15k, we have linear: 7.9 seconds, parallel: 4.4
	// At 30k, we have linear: 20.71 seconds, parallel: 10.5
	// So after 5000 we maintain about twice as fast.
	unsigned int axesA[2] = {100, 5000};
	Tensor *A = new_tensor(2, axesA);

	unsigned int axesB[2] = {5000, 100};
	Tensor *B = new_tensor(2, axesB);

	struct timespec start, finish;
	double elapsed;

	clock_gettime(CLOCK_MONOTONIC, &start);

	Tensor* res = tensor_matmul(A, B);
	
	if ( !res ) {
		printf("Failure\n");
	}
	
	clock_gettime(CLOCK_MONOTONIC, &finish);

	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
	
#ifdef MULTI_THREADING
	printf("Parallel took %.8f seconds\n", elapsed);
#else
	printf("Linear took %.8f seconds\n", elapsed);
#endif
}

int
main()
{
	run_tensor_matmul_bench();
}
