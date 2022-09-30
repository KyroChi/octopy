#include <stdio.h>
#include <stdlib.h>

#include "../src/math/tensor.h"
#include "../src/threading.h"

void run_tensor_tests (unsigned int *, unsigned int *);
void tensor_to_ones_test (unsigned int *, unsigned int *);
void tensor_mul_test (unsigned int *, unsigned int *);

#ifdef MULTI_THREADING
void run_threading_tests (unsigned int*, unsigned int *);
#endif

int
main ()
{
#ifdef MULTI_THREADING
	printf("Multithreading on\n");
#endif
	
	unsigned int passed = 0;
	unsigned int failed = 0;

	run_tensor_tests(&passed, &failed);

#ifdef MULTI_THREADING
	run_threading_tests(&passed, &failed);
#endif

	printf("Passed: %d\nFailed: %d\nTotal:  %d\n------------\n",
	       passed, failed, passed + failed);

	return 1;
}

void
run_tensor_tests (unsigned int *passed, unsigned int *failed)
{
	unsigned int axes[3] = {2, 2, 1};
	Tensor *t = new_tensor(3, axes);
	t->data[0] = 1.0;
	t->data[1] = 2.0;
	t->data[2] = 3.0;
	t->data[3] = 4.0;

	float test = 0.0;
	axes[0] = 1;
	axes[1] = 0;
	axes[2] = 0;

	// Test setting and indexing into a tensor
	test = get_tensor(t, axes);
	printf("Test: %.2f\n", test);
	set_tensor(t, axes, 1.3);
	test = get_tensor(t, axes);
	printf("Test: %.2f\n", test);

	unsigned int axes2[2] = {2, 2};
	Tensor *t2 = new_tensor(2, axes);
	t2->data[0] = 1.0;
	t2->data[1] = 2.0;
	t2->data[2] = 3.0;
	t2->data[3] = 4.0;

	test = 0.0;
	axes2[0] = 0;
	axes2[1] = 1;
	test = get_tensor(t2, axes2);
	printf("Test: %.2f\n", test);


	// A big tensor
	unsigned int axes3[10] = {64, 100, 100, 30, 20,
				  5, 20, 100, 100, 2};
	Tensor *bigT = new_tensor(10, axes3);


	// Test freeing a tensor.
	free_tensor(t);
	free_tensor(t2);
	free_tensor(bigT);

	tensor_mul_test(passed, failed);
	tensor_to_ones_test(passed, failed);

	return;
}

void
tensor_to_ones_test (unsigned int *passed,
		     unsigned int *failed)
{
	unsigned int axes[5] = {4, 10, 4, 28, 3};
	Tensor *A = new_tensor(2, axes);
	to_ones(A);

	unsigned int ii;
	for (ii = 0; ii < A->size; ii += 1) {
		if (A->data[ii] != 1.0) {
			*failed += 1;
			return;
		}
	}

	*passed += 1;
	return;
}

void
tensor_mul_test (unsigned int *passed,
		  unsigned int *failed)
/**
 * Test that tensor multiplication is working correctly.
 */
{
	unsigned int axes[2] = {2, 2};
	Tensor *A = new_tensor(2, axes);
	A->data[0] = 1.0;
	A->data[2] = 2.0;
	A->data[1] = 3.0;
	A->data[3] = 4.0;

	Tensor *A2;
	A2 = tensor_matmul(A, A);

	if (A2->data[0] != 7.0) {
		printf("0 %.5f\n", A2->data[0]);
		*failed += 1;
	} else if (A2->data[2] != 10) {
		printf("2 %.5f\n", A2->data[2]);
		*failed += 1;
	} else if (A2->data[1] != 15) {
		printf("1 %.5f\n", A2->data[1]);
		*failed += 1;
	} else if (A2->data[3] != 22) {
		printf("3 %.5f\n", A2->data[3]);
		*failed += 1;
	}

	free_tensor(A);
	free_tensor(A2);
	*passed += 1;

	return;
}

#ifdef MULTI_THREADING
void
run_threading_tests (unsigned int *passed,
		     unsigned int *failed)
{
	unsigned int n_test = 10;
	thread_scheduler_s* sch = new_thread_scheduler(n_test, 1);

	unsigned int ii;
	for (ii = 0; ii < n_test; ii += 1) {
		if ( !thread_scheduler_full(sch) ) {
			thread_scheduler_push(sch, NULL);
		}
	}

	if ( !thread_scheduler_full(sch) ) {
		printf("index: %d\n", sch->index);
		*failed += 1;
	} else {
		*passed += 1;
	}

	for (ii = 0; ii < n_test; ii += 1) {
		if ( thread_available(sch) ) {
			thread_scheduler_pop(sch);
		}
	}

	if ( thread_available(sch) ) {
		printf("index: %d\n", sch->index);
		*failed += 1;
	} else {
		*passed += 1;
	}

	return;
}
#endif
