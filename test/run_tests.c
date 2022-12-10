#include <stdio.h>
#include <stdlib.h>

#include "../src/math/tensor.h"
#include "../src/nn/sequential.h"
#include "../src/threading.h"
#include "../src/math/random.h"

void run_tensor_tests (unsigned int *, unsigned int *);
void tensor_copy_test (unsigned int *, unsigned int *);
void tensor_to_ones_test (unsigned int *, unsigned int *);
void tensor_scalar_mul_test (unsigned int *, unsigned int *);
void tensor_mul_test (unsigned int *, unsigned int *);
void run_sequential_net_tests (unsigned int *, unsigned int *);
void run_random_tests (unsigned int *, unsigned int *);

#ifdef MULTI_THREADING
void run_threading_tests (unsigned int*, unsigned int *);
#endif

int
main ()
{
#ifdef MULTI_THREADING
	printf("Multithreading on\n");
#endif
#ifdef CUDA
	// Check that a CUDA device is available. Otherwise
	// print something like "failed to find GPU device" and exit.
	printf("Using GPU device\n");
#endif
	
	unsigned int passed = 0;
	unsigned int failed = 0;

	run_tensor_tests(&passed, &failed);
	/* run_sequential_net_tests(&passed, &failed); */
	run_random_tests(&passed, &failed);

#ifdef MULTI_THREADING
	/* run_threading_tests(&passed, &failed); */
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

	tensor_copy_test(passed, failed);
	tensor_mul_test(passed, failed);
	tensor_scalar_mul_test(passed, failed);
	tensor_to_ones_test(passed, failed);

	return;
}

void
tensor_copy_test (unsigned int *passed,
		  unsigned int *failed)
{
	unsigned int axes[5] = {4, 10, 4, 28, 3};
	Tensor *A = new_tensor(5, axes);
	to_ones(A);

	Tensor *B = tensor_copy(A);
	
	unsigned int ii;
	for (ii = 0; ii < B->size; ii += 1) {
		if ( B->data[ii] != 1.0 ) {
			*failed += 1;
			return;
		}
	}

	*passed += 1;
	return;
}

void
tensor_to_ones_test (unsigned int *passed,
		     unsigned int *failed)
{
	unsigned int axes[5] = {4, 10, 4, 28, 3};
	Tensor *A = new_tensor(5, axes);
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
tensor_scalar_mul_test (unsigned int *passed,
			unsigned int *failed)
{
	unsigned int axes[2] = {4, 4};
	Tensor *A = new_tensor(2, axes);
	to_ones(A);

	Tensor *B = scalar_multiply(A, 3.0);
	if (B->data[0] != 3.0) {
		printf("Failed scalar multiply.");
		*failed += 1;
	} else {
		*passed += 1;
	}
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

void
run_sequential_net_tests (unsigned int *passed,
			  unsigned int *failed)
{
	Layer** layers = malloc( sizeof(Layer*) * 6 );
	layers[0] = create_dense_layer(10, 30, INIT_DEFAULT);
	layers[1] = create_activation_layer(ACT_TANH);
	layers[2] = create_dense_layer(30, 10, INIT_DEFAULT);
	layers[3] = create_activation_layer(ACT_TANH);
	layers[4] = create_dense_layer(10, 1, INIT_DEFAULT);
	layers[5] = create_activation_layer(ACT_SIGMOID);

	Sequential* net = create_sequential_net(6, layers);

	Tensor* input = NULL;
	Tensor** activations = NULL;
	Tensor** derivatives = NULL;

	unsigned int rank = 2;
	unsigned int *axes = malloc(sizeof(unsigned int) * rank);
    
	input = new_tensor(rank, axes);
	
	Tensor* net_out = feed_forward(net, 0, input,
				       activations,
				       derivatives);

	// I guess if we get here we have successfully run the tests?
	*passed += 1;
	return;
}

void
run_random_tests (unsigned int* passed, unsigned int* failed)
{
	float r;
	unsigned int ii;
	unsigned int bounds_good = 1;
	for (ii = 0; ii < 1000; ii += 1) {
		r = _rand_uniform();
		if ( r < 0 || r > 1 ) {
			bounds_good = 0;
			printf("Failed _rand_uniform test\n");
			break;
		}
	}

	if ( bounds_good ) {
		*passed += 1;
	} else {
		*failed += 1;
	}

	for (ii = 0; ii < 1000; ii += 1) {
		r = rand_uniform(-1, 1);
		if ( r < -1 || r > 1 ) {
			bounds_good = 0;
			printf("Failed rand_uniform test\n");
			printf("r=%.3f\n", r);
			break;
		}
	}

	if ( bounds_good ) {
		*passed += 1;
	} else {
		*failed += 1;
	}

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
