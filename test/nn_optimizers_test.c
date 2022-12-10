#include <stdio.h>
#include <math.h>

#include "../src/math/tensor.h"
#include "../src/nn/optimizers.h"
#include "octopy_test.h"

void
optimizer_construction_test (unsigned int* passed,
			     unsigned int* failed)
{
	return;
}

void
SGD_update_test (unsigned int* passed,
		 unsigned int* failed)
{
	unsigned int shape[2] = {2, 2};
	Tensor *z = new_tensor(2, shape);
	Tensor *o = new_tensor(2, shape);
	to_ones(o);

	Optimizer* opt = build_optimizer(OPT_SGD, 0.1, NULL);
	optimizer_update(opt, z, o);

	unsigned int ii;
	for (ii = 0; ii < z->size; ii += 1) {
		if ( fabs(z->data[ii] - 0.1) > 0.0000001 ) {
			printf("Failed SGD update test. Expected %.10f, got %.10f\n", 0.1, z->data[ii]);
			*failed += 1;
			return;
		}
	}

	*passed += 1;
	return;
}

void
run_nn_optimizers_tests (unsigned int* passed,
			 unsigned int* failed)
{
	optimizer_construction_test(passed, failed);
	SGD_update_test(passed, failed);
	
	return;
}
