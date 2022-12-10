#include "initializers.h"
#include "tensor.h"
#include "random.h"

Initializer initializer_default_uniform = {
	.type = INIT_UNIFORM,
	.p1 = -1,
	.p2 = 1,
	.func = &rand_uniform,
};

float
initializer_glorot (float a, float b)
{
	return 0.0;
}

void
_tensor_map_initializer (Tensor *T,
			 float (*init_func) (float, float),
			 float p1, float p2)
{
	unsigned int ii;
	for (ii = 0; ii < T->size; ii += 1) {
		T->data[ii] = init_func(p1, p2);
	}

	return;
}

void
initialize_tensor (Tensor *T, Initializer *I)
/*
 * Initializes tensor inplace using the supplied initializer.
 */
{
	_tensor_map_initializer(T, I->func, I->p1, I->p2);
	return;
}
