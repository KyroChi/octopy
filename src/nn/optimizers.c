#include <stdlib.h>

#include "../math/tensor.h"
#include "optimizers.h"

void
optimizers_SGD_update (Tensor* target, Tensor* dir,
		       float lr, void* params)
/**
 * Updates target in place.
 */
{
	scalar_multiply_inplace(dir, lr);
	tensor_add_inplace(target, dir);
	return;
}

void
optimizer_update (Optimizer* opt, Tensor* target, Tensor* dir)
{
	opt->update(target,
		    dir,
		    opt->learning_rate,
		    opt->optimizer_params);

	return;
}

Optimizer*
build_optimizer (optimizer_t type, float lr, void* params)
{
	Optimizer *opt = malloc( sizeof(Optimizer) );
	opt->learning_rate = lr;
	opt->optimizer_params = params;
	
	switch (type) {
	case OPT_SGD:
	default:
		opt->update = optimizers_SGD_update;
	}

	return opt;
}
