#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "../math/tensor.h"

typedef enum {
	OPT_SGD,
} optimizer_t;

typedef struct {
	float learning_rate;
	// Pass tensor target, tensor direction, and void*
	// parameters (if we need more than lr).
	void (*update) (Tensor*, Tensor*, float, void*);
	void* optimizer_params;
} Optimizer;

void optimizers_SGD_update (Tensor*, Tensor*, float, void*);
void optimizer_update (Optimizer* opt, Tensor* target, Tensor* dir);

Optimizer* build_optimizer (optimizer_t type, float lr, void* params);

#endif
