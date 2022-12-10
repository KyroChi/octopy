#ifndef INITIALIZERS_H
#define INITIALIZERS_H

#include "random.h"
#include "tensor.h"

typedef enum {
	INIT_DEFAULT,
	INIT_GAUSSIAN,
	INIT_UNIFORM,
	INIT_GLOROT,
	INIT_HE,
} initializer_t;

typedef struct {
	initializer_t type;
	float p1;
	float p2;
	float (*func) (float, float);
} Initializer;

Initializer initializer_default_uniform;
Initializer initializer_symmetric_uniform;

void _tensor_map_initializer (Tensor*, float (*)(float, float),
			      float, float);
void initialize_tensor (Tensor *, Initializer *);

#endif
