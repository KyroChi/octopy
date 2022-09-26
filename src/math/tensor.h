#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
	unsigned int rank;
	unsigned int size;
	unsigned int *shape;
	float *data;
} Tensor;

/*
 * Access operations: creation, deletion, indexing
 * (also includes relevant helper functions)
 */
Tensor *new_tensor (unsigned int, unsigned int *);
void free_tensor (Tensor *);

unsigned int get_linear_index (unsigned int, unsigned int *,
			       unsigned int *);
void set_tensor(Tensor *, unsigned int *, float);
float get_tensor (Tensor *, unsigned int *);

int check_index_validity (Tensor *, unsigned int *);

/*
 * Mapping
 * (includes relevant helper functions)
 */

float one (float);
void to_ones (Tensor *);
void _tensor_map_subroutine (Tensor *, Tensor *, float(*fun)(float));

#endif
