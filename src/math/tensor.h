#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
	unsigned int rank;
	unsigned int size;
	unsigned int *shape;
	unsigned int *indexer;
	float *data;
} Tensor;

/*
 * Access operations: creation, deletion, indexing
 * (also includes relevant helper functions)
 */
Tensor *new_tensor (unsigned int, unsigned int *);
void free_tensor (Tensor *);

unsigned int get_index_linear (Tensor *, unsigned int *);
void get_index_idxs (Tensor *, unsigned int, unsigned int *);

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
Tensor *zip_tensor_map (Tensor *, Tensor *,
			float (*map_fun)(float, float));
Tensor *zip_tensor_map_s (Tensor *, Tensor *,
			  float (*map_fun)(float, float));

/*
 * Tensor algebra
 * (includes relevant helper functions)
 */
float add (float, float);
float sub (float, float);
float mul (float, float);

Tensor *tensor_add (Tensor *, Tensor *);
Tensor *tensor_add_s (Tensor *, Tensor *);
Tensor *tensor_sub (Tensor *, Tensor *);
Tensor *tensor_sub_s (Tensor *, Tensor *);
Tensor *tensor_mul (Tensor *, Tensor *);
Tensor *tensor_mul_s (Tensor *, Tensor *);

Tensor *tensor_matmul (Tensor *, Tensor *);

float reduce_sum (Tensor *);


#endif
