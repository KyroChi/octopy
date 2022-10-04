#ifndef TENSOR_H
#define TENSOR_H

#include <pthread.h>
#include "../threading.h"

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

Tensor* tensor_copy (Tensor*);

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

Tensor* tensor_add (Tensor *, Tensor *);
Tensor* tensor_add_s (Tensor *, Tensor *);
Tensor* tensor_sub (Tensor *, Tensor *);
Tensor* tensor_sub_s (Tensor *, Tensor *);
Tensor* tensor_mul (Tensor *, Tensor *);
Tensor* tensor_mul_s (Tensor *, Tensor *);

Tensor* reshape (Tensor*, unsigned int, int*);

#ifdef MULTI_THREADING
typedef struct {
	Tensor *A;
	Tensor *B;
	Tensor *AB;
	unsigned int index;
	/* The following are required for the scheduler */
	thread_scheduler_s *t_sch;
	pthread_t my_id;
	pthread_mutex_t *mutex;
	pthread_cond_t *sf;
} matmul_loop_s;

void* matmul_loop_p_thread (void* args);
#endif
	
void matmul_loop (Tensor*, Tensor*, Tensor*, unsigned int);
Tensor* tensor_matmul (Tensor*, Tensor*);

float reduce_sum (Tensor*);


#endif
