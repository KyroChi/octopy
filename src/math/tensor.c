#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "../octopy_helper.h"
#include "../threading.h"
#include "tensor.h"

Tensor *
new_tensor (unsigned int rank, unsigned int *shape)
/* COPIES shape into the tensor. */
{
	Tensor *ptr;
	ptr = malloc( sizeof(*ptr) );
	if (ptr == NULL) {
		// TODO: Set error flags
		return NULL;
	}

	ptr->rank = rank;
	
	ptr->shape = malloc( sizeof(unsigned int) * rank );
	if ( ptr->shape == NULL ) {
		// TODO: Set error flags
		return NULL;
	}

	ptr->indexer = malloc( sizeof(unsigned int) * rank );
	if ( ptr->indexer == NULL ) {
		// TODO: Set error flags
		return NULL;
	}
	
	array_cpy_uint(shape, ptr->shape, rank);

	unsigned int tensor_size = 1;
	
	unsigned int ii;
	for (ii = 0; ii < rank; ii += 1) {
		tensor_size *= shape[ii];
	}

	ptr->indexer[0] = 1;
	for (ii = 1; ii < rank; ii += 1) {
		ptr->indexer[ii] = ptr->shape[ii - 1];
	}

	ptr->size = tensor_size;

	// Initialize with zero data
	ptr->data = calloc( tensor_size, sizeof(float) );
	if ( !ptr->data ) {
		// TODO: Set error flags
		return NULL;
	}
	if (ptr->data == NULL) {
		// TODO: Set error flags
		return NULL;
	}

	return ptr;
}

void
free_tensor (Tensor *T)
{
	free(T->data);
	free(T->indexer);
	free(T->shape);
	free(T);

	return;
}

void
tensor_print (Tensor *T)
/**
 * I think that there is a better algorithm, this is sloooowww. 
 * But you probably shouldn't print big tensors anyways.
 *
 * Currently only supports rank 1 or 2 tensors
 */
{
	unsigned int *idxs = malloc(sizeof(unsigned int)*T->rank);
	unsigned int ii;
	
	if (T->rank == 1) {
		printf("[");
		for (ii = 0; ii < T->size - 1; ii += 1) {
			printf("%.2f,\t", T->data[ii]);
		}
		printf("%.2f]\n", T->data[T->size - 1]);
	} else if (T->rank == 2) {
		printf("[[");
		for (ii = 0; ii < T->size - 1; ii += 1) {
			get_index_idxs(T, ii, idxs);
			if (idxs[0] + 1
			    == T->shape[T->rank - 1]) {
				printf("%.2f],\n", T->data[ii]);
			} else {
				if (idxs[0] == 0 && idxs[1] != 0) {
					printf(" [%.2f\t",
					       T->data[ii]);
				} else {
					printf("%.2f,\t",
					       T->data[ii]);
				}
			}
		}
		printf("%.2f]]\n", T->data[T->size - 1]);
	} else {
		printf("Printing currently only supports rank 1 or 2 tensors\n");
	}
	return;
}

unsigned int
get_index_linear (Tensor *T, unsigned int *idx)
{
	unsigned int index = 0;
	
	unsigned int ii;
	for (ii = 0; ii < T->rank; ii += 1) {
		index += T->indexer[ii] * idx[ii];
	}

	return index;
}

void
get_index_idxs (Tensor *T, unsigned int index, unsigned int *idx)
/* Assumes that idx is pre-allocated */
{
	unsigned int ii;
	for (ii = 0; ii < T->rank - 1; ii += 1) {
		idx[ii] = index % T->shape[ii];
		index = (index - idx[ii]) / T->shape[ii];
	}

	idx[ii] = index;
	
	return;
}

void
set_tensor(Tensor *T, unsigned int *idxs, float v)
/**
 * Checking of idxs is done by caller to avoid double calling
 */
{
	unsigned int index = get_index_linear(T, idxs);
	T->data[index] = v;
	return;
}

float
get_tensor (Tensor *T, unsigned int *idxs)
/* idxs length must match rank */
{
	unsigned int index = get_index_linear(T, idxs);
	return T->data[index];
}

int
check_index_validity (Tensor *T, unsigned int *idxs)
/**
 * Assumes that the caller has already checked that the length of idxs 
 * is correct .
 */
{
	unsigned int rank = T->rank;

	unsigned int ii;
	for (ii = 0; ii < rank; ii += 1) {
		if (T->shape[ii] <= idxs[ii]) {
			return -1;
		}
	}

	return 0;
}

Tensor*
tensor_copy (Tensor* A)
{

	if ( !A ) {
		// If we are passed a NULL value return a NULL value.
		// is this the desired behavior?
		return NULL;
	}
	
	Tensor* out = new_tensor(A->rank, A->shape);
	
	unsigned int ii;
	for (ii = 0; ii < A->size; ii += 1) {
		out->data[ii] = A->data[ii];
	}

	return out;
}

int
check_same_size (Tensor *A, Tensor *B)
{
	if ( A->rank != B->rank ) {
		return -1;
	}

	unsigned int ii;
	for (ii = 0; ii < A->rank; ii += 1) {
		if ( A->shape[ii] != B->shape[ii] ) {
			return -1;
		}
	}

	return 0;
}

float one (float a) { UNUSED(a); return 1.0; }

void
to_ones (Tensor *T)
{
	_tensor_map_subroutine(T, T, &one);
	return;
}

Tensor* 
scalar_multiply (Tensor* in, float a)
{
	// TODO: Merge with map
	printf("called\n");
	if (in->shape == NULL) {
		printf("Issue\n");
	}
	printf("no seg fault yet\n");
	
	Tensor* out = new_tensor(in->rank, in->shape);
	printf("no seg fault yet 2\n");
	
	unsigned int ii;
	for (ii = 0; ii < in->size; ii += 1) {
		out->data[ii] = a * in->data[ii];
	}
	
	return out;
}

void
_tensor_map_subroutine (Tensor *in, Tensor* out,
			float (*map_fun)(float))
/**
 * Call _matrix_map_subroutine(T, T, &fun) for inplace
 */
// TODO: Optimize
{
	// TODO: Check that in and out are same dimension
	// fast for inpace: check that they are same pointer
	float a;

	// This loop avoids having to index into the tensor.
	unsigned int ii;
	for (ii = 0; ii < in->size; ii += 1) {
		a = in->data[ii];
		out->data[ii] = map_fun(a);
	}

	return;
}

Tensor *
zip_tensor_map (Tensor *A, Tensor*B,
		float (*map_fun)(float, float))
/**
 * Assumes that the caller has checked that sizes match.
 * Call zip_tensor_map_s for safe call.
 */
// TODO: Optimize
{
	Tensor *res = new_tensor(A->rank, A->shape);
	if ( res == NULL ) {
		// TODO: Set error flags
		return NULL;
	}

	unsigned int ii;
	for (ii = 0; ii < A->size; ii += 1) {
		res->data[ii] = map_fun(A->data[ii], B->data[ii]);
	}

	return res;
}

float add (float a, float b) { return a + b; }
float sub (float a, float b) { return a - b; }
float mul (float a, float b) { return a * b; }

Tensor *
tensor_add (Tensor *A, Tensor *B)
{
	return zip_tensor_map(A, B, &add);
}

Tensor *
tensor_add_s (Tensor *A, Tensor *B)
{
	return zip_tensor_map_s(A, B, &add);
}

Tensor *
tensor_sub (Tensor *A, Tensor *B)
{
	return zip_tensor_map(A, B, &sub);
}

Tensor *
tensor_sub_s (Tensor *A, Tensor *B)
{
	return zip_tensor_map_s(A, B, &sub);
}

Tensor *
tensor_mul (Tensor *A, Tensor *B)
{
	return zip_tensor_map(A, B, &mul);
}

Tensor *
tensor_mul_s (Tensor *A, Tensor *B)
{
	return zip_tensor_map_s(A, B, &mul);
}

float
reduce_sum (Tensor *A)
// TODO: Fast version using CUDA and parallelization
{
	float sum = 0.0;
	unsigned int ii;
	for (ii = 0; ii < A->rank; ii += 1) {
		sum += A->data[ii];
	}

	return sum;
}

Tensor *
zip_tensor_map_s (Tensor *A, Tensor*B,
		  float (*map_fun)(float, float))
/**
 * Safe version of zip_tensor_map
 */
{
	if ( 0 > check_same_size(A, B) ) {
		// TODO: Set error flag
		return NULL;
	}

	return zip_tensor_map(A, B, map_fun);
}

void
matmul_loop (Tensor* A, Tensor* B, Tensor* AB, unsigned int ii)
/**
 * The inner for loop for the tensor multiplication.
 */
{
	float sum;
	unsigned int jj, kk;
	
	unsigned int *idxs =
		malloc( sizeof(unsigned int) * AB->rank);
	unsigned int *A_idxs =
		malloc( sizeof(unsigned int) * A->rank);
	unsigned int *B_idxs =
		malloc( sizeof(unsigned int) * B->rank);

	get_index_idxs(AB, ii, idxs);
	sum = 0;
	for (jj = 0; jj < A->shape[A->rank - 1]; jj += 1) {
		for (kk = 0; kk < A->rank - 1; kk += 1) {
			A_idxs[kk] = idxs[kk];
		}

		A_idxs[kk] = jj;

		B_idxs[0] = jj;
			
		for (kk = 1; kk < B->rank; kk += 1) {
			B_idxs[kk] = idxs[A->rank - 2 + kk];
		}

		sum += get_tensor(A, A_idxs) *
			get_tensor(B, B_idxs);
	}

	set_tensor(AB, idxs, sum);

	free(idxs);
	free(A_idxs);
	free(B_idxs);
	
	return;
}

#ifdef MULTI_THREADING
void*
matmul_loop_p_thread (void* args)
/**
 * Wrapper for the matmul_loop function which can be used with a 
 * thread_scheduler.
 */
{
	matmul_loop_s *p = (matmul_loop_s*) args;

	// Sets the value in the tensor
	matmul_loop(p->A, p->B, p->AB, p->index);

	pthread_mutex_lock(p->mutex);
	thread_scheduler_push(p->t_sch, (void *)p);
	pthread_mutex_unlock(p->mutex);

	pthread_cond_broadcast(p->sf);
	return NULL;
}
#endif

Tensor*
tensor_matmul (Tensor *A, Tensor *B)
/* Multiply tensors along the last axis of and the first axis of B */
{
	if ( A->shape[A->rank - 1] != B->shape[0] ) {
		// Set error flags
		return NULL;
	}

	unsigned int *new_shape = NULL;
	unsigned int new_rank = A->rank + B->rank - 2;

	if (new_rank == 0) {
		// TODO: Return dot product
		printf("New rank is zero\n");
		return NULL;
	} else {
		new_shape =
			malloc( sizeof(unsigned int) * new_rank );
	}

	unsigned int ii;
	for (ii = 0; ii < A->rank - 1; ii += 1) {
		new_shape[ii] = A->shape[ii];
	}

	for (ii = 1; ii < B->rank; ii += 1) {
		new_shape[A->rank + ii - 2] = B->shape[ii];
	}

	Tensor *AB = new_tensor(new_rank, new_shape);
	free(new_shape);
#ifdef CUDA
	// TODO: CUDA code goes here
	printf("CUDA is not implemented\n");
	exit(3);
#endif

#ifdef MULTI_THREADING
	// TODO: decide if I should use multi-threading or not based
	// on the size of the tensors.	
	thread_scheduler_s *t_sch =
		new_thread_scheduler(MAXIMUM_THREADS,
				     sizeof(matmul_loop_s));
	if ( !t_sch ) {
		// TODO: Set error flags
		printf("Failed to instantiate thread scheduler\n");
		return NULL;
	}

	pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
	pthread_mutex_t dummy = PTHREAD_MUTEX_INITIALIZER;
	pthread_cond_t scheduler_full = PTHREAD_COND_INITIALIZER;

	for (ii = 0; ii < MAXIMUM_THREADS; ii += 1) {
		((matmul_loop_s*) t_sch->av[ii])->A = A;
		((matmul_loop_s*) t_sch->av[ii])->B = B;
		((matmul_loop_s*) t_sch->av[ii])->AB = AB;
		((matmul_loop_s*) t_sch->av[ii])->mutex = &mutex;
		((matmul_loop_s*) t_sch->av[ii])->sf = &scheduler_full;
		((matmul_loop_s*) t_sch->av[ii])->t_sch = t_sch;
	}

	matmul_loop_s *args = NULL;
	
	for (ii = 0; ii < AB->size; ii += 1) {
		if ( !thread_available(t_sch) ) {
			// Wait for a thread to become available

			// dummy mutex since the workers are handled
			// by the thread_scheduler.
			pthread_cond_wait(&scheduler_full, &dummy);
		}

		if ( thread_available(t_sch) ) {
			args = (matmul_loop_s*)
				thread_scheduler_pop(t_sch);
			args->index = ii;
			pthread_create(&(args->my_id),
				       NULL,
				       &matmul_loop_p_thread,
				       args);
		} else {
			// TODO: Set error flags, something has gone
			// wrong. This shouldn't be called. Maybe the
			// threads are accessing each other's memory?
			printf("Threads sharing memory\n");
			return NULL;
		}
	}	

	for (ii = 0; ii < MAXIMUM_THREADS; ii += 1) {
			// Check that the threads finished. This works
			// even when we span fewer threads than
			// MAXIMUM_THREADS.
			pthread_join(((matmul_loop_s*)t_sch->av[ii])->my_id, NULL);
		}
				
#else
	for (ii = 0; ii < AB->size; ii += 1) {
		matmul_loop(A, B, AB, ii);
	}
#endif

	return AB;
}

Tensor *
reshape (Tensor *A, unsigned int new_rank, int *new_shape)
/*
 * int ptr instead of unsigned int ptr to allow for negative indexing
 * to infer dimension
 */
{
	// This should only change the rank and shape, need to do
	// math to infer new axes.
	return NULL;
}

Tensor *
tensor_concatenate (Tensor *A, Tensor *B, unsigned int axis)
{
	return NULL;
}

Tensor *
tensor_transpose (Tensor *T, unsigned int ax1, unsigned int ax2)
/* Is there a better way to do this? */
{ /*
	// TODO: Check validity of axes
	unsigned int *new_axes = calloc(T->rank,
					sizeof(unsigned int));
	array_cpy_uint(T->axes, new_axes);
	
	unsigned int tmp = 0;
	tmp = new_axes[ax1];
	new_axes[ax1] = new_axes[ax2];
	new_axes[ax2] = tmp;
	
	Tensor *newT = new_tensor(T->rank, new_axes);

	float ltmp;

	
	unsigned int ii, jj;
	for (ii = 0; ii < rank; ii += 1) {
		if (ii == ax1) {
		} else if (ii == ax2) {
			// Do nothing
		} else {
			for (jj = 0; jj < new_axes[ii]; jj += 1) {
				set_tensor(newT, 
			}
		}
		} */ return NULL; // Finish this
}

void
tensor_transpose_inplace (Tensor *T, unsigned int ax1,
			  unsigned int ax2)
{
	return;
}
