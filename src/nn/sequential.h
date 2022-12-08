#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "../math/tensor.h"

typedef enum {
	INIT_DEFAULT,
	INIT_GAUSSIAN,
	INIT_UNIFORM,
	INIT_GLOROT,
	INIT_HE,
} initializer_t;

typedef enum {
	ACT_IDENTITY,
	ACT_TANH,
	ACT_RELU,
	ACT_LEAKY_RELU,
	ACT_SIGMOID,
	ACT_NONE,
} activation_t;

typedef enum {
	LAY_INPUT,
	LAY_FLATTEN,
	LAY_DENSE,
	LAY_ACTIVATION,
	LAY_DROPOUT,
} layer_t;

typedef struct {
	layer_t type;
	Tensor *weights;
	Tensor *bias;
	activation_t activ;
	/* Rank of input tensor */
	unsigned int input_rank;
	/* Shape of input tensor (unbatched) 
	 * for batched the first dimensions will change from 1 to 
	 * batch_size.
	 *
	 * For example, a dense layer has input_shape (1, in_size), 
	 * and a batched computation will be (batch_size, in_size).
	 */
	unsigned int* input_shape;
} Layer;

typedef struct {
	// Hmmm.....
} Optimizer;

typedef struct {
	activation_t activ;
	float (*func) (float);   // The activation function
	float (*func_d) (float); // It's derivative
} Activation;

float activ_identity (float);
float activ_identity_d (float);

Activation identity;

typedef struct {
	Layer** layers;
	unsigned int n_layers;
	Optimizer *optimizer;
} Sequential;

Sequential* create_sequential_net (unsigned int n_layers,
				   Layer **layers);
void free_sequential_net (Sequential*);

void initializer (Tensor *, initializer_t);
void layer_copy (Layer* in, Layer* out);

Layer* create_dense_layer (unsigned int in_size,
			   unsigned int out_size,
			   initializer_t init);
Layer* create_flatten_layer (unsigned int *axes, unsigned int rank);
Layer* create_activation_layer (activation_t activ);
Layer* create_dropout_layer (float dropout);

void free_layer (Layer*);

Tensor* evaluate_layer_not_training (Layer* layer, Tensor* Input);
Tensor* evaluate_layer_training (Layer* layer, Tensor* Input,
				 Tensor* activation,
				 Tensor* local_derivitives);
void update_layer_weights(Layer* layer, Tensor* grads,
			  Optimizer* opt);

Tensor* feed_forward(Sequential* seq,
		     unsigned int training, Tensor* input,
		     Tensor** activations, Tensor** derivatives);
Tensor** backprop(Sequential* seq, Tensor** activations);

#endif
