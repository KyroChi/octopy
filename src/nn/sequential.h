#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "../math/tensor.h"
#include "../math/initializers.h"

#include "activation.h"
#include "optimizers.h"

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
	Layer** layers;
	unsigned int n_layers;
	Optimizer *optimizer;
	Tensor** activs;
	Tensor** derivs;
} Sequential;

Sequential* create_sequential_net (unsigned int n_layers,
				   Layer **layers);
Sequential* create_sequential_net_basic (unsigned int,
					 unsigned int,
					 unsigned int,
					 unsigned int,
					 activation_t,
					 activation_t,
					 Optimizer*);
void free_sequential_net (Sequential*);

void initializer (Tensor *, initializer_t);
void layer_copy (Layer* in, Layer* out);

Layer* create_dense_layer (unsigned int in_size,
			   unsigned int out_size,
			   Initializer* init);
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
		     unsigned int training, Tensor* input);
Tensor** backprop(Sequential* seq, Tensor** activations);

#endif
