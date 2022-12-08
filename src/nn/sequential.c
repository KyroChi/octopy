#include <stdlib.h>

#include "../math/tensor.h"
#include "sequential.h"

void
initializer (Tensor *T, initializer_t init)
{
	return;
}

void
layer_copy (Layer* in, Layer* out)
/*
 * Assumes out is allocated
 */
{
	out->type = in->type;
	// TODO: Standardize the copying across data types
	out->weights = tensor_copy(in->weights);

	return;
}

Layer*
create_dense_layer (unsigned int in_size,
		    unsigned int out_size,
		    initializer_t init)
{
	Layer* layer = malloc( sizeof(layer) );
	if (!layer) {
		// TODO: Set error flags. Failed to allocate memory
		return NULL;
	}

	layer->type = LAY_DENSE;

	unsigned int shape[2] = {in_size, out_size};
	layer->weights = new_tensor(2, shape);
	// TODO: Use initializer for the weights
	initializer(layer->weights, init);
	
	layer->activ = ACT_NONE;

	layer->input_rank = 2;
	layer->input_shape = malloc( sizeof(unsigned int) * 2 );
	layer->input_shape[0] = 1;
	layer->input_shape[1] = in_size;

	return layer;
}

Layer*
create_activation_layer (activation_t activ)
{
	Layer* layer = malloc( sizeof(layer) );
	if (!layer) {
		// TODO: Set error flags. Failed to allocate memory
		return NULL;
	}

	layer->type = LAY_ACTIVATION;
	layer->weights = NULL;
	layer->activ = activ;

	return layer;
}

Sequential*
create_sequential_net (unsigned int n_layers,
		       Layer **layers)
/*
 * Copies layers into a new array of layers
 */
{
	Sequential* net = malloc( sizeof(net) );
	if (!net) {
		// TODO: Set error flags
		return NULL;
	}

	net->layers = malloc( sizeof(Layer*) * n_layers );
	
	unsigned int ii;
	for (ii = 0; ii < n_layers; ii += 1) {
		net->layers[ii] = malloc( sizeof(Layer) );
	        layer_copy(layers[ii], net->layers[ii]);
	}

	return net;
}

Tensor*
feed_forward (Sequential* seq, unsigned int training,
	      Tensor *input,
	      Tensor** activations,
	      Tensor** derivatives)
/**
 * training: 0 or 1. 0 means not training, 1 means training.
 * activations: pointer to a Tensor**. Allocated by the function.
 */
{
	Tensor* output, *tmp;

	output = tensor_copy(input);
	
	unsigned int ii;
	for (ii = 0; ii < seq->n_layers; ii += 1) {
		if (!training) {
			tmp = evaluate_layer_not_training(seq->layers[ii], output);
		}
		// Also must do the activations.
		free(output);
		output = tmp;
		free(tmp);
	}

	return output;
}

Tensor*
evaluate_layer_not_training (Layer* layer, Tensor* Input)
{
	return tensor_copy(Input);
}

Tensor*
evaluate_layer_training (Layer* layer, Tensor* Input,
			 Tensor* activation,
			 Tensor* local_derivitives)
{
	return tensor_copy(Input);
}
