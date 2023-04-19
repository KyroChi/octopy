#include <stdlib.h>
#include <stdio.h>

#include "../octopy_helper.h"
#include "../math/tensor.h"

#include "activation.h"
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
	out->activ = in->activ;
	out->input_rank = in->input_rank;

	unsigned int *o_shape =
		malloc(sizeof(unsigned int) * out->input_rank);
	array_cpy_uint(in->input_shape, o_shape, out->input_rank);
	out->input_shape = o_shape;

	return;
}

Layer*
create_dense_layer (unsigned int in_size,
		    unsigned int out_size,
		    Initializer* init)
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
	initialize_tensor(layer->weights, init);
	
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
 * Eats layers, i.e. copy by caller if needed in multiple places
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
		net->layers[ii] = layers[ii];
	}

	return net;
}

Sequential*
create_sequential_net_basic (unsigned int n_layers,
			     unsigned int hidden_dim,
			     unsigned int in_dim,
			     unsigned int out_dim,
			     activation_t hidden_activ_type,
			     activation_t output_activ_type,
			     Optimizer* optimizer)
/**
 * Currently uses uniform initialization by default.
 */
{
	Layer** layers = malloc( sizeof(Layer*) * 2 * n_layers );
	Tensor** activations = malloc( sizeof(Tensor *) * n_layers);
	Tensor** derivatives = malloc( sizeof(Tensor *) * n_layers);

	layers[0] =
		create_dense_layer(in_dim,
				   hidden_dim,
				   &initializer_symmetric_uniform);

	unsigned int rank = 2;
	unsigned int *shape = malloc( sizeof(unsigned int) * rank );
	shape[0] = in_dim;
	shape[1] = hidden_dim;
			
	activations[0] = new_tensor(rank, shape);
	derivatives[0] = new_tensor(rank, shape);

	shape[0] = hidden_dim;

	unsigned int ii;
	for (ii = 1; ii < 2 * n_layers - 1; ii += 2) {
		layers[ii] =
			create_activation_layer(hidden_activ_type);
		layers[ii + 1] =
			create_dense_layer(hidden_dim,
					   hidden_dim,
					   &initializer_symmetric_uniform);
		activations[(ii - 1)/2] = new_tensor(rank, shape);
		derivatives[(ii - 1)/2] = new_tensor(rank, shape);
	}

	layers[2 * n_layers - 1] =
		create_activation_layer(output_activ_type);

	shape[1] = out_dim;
	activations[n_layers - 1] = new_tensor(rank, shape);
	derivatives[n_layers - 1] = new_tensor(rank, shape);

	Sequential* net = create_sequential_net(2 * n_layers,
						layers);
	net->activs = activations;
	net->derivs = derivatives;
	net->optimizer = optimizer;

	return net;
}

Tensor*
feed_forward (Sequential* seq, unsigned int training, Tensor *input)
/**
 * training: 0 or 1. 0 means not training, 1 means training.
 * activations: pointer to a Tensor**. Allocated by the function.
 */
{
	if (!input) {
		// TODO: Error flags
		printf("Must give non-null tensor-input\n");
		return NULL;
	}
	
	Tensor* output, *tmp;

	output = tensor_copy(input);

	Layer* lptr = NULL;
	
	unsigned int ii;
	for (ii = 0; ii < seq->n_layers; ii += 1) {
		lptr = seq->layers[ii];
		tmp = evaluate_layer_not_training(lptr, output);
		
		if (training) {
			_tensor_map_subroutine(tmp,
					       seq->activs[ii],
					       get_activ(lptr->activ));
			
			_tensor_map_subroutine(tmp,
					       seq->derivs[ii],
					       get_deriv(lptr->activ));
		}
		
		free(output);
		output = tmp;
		free(tmp);
	}

	return NULL;
}

void
back_prop (Sequential* net, Tensor* loss, Optimizer* opt,
	   Tensor** activations, Tensor** derivatives,
	   Tensor** deltas)
/*
 * Assumes that all of the inputs are allocated!!
 */
{
	// TODO: Check that the loss vector is the correct size

	// We run backwards through the situation

	Tensor* tmp = NULL;
	unsigned int ii, ind;
	ind = net->n_layers - 1;

	_tensor_matmul(derivatives[ind], loss, deltas[ind]);
	
	for (ii = 0; ii < net->n_layers; ii += 1) {
		// every other layer is an activation layer!
		ind -= 1;
		// delta[-1] @ layer.weights.T
		// Implement matmul transposed
		tmp = new_tensor(
		
	}
	return;
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
