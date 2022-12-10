#include <stdio.h>
#include <stdlib.h>

#include "../src/math/tensor.h"
#include "../src/nn/sequential.h"
#include "../src/nn/activation.h"
#include "../src/nn/optimizers.h"

int
main ()
{
	// Network hyperparameters
	unsigned int n_layers = 6;
	unsigned int hidden_dim = 30;
	unsigned int in_dim = 10;
	unsigned int out_dim = 3;

	unsigned int batch_size = 64;

	// Build the network architecture
	// It is convinient to also initialize the activation
	// and derivative arrays here
	Layer** layers = malloc( sizeof(Layer *) * 2 * n_layers);

	Tensor** activations = malloc( sizeof(Tensor *) * n_layers);
	Tensor** derivatives = malloc( sizeof(Tensor *) * n_layers);
	
	layers[0] = create_dense_layer(in_dim,
				       hidden_dim,
				       INIT_DEFAULT);

	unsigned int rank = 2;
	unsigned int *shape = malloc( sizeof(unsigned int) * rank );
	shape[0] = in_dim;
	shape[1] = hidden_dim;
			
	activations[0] = new_tensor(rank, shape);
	derivatives[0] = new_tensor(rank, shape);

	shape[0] = hidden_dim;
	
	unsigned int ii;
	for (ii = 1; ii < 2 * n_layers - 1; ii += 2) {
		layers[ii] = create_activation_layer(ACT_TANH);
		layers[ii + 1] = create_dense_layer(hidden_dim,
						    hidden_dim,
						    INIT_DEFAULT);
		activations[(ii - 1)/2] = new_tensor(rank, shape);
		derivatives[(ii - 1)/2] = new_tensor(rank, shape);
	}

	layers[2 * n_layers - 1] =
		create_activation_layer(ACT_SIGMOID);

	shape[1] = out_dim;
	activations[n_layers - 1] = new_tensor(rank, shape);
	derivatives[n_layers - 1] = new_tensor(rank, shape);

	// Create the network
	Sequential* model = create_sequential_net(2 * n_layers,
						  layers);

	// Create the input tensor
	shape[0] = batch_size;
	shape[2] = in_dim;
	
	Tensor* input = new_tensor(rank, shape);
	initialize_tensor(input, &initializer_symmetric_uniform);

	printf("random: %.3f\n", input->data[0]);

	// We can now train the network
	Tensor* new_out = feed_forward(model,
				       0,
				       input,
				       activations,
				       derivatives);

	unsigned int epochs = 10;

	Optimizer* opt = build_optimizer(OPT_SGD, 0.1, NULL);
	optimizer_update(opt, layers[0]->weights, derivatives[0]);
	
	
	return 1;
}
