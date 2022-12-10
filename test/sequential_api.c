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

	Optimizer* opt = build_optimizer(OPT_SGD, 0.1, NULL);

	Sequential* model =
		create_sequential_net_basic(n_layers,
					    hidden_dim,
					    in_dim,
					    out_dim,
					    ACT_TANH,
					    ACT_SIGMOID,
					    opt);
	
	unsigned int batch_size = 64;
					    
	// Create the input tensor
	unsigned int shape[2] = {0, 0};
	shape[0] = batch_size;
	shape[1] = in_dim;
	
	Tensor* input = new_tensor(2, shape);
	initialize_tensor(input, &initializer_symmetric_uniform);

	feed_forward(model, 1, input);

	/* printf("random: %.3f\n", input->data[0]); */

	// We can now train the network

	unsigned int epochs = 10;
	
	return 1;
}
