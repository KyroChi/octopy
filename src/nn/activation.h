#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

typedef enum {
	ACT_IDENTITY,
	ACT_TANH,
	ACT_RELU,
	ACT_LEAKY_RELU,
	ACT_SIGMOID,
	ACT_NONE,
} activation_t;

typedef struct {
	activation_t activ;
	float (*func) (float);   // The activation function
	float (*func_d) (float); // It's derivative
} Activation;

float (*get_activ(activation_t activ))(float);
float (*get_deriv(activation_t activ))(float);

#endif
