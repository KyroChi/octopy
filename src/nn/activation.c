#include <math.h>
#include "activation.h"

static float activation_identity_func (float a) { return a; }
static float activation_identity_func_d (float a) { return 1.0; }

static Activation activation_identity = {
	.activ = ACT_IDENTITY,
	.func = activation_identity_func,
	.func_d = activation_identity_func_d,
};

static float activation_tanh_func (float a) { return tanh(a); }
static float activation_tanh_func_d (float a)
{ return 1 - pow(tanh(a), 2); }

static Activation activation_tanh = {
	.activ = ACT_TANH,
	.func = activation_tanh_func,
	.func_d = activation_tanh_func_d,
};

static float activation_sigmoid_func (float a)
{ return exp(a) / (exp(a) - 1); }
static float activation_sigmoid_func_d (float a)
{ return exp(-a) / pow(exp(-a) + 1, 2); }

static Activation activation_sigmoid = {
	.activ = ACT_IDENTITY,
	.func = activation_sigmoid_func,
	.func_d = activation_sigmoid_func_d,
};

float (*get_activ(activation_t activ))(float)
{
	switch (activ) {
	case ACT_NONE:
		return activation_identity.func;
	case ACT_IDENTITY:
		return activation_identity.func;
	case ACT_TANH:
		return activation_tanh.func;
	case ACT_SIGMOID:
		return activation_sigmoid.func;
	}

	// TODO: Set an error flag for fall-through
	return activation_identity.func;
}

float (*get_deriv(activation_t activ))(float)
{
	switch (activ) {
	case ACT_NONE:
		return activation_identity.func_d;
	case ACT_IDENTITY:
		return activation_identity.func_d;
	case ACT_TANH:
		return activation_tanh.func_d;
	case ACT_SIGMOID:
		return activation_sigmoid.func_d;
	}

	// TODO: Set an error flag for fall-through
	return activation_identity.func_d;
}
