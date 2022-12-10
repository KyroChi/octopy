/**
 * Some notes:
 * 1. Do not use my random number implementations for cryptography or
 *    security applications, I do not think they are strong enough.
 * 2. We do not need cryptography grade RNGs or PRNGs for the use
 *    case of machine learning.
 * 3. These are primarily used for initializing network weights and
 *    doing stochastic optimization.
 */

#include <time.h>

#include "mtwister/mtwister.h"
#include "random.h"

unsigned int _RNG_SEEDED = 0;
time_t _SEED;
MTRand _RAND_SEED = {0.0, 0};

void
_set_seed (void)
/* Sets seed if it isn't already set. Seed is set to system time 
 * upon first call to a RNG when the seed isn't set yet. */
{
	time( &_SEED);
	_RAND_SEED = seedRand( _SEED );
	_RNG_SEEDED = 1;
	
	return;
}

float
_rand_uniform ()
{
	if ( !_RNG_SEEDED ) { _set_seed(); }	
	return genRand( &_RAND_SEED );
}

float
rand_uniform (float min, float max)
/**
 * Wraps _rand_uniform in a more user friendly api.
 * Returns random numbers in a uniform distribution from [min, max].
 */
{
	if (min >= max) {
		// TODO: Set error flags
		return (min - max) * _rand_uniform() + max;
	}
	return (max - min) * _rand_uniform() + min;
}

float
_rand_normal()
/**
 * Returns a random number from NORMAL(0, 1).
 */
{
	if ( !_RNG_SEEDED ) { _set_seed(); }
	return 0.0;
}

float
rand_normal (float mu, float sigma)
/**
 * Return a random number from NORMAL(mu, sigma).
 */
{
	return 0.0;
}
