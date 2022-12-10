#ifndef RANDOM_H
#define RANDOM_H

#include <time.h>

// We're using Evan Sultanik's Mersanne twister implementation untill
// I write my own: https://github.com/ESultanik/mtwister
#include "./mtwister/mtwister.h"

// Seed the RNG with the current time
// To set the RNG yourself call
// _RAND_SEED = seedRand(your_number), then set _RNG_SEEDED = 1.
// This will globally set the seed and make your program
// deterministic.
unsigned int _RNG_SEEDED;
time_t _SEED;
MTRand _RAND_SEED;

float _rand_uniform ();
float _rand_normal ();

float rand_uniform (float, float);
float rand_normal (float, float);

#endif
