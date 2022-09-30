#ifndef OCTOPY_HELPER_H
#define OCTOPY_HELPER_H

// Pre-processor macro to avoid 'unused parameter' in Werror, Wall.o
#define UNUSED(x) (void)(x)

void array_cpy_float (float *, float *, unsigned int);
void array_cpy_uint (unsigned int *, unsigned int *, unsigned int);

#endif 
