void
array_cpy_float (float *A, float *B, unsigned int length)
/**
 * Copy contents of A into B. B must be allocated.
 */
{
	unsigned int ii;
	for (ii = 0; ii < length; ii += 1) {
		B[ii] = A[ii];
	}

	return;
}

void
array_cpy_uint (unsigned int *A, unsigned int* B, unsigned int length)
{
	unsigned int ii;
	for (ii = 0; ii < length; ii += 1) {
		B[ii] = A[ii];
	}

	return;
}
