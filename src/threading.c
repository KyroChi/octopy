#ifdef MULTI_THREADING
#include <pthread.h>
#include <stdlib.h>

#include "threading.h"

thread_scheduler_s*
new_thread_scheduler (unsigned int size, size_t item_size)
{
	if ( size > MAXIMUM_THREADS ) {
		// TODO: Set error flags
		return NULL;
	}

	thread_scheduler_s* sch = malloc( sizeof(sch) );
	if ( !sch ) {
		// TODO: Set error flags
		return NULL;
	}
	
	sch->size = size;
	sch->index = size - 1;
	sch->av = malloc ( sizeof(void *) * size );

	unsigned int ii;
	for (ii = 0; ii < size; ii += 1) {
		sch->av[ii] = malloc( item_size );
		if ( !sch->av[ii] ) {
			// TODO: Set error flags
			return NULL;
		}
	}

	return sch;
}

void
free_thread_scheduler (thread_scheduler_s* sch)
{
	unsigned int ii;
	for (ii = 0; ii < sch->size; ii += 1) {
		free(sch->av[ii]);
	}

	free(sch->av);
	free(sch);

	return;
}

int
thread_available (thread_scheduler_s* sch)
{
	if ( sch->index < 0 ) {
		// index == -1 means stack is empty
		return 0;
	} else {
		return 1;
	}
}

int
thread_scheduler_full (thread_scheduler_s* sch)
{
	if ( sch->index == sch->size - 1 ) {
		return 1;
	} else {
		return 0;
	}
}

void*
thread_scheduler_pop (thread_scheduler_s* sch)
{
	if ( sch->index >= 0 ) {
	        void* ptr = sch->av[sch->index];
		sch->index -= 1;
		return ptr;
	} else {
		// index == -1 means stack is empty
		return NULL;
	}
}

int
thread_scheduler_push (thread_scheduler_s* sch, void* ptr)
{
	if ( sch->index == sch->size - 1 ) {
		// stack is full
		return -1;
	} else {
		sch->index += 1;
		sch->av[sch->index] = ptr;
		return 0;
	}
}

#endif
