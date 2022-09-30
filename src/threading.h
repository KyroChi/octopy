#ifndef THREADING_H
#define THREADING_H

#include <pthread.h>

/* Throw errors if you use multi-threading options without compiling
 * with threading */
#ifdef MULTI_THREADING
static unsigned int MAXIMUM_THREADS = 9;

typedef struct {
	void **av;
	int size;
	int index;
} thread_scheduler_s;

thread_scheduler_s* new_thread_scheduler (unsigned int, size_t);
void free_thread_scheduler (thread_scheduler_s*);
int thread_available (thread_scheduler_s*);
int thread_scheduler_full (thread_scheduler_s*);
void* thread_scheduler_pop (thread_scheduler_s*);
int thread_scheduler_push (thread_scheduler_s*, void*);

#endif
#endif
