// Generated with SOL v0.7.0rc9
#include "sol_monocular.h"
#include <stdlib.h>
#ifdef __cplusplus
	#include <iostream>
#else
	#include <stdio.h>
#endif

#ifdef __cplusplus
	using std::rand;
	using std::srand;
#endif

int main(int argc, char** argv) {
	
	int64_t vdims[0]; // do not change! (must be int64_t array of size 0)
	
	
	// Define and Initialize Inputs---------------------------------------------
	sol_f32 *in__input_1 = (sol_f32*) malloc(1ll * 256ll * 256ll * 3ll *  sizeof(sol_f32));
	
	// Define and Initialize Outputs---------------------------------------------
	sol_f32 *out__0 = (sol_f32*) malloc(1ll * 128ll * 128ll * 1ll *  sizeof(sol_f32));
	
	// Generate Random Input---------------------------------------------
	srand(0);
	for(size_t n = 0; n < 1ll * 256ll * 256ll * 3ll * 1; n++)
		in__input_1[n] = rand()/(sol_f32)RAND_MAX;
	
	for(size_t n = 0; n < 1ll * 128ll * 128ll * 1ll * 1; n++)
		out__0[n] = 0;
	
	// Call generated library---------------------------------------------
	sol_monocular_init(); // optional, reads parameters and moves them to device if necessary
	sol_monocular_set_seed(314159); // optional
	sol_predict(in__input_1, out__0, vdims);
	
	
	// Checking Results Example - Print Max, assuming first dimension is batch
	int max_index = -1;
	sol_f32 max_value = -9999999999;
	size_t data_per_batch = 0;
	
	// Output #0
	data_per_batch = (1ll * 128ll * 128ll * 1ll * 1)/1ll;
	for(int i = 0; i < 1ll; i++){
		max_index = -1;
		max_value = -9999999999;
		for(size_t n = i * data_per_batch; n < (i+1) * data_per_batch; n++){
			if(out__0[n] > max_value){
				max_value  = out__0[n];
				max_index = n - i * data_per_batch;
			}
		}
		#ifdef __cplusplus
			std::cout << "Max_V: " << max_value << std::endl;
			std::cout << "Max_I: " << max_index << std::endl;
		#else
			printf("Max_V: %f\n", max_value);
			printf("Max_I: %d\n", max_index);
		#endif
	}

	sol_monocular_free(); // frees allocated parameters by lib
	
	// free all memory allocated by this example
	free(in__input_1);
	free(out__0);
	
	return 0;
}
