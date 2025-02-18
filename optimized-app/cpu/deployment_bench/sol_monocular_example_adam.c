#include "sol_monocular.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int64_t vdims[0]; // do not change! (must be int64_t array of size 0)

    // Define and Initialize Inputs
    sol_f32 *in__input_1 = (sol_f32*) malloc(1ll * 256ll * 256ll * 3ll * sizeof(sol_f32));
    
    // Define and Initialize Outputs
    sol_f32 *out__0 = (sol_f32*) malloc(1ll * 128ll * 128ll * 1ll * sizeof(sol_f32));

    // Fill input with a known good pattern (e.g., sample input provided by SOL)
    // Replace this with the actual known good input
    for (size_t n = 0; n < 1ll * 256ll * 256ll * 3ll; n++) {
        in__input_1[n] = 0.5f; // Replace with known good input
    }

    // Initialize output to 0
    for (size_t n = 0; n < 1ll * 128ll * 128ll * 1ll; n++) {
        out__0[n] = 0.0f;
    }

    // Call generated library
    sol_monocular_init(); // Initialize the library
    sol_monocular_set_seed(314159); // Optional: Set seed for reproducibility
    sol_predict(in__input_1, out__0, vdims); // Run prediction

    // Check output for valid values
    int valid_output = 1;
    for (size_t n = 0; n < 1ll * 128ll * 128ll * 1ll; n++) {
        if (out__0[n] == 0.0f || out__0[n] != out__0[n]) { // Check for 0 or NaN
            valid_output = 0;
            break;
        }
    }

    if (valid_output) {
        printf("Output is valid.\n");
    } else {
        printf("Output is invalid (contains 0 or NaN).\n");
    }

    // Free resources
    sol_monocular_free(); // Free allocated parameters by lib
    free(in__input_1);
    free(out__0);

    return 0;
}