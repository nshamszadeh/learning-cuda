// From Introduction to High Performance Computing by D.L. Chopp
#ifdef DO_ERROR_CHECKING
    static void CHECK_ERROR(cudaError_t err, const char* file, int line) {
        if (err != cudaSuccess) {
            printf("%s in %s at line %d", cudaGetErrorString(err), file, line);
            exit(1);
        }
    }
    #define CheckError( err ) (CHECK_ERROR(err, __FILE__, __LINE__))
#else
    #define CheckError( err ) err
#endif
/*
To use this code, whenever a CUDA function is called, simply wrap it inside a function
call to CheckError, i.e., CheckError(cudaMalloc((void**)&dev_u, N * sizeof(double)));

The "function" CheckError on line 10 is not actually a function but a macro which takes an argument.
If DO_ERROR_CHECKING is defined, then when the preprocessor sees the text "CheckError(X)" it replaces it
with "CHECK_ERROR(X, __FILE__, __LINE__)". The preprocessor also identifies the strings "__FILE__" and 
"__LINE__" and will substitute the name of the current file and the line number in the file respectively.

CHECK_ERROR ~~is~~ a regular function, and if the value of the first argument is not cudaSuccess, then it prints 
an error message using cudaGetErrorString and exits the program, which is the minimum of what an error handler should do.
*/