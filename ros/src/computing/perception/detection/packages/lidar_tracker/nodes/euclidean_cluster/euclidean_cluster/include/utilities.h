#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

class GUtilities {
public:
	static void inclusiveScan(int *input, int ele_num);

	static void exclusiveScan(int *input, int ele_num);

	static void exclusiveScan(int *input, int ele_num, int *sum);

	static void exclusiveScan(long long int *input, int ele_num, long long int *sum);

	static void exclusiveScan(unsigned long long int *input, int ele_num, unsigned long long int *sum);

	template <typename T = int>
	static void exclusiveScan(T *input, int ele_num, T *sum);

	template <typename T = int>
	static void exclusiveScan(T *input, int ele_num);

	static void sort(int *input, int ele_num);

	static void sortByKey(int *key, int *value, int ele_num);
};

inline void gassert(cudaError_t err_code, const char *file, int line)
{
	if (err_code != cudaSuccess) {
		fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(err_code), file, line);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

#define checkCudaErrors(err_code) gassert(err_code, __FILE__, __LINE__)

#ifndef timeDiff
#define timeDiff(start, end) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec))
#endif

#endif
