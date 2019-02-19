#include <cuda.h>

class GUtilities {
public:
	static void exclusiveScan(int *input, int ele_num);

	static void exclusiveScan(int *input, int ele_num, int *sum);

	static void exclusiveScan(long long int *input, int ele_num, long long int *sum);

	static void exclusiveScan(unsigned long long int *input, int ele_num, unsigned long long int *sum);

	template <typename T = int>
	static void exclusiveScan(T *input, int ele_num, T *sum);

	template <typename T = int>
	static void exclusiveScan(T *input, int ele_num);
};
