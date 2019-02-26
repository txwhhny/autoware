#include "include/utilities.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/sort.h>


void GUtilities::inclusiveScan(int *input, int ele_num)
{
	thrust::device_ptr<int> dev_ptr(input);

	thrust::inclusive_scan(dev_ptr, dev_ptr + ele_num, dev_ptr);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void GUtilities::exclusiveScan(int *input, int ele_num)
{
	exclusiveScan<int>(input, ele_num);
}

void GUtilities::exclusiveScan(int *input, int ele_num, int *sum)
{
	exclusiveScan<int>(input, ele_num, sum);
}

void GUtilities::exclusiveScan(long long int *input, int ele_num, long long int *sum)
{
	exclusiveScan<long long int>(input, ele_num, sum);
}

void GUtilities::exclusiveScan(unsigned long long int *input, int ele_num, unsigned long long int *sum)
{
	exclusiveScan<unsigned long long int>(input, ele_num, sum);
}

template <typename T>
void GUtilities::exclusiveScan(T *input, int ele_num, T *sum)
{
	thrust::device_ptr<T> dev_ptr(input);

	thrust::exclusive_scan(dev_ptr, dev_ptr + ele_num, dev_ptr);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	*sum = *(dev_ptr + ele_num - 1);
}

template <typename T>
void GUtilities::exclusiveScan(T *input, int ele_num)
{
	thrust::device_ptr<T> dev_ptr(input);

	thrust::exclusive_scan(dev_ptr, dev_ptr + ele_num, dev_ptr);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void GUtilities::sort(int *input, int ele_num)
{
	thrust::device_ptr<int> dev_ptr(input);

	thrust::sort(dev_ptr, dev_ptr + ele_num);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void GUtilities::sortByKey(int *key, int *value, int ele_num)
{
	thrust::device_ptr<int> dev_key(key);
	thrust::device_ptr<int> dev_val(value);

	thrust::sort_by_key(dev_key, dev_key + ele_num, dev_val);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}
