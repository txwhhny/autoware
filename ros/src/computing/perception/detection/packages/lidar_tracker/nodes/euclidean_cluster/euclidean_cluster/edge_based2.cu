#include "include/euclidean_cluster.h"
#include "include/utilities.h"
#include <cuda.h>
#include "include/voxel_grid.h"

__global__ void markValidPoints0(int *starting_neighbor_ids, int point_num, int *mark)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < point_num - 1; i += blockDim.x * gridDim.x) {
		if (starting_neighbor_ids[i] < starting_neighbor_ids[i + 1]) {
			mark[starting_neighbor_ids[i]] = i;
		}
	}
}


__global__ void markValidPoints1(int *starting_neighbor_ids, int point_num, int *mark)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x + 1; i < point_num; i += blockDim.x * gridDim.x) {
		if (starting_neighbor_ids[i] > starting_neighbor_ids[i - 1]) {
			mark[starting_neighbor_ids[i]] -= (i - 1);
		}
	}
}

__global__ void graphBuild(int *valid_point_ids, int *neighbor_ids, int2 *edge_set, int edge_num)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < edge_num; i += blockDim.x * gridDim.x) {
		int2 val;

		val.x = valid_point_ids[i];
		val.y = neighbor_ids[i];

		edge_set[i] = val;
	}
}


__global__ void edgeBasedClustering(int2 *edge_set, int size, int *cluster_name, bool *changed)
{
	__shared__ bool schanged;

	if (threadIdx.x == 0)
		schanged = false;
	__syncthreads();

	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x) {
		int2 cur_edge = edge_set[i];
		int x = cur_edge.x;
		int y = cur_edge.y;

		int x_name = cluster_name[x];
		int y_name = cluster_name[y];
		int *changed_addr = NULL;
		int change_name;

		if (x_name < y_name) {
			changed_addr = cluster_name + y;
			change_name = x_name;
		} else if (x_name > y_name) {
			changed_addr = cluster_name + x;
			change_name = y_name;
		}

		if (changed_addr != NULL) {
			atomicMin(changed_addr, change_name);
			schanged = true;
		}
		__syncthreads();
	}

	__syncthreads();

	if (threadIdx.x == 0 && schanged)
		*changed = true;
}

__global__ void clusterCount2(int *cluster_name, int *count, int point_num)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < point_num; i += blockDim.x * gridDim.x) {
		count[cluster_name[i]] = 1;
	}
}


void GpuEuclideanCluster2::extractClusters5()
{
	long long total_time, build_graph, clustering_time;
	int iteration_num;

	extractClusters5(total_time, build_graph, clustering_time, iteration_num);
}

void GpuEuclideanCluster2::extractClusters5(long long &total_time, long long &build_graph, long long &clustering_time, int &iteration_num)
{
	std::cout << "EDGE-BASED 2 METHOD point_num = " << point_num_ << std::endl;

	initClusters();

	total_time = build_graph = clustering_time = 0;
	iteration_num = 0;

	struct timeval start, end;

	gettimeofday(&start, NULL);

	GVoxelGrid new_grid(x_, y_, z_, point_num_, threshold_, threshold_, threshold_);

	int2 *edge_set;
	int edge_num;

	new_grid.createEdgeSet(&edge_set, &edge_num, threshold_);

	if (edge_num == 0) {
		cluster_num_ = point_num_;

		return;
	}

	gettimeofday(&end, NULL);

	build_graph += timeDiff(start, end);
	total_time += timeDiff(start, end);

	std::cout << "Build Edge Set = " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);

	bool *changed;
	bool hchanged;

	checkCudaErrors(cudaMalloc(&changed, sizeof(bool)));

	int block_x = (edge_num > block_size_x_) ? block_size_x_ : edge_num;
	int grid_x = (edge_num - 1) / block_x + 1;

	int itr = 0;

	do {
		hchanged = false;

		checkCudaErrors(cudaMemcpy(changed, &hchanged, sizeof(bool), cudaMemcpyHostToDevice));

		edgeBasedClustering<<<grid_x, block_x>>>(edge_set, edge_num, cluster_name_, changed);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaMemcpy(&hchanged, changed, sizeof(bool), cudaMemcpyDeviceToHost));
		itr++;
	} while (hchanged);

	gettimeofday(&end, NULL);

	clustering_time += timeDiff(start, end);
	total_time += timeDiff(start, end);
	iteration_num = itr;
	std::cout << "Iteration time = " << timeDiff(start, end) << " itr_num = " << itr << std::endl;

	int *count;

	gettimeofday(&start, NULL);
	checkCudaErrors(cudaMalloc(&count, sizeof(int) * (point_num_ + 1)));
	checkCudaErrors(cudaMemset(count, 0, sizeof(int) * (point_num_ + 1)));

	block_x = (point_num_ > block_size_x_) ? block_size_x_ : point_num_;
	grid_x = (point_num_ - 1) / block_x + 1;

	clusterCount2<<<grid_x, block_x>>>(cluster_name_, count, point_num_);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	GUtilities::exclusiveScan(count, point_num_ + 1, &cluster_num_);

	renamingClusters(cluster_name_, count, point_num_);

	checkCudaErrors(cudaMemcpy(cluster_name_host_, cluster_name_, sizeof(int) * point_num_, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(edge_set));
	checkCudaErrors(cudaFree(changed));
	checkCudaErrors(cudaFree(count));
	gettimeofday(&end, NULL);

	total_time += timeDiff(start, end);

	std::cout << "FINAL CLUSTER NUM = " << cluster_num_ << std::endl << std::endl;

}
