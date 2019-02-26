#include "include/euclidean_cluster.h"
#include "include/utilities.h"
#include "include/voxel_grid.h"
#include <cuda.h>

#define TEST_VERTEX_ 1

__global__ void frontierInitialize2(int *frontier_array, int point_num)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < point_num; i += blockDim.x * gridDim.x) {
		frontier_array[i] = 1;
	}
}

__global__ void vertexBasedClustering(int *adjacent_list_loc, int *adjacent_list, int point_num, int *cluster_name, int *frontier_array1, int *frontier_array2, bool *changed)
{
	__shared__ bool schanged;

	if (threadIdx.x == 0)
		schanged = false;
	__syncthreads();

	for (int pid = threadIdx.x + blockIdx.x * blockDim.x; pid < point_num; pid += blockDim.x * gridDim.x) {
		if (frontier_array1[pid] == 1) {
			frontier_array1[pid] = 0;
			int cname = cluster_name[pid];
			bool c = false;
			int start = adjacent_list_loc[pid];
			int end = adjacent_list_loc[pid + 1];

			// Iterate through neighbors' ids
			for (int i = start; i < end; i++) {
				int nid = adjacent_list[i];
				int nname = cluster_name[nid];
				if (cname < nname) {
					atomicMin(cluster_name + nid, cname);
					frontier_array2[nid] = 1;
					schanged = true;
					//*changed = true;
				} else if (cname > nname) {
					cname = nname;
					c = true;
				}
			}

			if (c) {
				atomicMin(cluster_name + pid, cname);
				frontier_array2[pid] = 1;
				schanged = true;
				//*changed = true;
			}
		}
	}
	__syncthreads();

	if (threadIdx.x == 0 && schanged)
		*changed = true;
}

/* Iterate through the list of remaining clusters and mark the corresponding
 * location on cluster location array by 1
 */
__global__ void clusterMark3(int *cluster_list, int *cluster_location, int cluster_num)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = idx; i < cluster_num; i += blockDim.x * gridDim.x) {
		cluster_location[cluster_list[i]] = 1;
	}
}


void GpuEuclideanCluster2::extractClusters6()
{
	long long total_time, build_graph, clustering_time;
	int iteration_num;

	extractClusters6(total_time, build_graph, clustering_time, iteration_num);
}



void GpuEuclideanCluster2::extractClusters6(long long &total_time, long long &build_graph, long long &clustering_time, int &iteration_num)
{
	std::cout << "VERTEX-BASED 2: Use octree" << std::endl;
	total_time = build_graph = clustering_time = 0;
	iteration_num = 0;

#ifdef TEST_VERTEX_
	struct timeval start, end;

	gettimeofday(&start, NULL);
#endif

	initClusters();

	GVoxelGrid new_grid(x_, y_, z_, point_num_, threshold_, threshold_, threshold_);

	int *adjacent_count = NULL;
	int *adjacent_list = NULL;
	int adjacent_list_size = 0;

	new_grid.createAdjacentList(&adjacent_count, &adjacent_list, &adjacent_list_size, threshold_);

	if (adjacent_list_size == 0) {
		cluster_num_ = point_num_;

		return;
	}

	int block_x = (point_num_ < block_size_x_) ? point_num_ : block_size_x_;
	int grid_x = (point_num_ - 1) / block_x + 1;

#ifdef TEST_VERTEX_
	gettimeofday(&end, NULL);

	build_graph += timeDiff(start, end);
	total_time += timeDiff(start, end);
	std::cout << "Build graph = " << timeDiff(start, end) << std::endl;
#endif

	bool *changed;

	bool hchanged;
	checkCudaErrors(cudaMalloc(&changed, sizeof(bool)));

	int *frontier_array1, *frontier_array2;

#ifdef TEST_VERTEX_
	gettimeofday(&start, NULL);
#endif

	checkCudaErrors(cudaMalloc(&frontier_array1, sizeof(int) * point_num_));
	checkCudaErrors(cudaMalloc(&frontier_array2, sizeof(int) * point_num_));

	frontierInitialize2<<<grid_x, block_x>>>(frontier_array1, point_num_);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemset(frontier_array2, 0, sizeof(int) * point_num_));
	checkCudaErrors(cudaDeviceSynchronize());

	int itr = 0;

	do {
		hchanged = false;
		checkCudaErrors(cudaMemcpy(changed, &hchanged, sizeof(bool), cudaMemcpyHostToDevice));

		vertexBasedClustering<<<grid_x, block_x>>>(adjacent_count, adjacent_list, point_num_, cluster_name_, frontier_array1, frontier_array2, changed);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		int *tmp;

		tmp = frontier_array1;
		frontier_array1 = frontier_array2;
		frontier_array2 = tmp;

		checkCudaErrors(cudaMemcpy(&hchanged, changed, sizeof(bool), cudaMemcpyDeviceToHost));

		itr++;
	} while (hchanged);



#ifdef TEST_VERTEX_
	gettimeofday(&end, NULL);

	clustering_time += timeDiff(start, end);
	total_time += timeDiff(start, end);
	std::cout << "Iteration = " << timeDiff(start, end) << " itr_num = " << itr << std::endl;
	iteration_num = itr;
#endif


	// renaming clusters
	int *cluster_location;

	gettimeofday(&start, NULL);
	checkCudaErrors(cudaMalloc(&cluster_location, sizeof(int) * (point_num_ + 1)));
	checkCudaErrors(cudaMemset(cluster_location, 0, sizeof(int) * (point_num_ + 1)));

	clusterMark3<<<grid_x, block_x>>>(cluster_name_, cluster_location, point_num_);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	GUtilities::exclusiveScan(cluster_location, point_num_ + 1, &cluster_num_);

	renamingClusters(cluster_name_, cluster_location, point_num_);

	checkCudaErrors(cudaMemcpy(cluster_name_host_, cluster_name_, sizeof(int) * point_num_, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(adjacent_count));
	checkCudaErrors(cudaFree(adjacent_list));
	checkCudaErrors(cudaFree(frontier_array1));
	checkCudaErrors(cudaFree(frontier_array2));
	checkCudaErrors(cudaFree(changed));
	checkCudaErrors(cudaFree(cluster_location));
	gettimeofday(&end, NULL);

	total_time += timeDiff(start, end);

	std::cout << "FINAL CLUSTER NUM = " << cluster_num_ << std::endl << std::endl;
}
