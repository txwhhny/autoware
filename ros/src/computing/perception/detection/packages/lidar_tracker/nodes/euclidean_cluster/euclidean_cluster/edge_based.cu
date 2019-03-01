#include "include/euclidean_cluster.h"
#include "include/utilities.h"
#include <cuda.h>


extern __shared__ float local_buff[];

// Build edge set
__global__ void edgeCount(float *x, float *y, float *z, int point_num, long long *edge_count, float threshold)
{
	float *local_x = local_buff;
	float *local_y = local_x + blockDim.x;
	float *local_z = local_y + blockDim.x;
	int pid;
	int last_point = (point_num / blockDim.x) * blockDim.x;	// Exclude the last block
	float dist;

	for (pid = threadIdx.x + blockIdx.x * blockDim.x; pid < last_point; pid += blockDim.x * gridDim.x) {
		float tmp_x = x[pid];
		float tmp_y = y[pid];
		float tmp_z = z[pid];
		long long count = 0;

		int block_id;

		for (block_id = blockIdx.x * blockDim.x; block_id + blockDim.x < point_num; block_id += blockDim.x) {
			local_x[threadIdx.x] = x[block_id + threadIdx.x];
			local_y[threadIdx.x] = y[block_id + threadIdx.x];
			local_z[threadIdx.x] = z[block_id + threadIdx.x];
			__syncthreads();

			for (int i = 0; i < blockDim.x; i++) {
				dist = norm3df(tmp_x - local_x[i], tmp_y - local_y[i], tmp_z - local_z[i]);
				count += (i + block_id > pid && dist < threshold) ? 1 : 0;
			}
			__syncthreads();
		}

		__syncthreads();

		// Compare with last block
		if (threadIdx.x < point_num - block_id) {
			local_x[threadIdx.x] = x[block_id + threadIdx.x];
			local_y[threadIdx.x] = y[block_id + threadIdx.x];
			local_z[threadIdx.x] = z[block_id + threadIdx.x];
		}
		__syncthreads();

		for (int i = 0; i < point_num - block_id; i++) {
			dist = norm3df(tmp_x - local_x[i], tmp_y - local_y[i], tmp_z - local_z[i]);
			count += (i + block_id > pid && dist < threshold) ? 1 : 0;
		}

		__syncthreads();

		edge_count[pid] = count;
	}
	__syncthreads();


	// Handle last block
	if (pid >= last_point && pid < point_num) {
		int count = 0;
		float tmp_x, tmp_y, tmp_z;

		if (pid < point_num) {
			tmp_x = x[pid];
			tmp_y = y[pid];
			tmp_z = z[pid];
		}

		int block_id = blockIdx.x * blockDim.x;

		__syncthreads();

		if (pid < point_num) {
			local_x[threadIdx.x] = x[pid];
			local_y[threadIdx.x] = y[pid];
			local_z[threadIdx.x] = z[pid];
			__syncthreads();

			for (int i = 0; i < point_num - block_id; i++) {
				dist = norm3df(tmp_x - local_x[i], tmp_y - local_y[i], tmp_z - local_z[i]);
				count += (i + block_id > pid && dist < threshold) ? 1 : 0;
			}
			__syncthreads();

			edge_count[pid] = count;
		}
	}

	__syncthreads();

}


__global__ void buildEdgeSet(float *x, float *y, float *z, int point_num, long long *edge_count, int2 *edge_set, float threshold, long long edge_num)
{
	float *local_x = local_buff;
	float *local_y = local_x + blockDim.x;
	float *local_z = local_y + blockDim.x;
	int pid;
	int last_point = (point_num / blockDim.x) * blockDim.x;
	int2 new_edge;

	for (pid = threadIdx.x + blockIdx.x * blockDim.x; pid < last_point; pid += blockDim.x * gridDim.x) {
		long long writing_location = edge_count[pid];
		float tmp_x = x[pid];
		float tmp_y = y[pid];
		float tmp_z = z[pid];

		int block_id;

		new_edge.x = pid;

		for (block_id = blockIdx.x * blockDim.x; block_id + blockDim.x < point_num; block_id += blockDim.x) {
			local_x[threadIdx.x] = x[block_id + threadIdx.x];
			local_y[threadIdx.x] = y[block_id + threadIdx.x];
			local_z[threadIdx.x] = z[block_id + threadIdx.x];
			__syncthreads();

			for (int i = 0; i < blockDim.x; i++) {
				if (i + block_id > pid && norm3df(tmp_x - local_x[i], tmp_y - local_y[i], tmp_z - local_z[i]) < threshold) {
					new_edge.y = i + block_id;

					edge_set[writing_location++] = new_edge;
				}
			}
			__syncthreads();
		}
		__syncthreads();


		if (threadIdx.x < point_num - block_id) {
			local_x[threadIdx.x] = x[block_id + threadIdx.x];
			local_y[threadIdx.x] = y[block_id + threadIdx.x];
			local_z[threadIdx.x] = z[block_id + threadIdx.x];
		}
		__syncthreads();

		for (int i = 0; i < point_num - block_id; i++) {
			if (i + block_id > pid && norm3df(tmp_x - local_x[i], tmp_y - local_y[i], tmp_z - local_z[i]) < threshold) {
				new_edge.y = i + block_id;
				edge_set[writing_location++] = new_edge;
			}
		}
		__syncthreads();

	}

	if (pid >= last_point) {
		float tmp_x, tmp_y, tmp_z;
		int writing_location;

		if (pid < point_num) {
			new_edge.x = pid;
			tmp_x = x[pid];
			tmp_y = y[pid];
			tmp_z = z[pid];
			writing_location = edge_count[pid];


			int block_id = blockIdx.x * blockDim.x;

			local_x[threadIdx.x] = x[pid];
			local_y[threadIdx.x] = y[pid];
			local_z[threadIdx.x] = z[pid];
			__syncthreads();

			for (int i = 0; i < point_num - block_id; i++) {
				if (i + block_id > pid && norm3df(tmp_x - local_x[i], tmp_y - local_y[i], tmp_z - local_z[i]) < threshold) {
					new_edge.y = i + block_id;
					edge_set[writing_location++] = new_edge;
				}
			}
		}
	}
}



__global__ void clustering(int2 *edge_set, int size, int *cluster_name, bool *changed)
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

__global__ void clusterCount(int *cluster_name, int *count, int point_num)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < point_num; i += blockDim.x * gridDim.x) {
		count[cluster_name[i]] = 1;
	}
}

void GpuEuclideanCluster2::extractClusters3()
{
	long long total_time, build_graph, clustering_time;
	int iteration_num;

	extractClusters3(total_time, build_graph, clustering_time, iteration_num);
}

void GpuEuclideanCluster2::extractClusters3(long long &total_time, long long &build_graph, long long &clustering_time, int &iteration_num)
{
#ifdef DEBUG_
	std::cout << "EDGE-BASED 1: compare every pairwise distance" << std::endl;
#endif

	total_time = build_graph = clustering_time = 0;
	iteration_num = 0;

	struct timeval start, end;

	initClusters();

	int block_x, grid_x;

	block_x = (point_num_ > block_size_x_) ? block_size_x_ : point_num_;
	grid_x = (point_num_ - 1) / block_x + 1;


	long long *edge_count;

	gettimeofday(&start, NULL);
	checkCudaErrors(cudaMalloc(&edge_count, sizeof(long long) * (point_num_ + 1)));
	checkCudaErrors(cudaMemset(edge_count, 0x00, sizeof(long long) * (point_num_ + 1)));


	edgeCount<<<grid_x, block_x, sizeof(float) * block_size_x_ * 3 + sizeof(long long) * block_size_x_>>>(x_, y_, z_, point_num_, edge_count, threshold_);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	long long edge_num;

	gettimeofday(&end, NULL);

	build_graph += timeDiff(start, end);
	total_time += timeDiff(start, end);

#ifdef DEBUG_
	std::cout << "Count Edge Set = " << timeDiff(start, end) << std::endl;
#endif

	GUtilities::exclusiveScan(edge_count, point_num_ + 1, &edge_num);

	if (edge_num == 0) {
		checkCudaErrors(cudaFree(edge_count));
		cluster_num_ = 0;
		return;
	}

	int2 *edge_set;

	gettimeofday(&start, NULL);

	checkCudaErrors(cudaMalloc(&edge_set, sizeof(int2) * edge_num));

	buildEdgeSet<<<grid_x, block_x, sizeof(float) * block_size_x_ * 3>>>(x_, y_, z_, point_num_, edge_count, edge_set, threshold_, edge_num);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	bool *changed;
	bool hchanged;

	checkCudaErrors(cudaMalloc(&changed, sizeof(bool)));

	block_x = (edge_num > block_size_x_) ? block_size_x_ : edge_num;
	grid_x = (edge_num - 1) / block_x + 1;

	int itr = 0;

	gettimeofday(&end, NULL);

	build_graph += timeDiff(start, end);
	total_time += timeDiff(start, end);

#ifdef DEBUG_
	std::cout << "Build Edge Set = " << timeDiff(start, end) << std::endl;
#endif

	gettimeofday(&start, NULL);
	do {
		hchanged = false;

		checkCudaErrors(cudaMemcpy(changed, &hchanged, sizeof(bool), cudaMemcpyHostToDevice));

		clustering<<<grid_x, block_x>>>(edge_set, edge_num, cluster_name_, changed);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaMemcpy(&hchanged, changed, sizeof(bool), cudaMemcpyDeviceToHost));
		itr++;
	} while (hchanged);

	gettimeofday(&end, NULL);

	clustering_time += timeDiff(start, end);
	total_time += timeDiff(start, end);
	iteration_num = itr;

#ifdef DEBUG_
	std::cout << "Iteration time = " << timeDiff(start, end) << " itr_num = " << itr << std::endl;
#endif

	int *count;

	gettimeofday(&start, NULL);
	checkCudaErrors(cudaMalloc(&count, sizeof(int) * (point_num_ + 1)));
	checkCudaErrors(cudaMemset(count, 0, sizeof(int) * (point_num_ + 1)));

	block_x = (point_num_ > block_size_x_) ? block_size_x_ : point_num_;
	grid_x = (point_num_ - 1) / block_x + 1;

	clusterCount<<<grid_x, block_x>>>(cluster_name_, count, point_num_);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	GUtilities::exclusiveScan(count, point_num_ + 1, &cluster_num_);

	renamingClusters(cluster_name_, count, point_num_);

	checkCudaErrors(cudaMemcpy(cluster_name_host_, cluster_name_, sizeof(int) * point_num_, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(edge_count));
	checkCudaErrors(cudaFree(edge_set));
	checkCudaErrors(cudaFree(changed));
	checkCudaErrors(cudaFree(count));
	gettimeofday(&end, NULL);

	total_time += timeDiff(start, end);

#ifdef DEBUG_
	std::cout << "FINAL CLUSTER NUM = " << cluster_num_ << std::endl << std::endl;
#endif
}
