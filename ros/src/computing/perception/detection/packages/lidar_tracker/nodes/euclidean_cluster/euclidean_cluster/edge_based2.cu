#include "include/euclidean_cluster.h"
#include "include/utilities.h"
#include <cuda.h>

extern __shared__ float local_buff[];

// Search for the furthest neighbor
__global__ void sampleGraphBuild(float *x, float *y, float *z, int point_num, int2 *edge_set, float threshold)
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
		float min_dist = threshold;
		int min_pid = -1;

		int block_id;

		for (block_id = blockIdx.x * blockDim.x; block_id + blockDim.x < point_num; block_id += blockDim.x) {
			local_x[threadIdx.x] = x[block_id + threadIdx.x];
			local_y[threadIdx.x] = y[block_id + threadIdx.x];
			local_z[threadIdx.x] = z[block_id + threadIdx.x];
			__syncthreads();

			for (int i = 0; i < blockDim.x; i++) {
				dist = norm3df(tmp_x - local_x[i], tmp_y - local_y[i], tmp_z - local_z[i]);

				if (i + block_id > pid && dist < threshold && min_dist > dist) {
					min_dist = dist;
					min_pid = i + block_id;
				}
				__syncthreads();
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

			if (i + block_id > pid && dist < threshold && min_dist > dist) {
				min_dist = dist;
				min_pid = i + block_id;
			}
			__syncthreads();
		}

		__syncthreads();

		edge_set[pid].x = pid;
		edge_set[pid].y = min_pid;
	}
	__syncthreads();


	// Handle last block
	if (pid >= last_point && pid < point_num) {
		float tmp_x, tmp_y, tmp_z;
		float min_dist = threshold;
		int min_pid = -1;

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

				if (i + block_id > pid && dist < threshold && min_dist > dist) {
					min_dist = dist;
					min_pid = i + block_id;
				}
				__syncthreads();
			}

			__syncthreads();
		}

		edge_set[pid].x = pid;
		edge_set[pid].y = min_pid;
	}

	__syncthreads();

}

__global__ void sampleGraphBuild2(float *x, float *y, float *z, int point_num, int2 *edge_set, float threshold)
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
		float lmin = threshold, rmin = threshold;
		int lpid = -1, rpid = -1;

		int block_id;

		for (block_id = 0; block_id + blockDim.x < point_num; block_id += blockDim.x) {
			local_x[threadIdx.x] = x[block_id + threadIdx.x];
			local_y[threadIdx.x] = y[block_id + threadIdx.x];
			local_z[threadIdx.x] = z[block_id + threadIdx.x];
			__syncthreads();

			for (int i = 0; i < blockDim.x; i++) {
				dist = norm3df(tmp_x - local_x[i], tmp_y - local_y[i], tmp_z - local_z[i]);

				if (i + block_id > pid && dist < threshold && rmin > dist) {
					rmin = dist;
					rpid = i + block_id;
				}

				__syncthreads();

				if (i + block_id < pid && dist < threshold && lmin > dist) {
					lmin = dist;
					lpid = i + block_id;
				}
				__syncthreads();
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

			if (i + block_id > pid && dist < threshold && rmin > dist) {
				rmin = dist;
				rpid = i + block_id;
			}
			__syncthreads();

			if (i + block_id < pid && dist < threshold && lmin > dist) {
				lmin = dist;
				lpid = i + block_id;
			}
			__syncthreads();
		}

		__syncthreads();

		edge_set[pid].x = pid;
		edge_set[pid].y = rpid;
		edge_set[pid + point_num].x = pid;
		edge_set[pid + point_num].y = lpid;
	}
	__syncthreads();


	// Handle last block
	if (pid >= last_point) {
		float tmp_x, tmp_y, tmp_z;
		float lmin = threshold, rmin = threshold;
		int lpid = -1, rpid = -1;

		if (pid < point_num) {
			tmp_x = x[pid];
			tmp_y = y[pid];
			tmp_z = z[pid];
		}

		//int block_id = blockIdx.x * blockDim.x;
		int block_id;

		__syncthreads();

		for (block_id = 0; block_id + blockDim.x < point_num; block_id += blockDim.x) {
			local_x[threadIdx.x] = x[block_id + threadIdx.x];
			local_y[threadIdx.x] = y[block_id + threadIdx.x];
			local_z[threadIdx.x] = z[block_id + threadIdx.x];
			__syncthreads();

			if (pid < point_num) {
				for (int i = 0; i < blockDim.x; i++) {
					dist = norm3df(tmp_x - local_x[i], tmp_y - local_y[i], tmp_z - local_z[i]);

					if (i + block_id > pid && dist < threshold && rmin > dist) {
						rmin = dist;
						rpid = i + block_id;
					}
					__syncthreads();

					if (i + block_id < pid && dist < threshold && lmin > dist) {
						lmin = dist;
						lpid = i + block_id;
					}
					__syncthreads();
				}
			}
			__syncthreads();
		}


		if (pid < point_num) {
			local_x[threadIdx.x] = x[pid];
			local_y[threadIdx.x] = y[pid];
			local_z[threadIdx.x] = z[pid];
			__syncthreads();

			for (int i = 0; i < point_num - block_id; i++) {
				dist = norm3df(tmp_x - local_x[i], tmp_y - local_y[i], tmp_z - local_z[i]);

				if (i + block_id > pid && dist < threshold && rmin > dist) {
					rmin = dist;
					rpid = i + block_id;
				}
				__syncthreads();

				if (i + block_id < pid && dist < threshold && lmin > dist) {
					lmin = dist;
					lpid = i + block_id;
				}
				__syncthreads();
			}

			__syncthreads();

			edge_set[pid].x = pid;
			edge_set[pid].y = rpid;
			edge_set[pid + point_num].x = pid;
			edge_set[pid + point_num].y = lpid;
		}

		__syncthreads();


	}

	__syncthreads();

}

__global__ void markInvalidEdge(int2 *sample_edge_set, int size, int *mark)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x) {
		mark[i] = (sample_edge_set[i].y >= 0) ? 1 : 0;
	}
}

__global__ void graphBuild(int2 *sample_edge_set, int size, int *location, int2 *edge_set)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x) {
		if (sample_edge_set[i].y >= 0) {
			edge_set[location[i]] = sample_edge_set[i];
		}
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
	struct timeval start, end;

	initClusters();

	int block_x, grid_x;

	block_x = (point_num_ > block_size_x_) ? block_size_x_ : point_num_;
	grid_x = (point_num_ - 1) / block_x + 1;

	int2 *sample_edge_set;

	gettimeofday(&start, NULL);

	checkCudaErrors(cudaMalloc(&sample_edge_set, sizeof(int2) * point_num_ * 2));

	sampleGraphBuild2<<<grid_x, block_x, sizeof(float) * block_size_x_ * 3>>>(x_, y_, z_, point_num_, sample_edge_set, threshold_);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	gettimeofday(&end, NULL);

	std::cout << "Sample graph build = " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);

	int edge_num;
	int *mark, *location;

	checkCudaErrors(cudaMalloc(&mark, sizeof(int) * (point_num_ * 2 + 1)));

	location = mark;

	markInvalidEdge<<<grid_x, block_x>>>(sample_edge_set, point_num_ * 2, mark);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	GUtilities::exclusiveScan(mark, point_num_ * 2 + 1, &edge_num);

	if (edge_num == 0) {
		checkCudaErrors(cudaFree(sample_edge_set));
		checkCudaErrors(cudaFree(mark));
		cluster_num_ = point_num_;
		return;
	}

	int2 *edge_set;

	checkCudaErrors(cudaMalloc(&edge_set, sizeof(int2) * edge_num));

	graphBuild<<<grid_x, block_x>>>(sample_edge_set, point_num_ * 2, location, edge_set);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	bool *changed;
	bool hchanged;

	checkCudaErrors(cudaMalloc(&changed, sizeof(bool)));

	block_x = (edge_num > block_size_x_) ? block_size_x_ : edge_num;
	grid_x = (edge_num - 1) / block_x + 1;

	int itr = 0;

	gettimeofday(&end, NULL);

	std::cout << "Build Edge Set = " << timeDiff(start, end) << std::endl;


	gettimeofday(&start, NULL);
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

	std::cout << "Iteration time = " << timeDiff(start, end) << " itr_num = " << itr << std::endl;

	int *count;

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

	checkCudaErrors(cudaFree(sample_edge_set));
	checkCudaErrors(cudaFree(mark));
	checkCudaErrors(cudaFree(edge_set));
	checkCudaErrors(cudaFree(changed));
	checkCudaErrors(cudaFree(count));

	std::cout << "FINAL CLUSTER NUM = " << cluster_num_ << std::endl;

}

void GpuEuclideanCluster2::extractClusters5(long long &total_time, long long &build_graph, long long &clustering_time, int &iteration_num)
{
	total_time = build_graph = clustering_time = 0;
	iteration_num = 0;

	struct timeval start, end;

	initClusters();

	int block_x, grid_x;

	block_x = (point_num_ > block_size_x_) ? block_size_x_ : point_num_;
	grid_x = (point_num_ - 1) / block_x + 1;


	int2 *sample_edge_set;

	gettimeofday(&start, NULL);

	checkCudaErrors(cudaMalloc(&sample_edge_set, sizeof(int2) * point_num_));

	sampleGraphBuild<<<grid_x, block_x, sizeof(float) * block_size_x_ * 3>>>(x_, y_, z_, point_num_, sample_edge_set, threshold_);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	gettimeofday(&end, NULL);

	build_graph += timeDiff(start, end);
	total_time += timeDiff(start, end);
	std::cout << "Count Edge Set = " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	int edge_num;
	int *mark, *location;

	checkCudaErrors(cudaMalloc(&mark, sizeof(int) * (point_num_ + 1)));

	location = mark;

	markInvalidEdge<<<grid_x, block_x>>>(sample_edge_set, point_num_, mark);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	GUtilities::exclusiveScan(mark, point_num_ + 1, &edge_num);

	//std::cout << "Edge num = " << edge_num << std::endl;

	gettimeofday(&end, NULL);

	build_graph += timeDiff(start, end);
	total_time += timeDiff(start, end);

	if (edge_num == 0) {
		checkCudaErrors(cudaFree(sample_edge_set));
		checkCudaErrors(cudaFree(mark));
		cluster_num_ = point_num_;
		return;
	}


	int2 *edge_set;

	gettimeofday(&start, NULL);

	checkCudaErrors(cudaMalloc(&edge_set, sizeof(int2) * edge_num));

	graphBuild<<<grid_x, block_x>>>(sample_edge_set, point_num_, location, edge_set);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	gettimeofday(&end, NULL);

	build_graph += timeDiff(start, end);
	total_time += timeDiff(start, end);

	std::cout << "Build Edge Set = " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);

	bool *changed;
	bool hchanged;

	checkCudaErrors(cudaMalloc(&changed, sizeof(bool)));

	block_x = (edge_num > block_size_x_) ? block_size_x_ : edge_num;
	grid_x = (edge_num - 1) / block_x + 1;

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
	//std::cout << "Iteration time = " << timeDiff(start, end) << " itr_num = " << itr << std::endl;

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


	checkCudaErrors(cudaFree(sample_edge_set));
	checkCudaErrors(cudaFree(mark));
	checkCudaErrors(cudaFree(edge_set));
	checkCudaErrors(cudaFree(changed));
	checkCudaErrors(cudaFree(count));
	gettimeofday(&end, NULL);

	total_time += timeDiff(start, end);

	std::cout << "FINAL CLUSTER NUM = " << cluster_num_ << std::endl;
}
