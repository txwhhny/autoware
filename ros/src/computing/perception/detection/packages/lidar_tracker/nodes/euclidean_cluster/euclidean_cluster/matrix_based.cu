#include "include/euclidean_cluster.h"
#include "include/utilities.h"
#include <cuda.h>

#define NON_ATOMIC_ 1

extern __shared__ float local_buff[];

/* Arrays to remember:
 * cluster_name: the index of the cluster that each point belong to
 * 				i.e. point at index i belong to cluster cluster_name[i]
 * cluster_list: the list of remaining clusters
 * cluster_location: location of the remaining clusters in the cluster list
 * 					i.e. cluster A locate in index cluster_location[A] in the
 * 					cluster_list
 * matrix: the adjacency matrix of the cluster list, each cluster is a vertex.
 * 			This matrix is rebuilt whenever some clusters are merged together
 */


/* Connected component labeling points at GPU block thread level.
 * Input list of points is divided into multiple smaller groups.
 * Each group of point is assigned to a block of GPU thread.
 * Each thread in a block handles one point in the group. It iterates over
 * points in the group and compare the distance between the current point A
 * and the point B it has to handle.
 *
 * If the distance between A and B is less than the threshold, then those
 * two points belong to a same connected component and the cluster_changed
 * is marked by 1.
 *
 * A synchronization is called to make sure all thread in the block finish A
 * before moving to the update phase.
 * After finishing checking cluster_changed, threads update the cluster
 * index of all points. If a thread has cluster_changed is 1, then the corresponding
 * cluster of the point it is handling is changed to the cluster of B. Otherwise
 * the original cluster of A remains unchanged.
 *
 * Another synchronization is called before all threads in the block move to
 * other points after done checking A.
 *
 * After this kernel finishes, all points in each block are labeled.
 */
__global__ void blockClustering(float *x, float *y, float *z, int point_num, int *cluster_name, float threshold)
{
	int block_start = blockIdx.x * blockDim.x;
	int block_end = (block_start + blockDim.x > point_num) ? point_num : block_start + blockDim.x;

	float *local_x = local_buff;
	float *local_y = local_x + blockDim.x;
	float *local_z = local_y + blockDim.x;
	/* Each thread is in charge of one point in the block.*/
	int pid = threadIdx.x + block_start;
	/* Local cluster to record the change in the name of the cluster each point belong to */
	int *local_cluster_idx = (int*)(local_z + blockDim.x);
	/* Cluster changed to check if a cluster name has changed after each comparison */
	int *cluster_changed = local_cluster_idx + blockDim.x;

	if (pid < block_end) {
		local_cluster_idx[threadIdx.x] = threadIdx.x;
		local_x[threadIdx.x] = x[pid];
		local_y[threadIdx.x] = y[pid];
		local_z[threadIdx.x] = z[pid];
		__syncthreads();

		float cx = local_x[threadIdx.x];
		float cy = local_y[threadIdx.x];
		float cz = local_z[threadIdx.x];

		/* Iterate through all points in the block and check if the point at row index
		 * and at column index belong to the same cluster.
		 * If so, then name of the cluster of the row point is changed into the name
		 * of the cluster of column point.
		 * */
		for (int rid = 0; rid < block_end - block_start; rid++) {
			float distance = norm3df(cx - local_x[rid], cy - local_y[rid], cz - local_z[rid]);
			int row_cluster = local_cluster_idx[rid];
			int col_cluster = local_cluster_idx[threadIdx.x];

			cluster_changed[threadIdx.x] = 0;
			__syncthreads();

			if (threadIdx.x > rid && distance < threshold) {
				cluster_changed[col_cluster] = 1;
			}
			__syncthreads();

			local_cluster_idx[threadIdx.x] = (cluster_changed[col_cluster] == 1) ? row_cluster : col_cluster;
			__syncthreads();
		}
		__syncthreads();

		int new_cluster = cluster_name[block_start + local_cluster_idx[threadIdx.x]];
		__syncthreads();

		cluster_name[pid] = new_cluster;
	}
}

__global__ void foreignBlockClustering(float *x, float *y, float *z, int point_num, int *cluster_name, float threshold,
											int shift_level,
											int sub_mat_size,
											int sub_mat_offset)
{
	int sub_mat_idx = blockIdx.x / sub_mat_size;
	int col_start = (sub_mat_size + sub_mat_idx * sub_mat_offset + (shift_level + blockIdx.x) % sub_mat_size) * blockDim.x;
	int col_end = (col_start + blockDim.x <= point_num) ? col_start + blockDim.x : point_num;
	int row_start = (sub_mat_idx * sub_mat_offset + blockIdx.x % sub_mat_size) * blockDim.x;
	int row_end = (row_start + blockDim.x <= point_num) ? row_start + blockDim.x : point_num;
	int col = col_start + threadIdx.x;

	float *row_x = local_buff;
	float *row_y = row_x + blockDim.x;
	float *row_z = row_y + blockDim.x;
	int *changed_status = (int*)(row_z + blockDim.x);
	int *row_label = (int*)(changed_status + blockDim.x * 2);
	bool tchanged = false;
	float cx, cy, cz;
	int clabel;

	if (row_start + threadIdx.x < row_end) {
		row_x[threadIdx.x] = x[row_start + threadIdx.x];
		row_y[threadIdx.x] = y[row_start + threadIdx.x];
		row_z[threadIdx.x] = z[row_start + threadIdx.x];
		row_label[threadIdx.x] = cluster_name[row_start + threadIdx.x] - row_start + blockDim.x;
	}
	__syncthreads();

	if (col < col_end) {
		cx = x[col];
		cy = y[col];
		cz = z[col];

		// Local label of the column handled by the thread
		clabel = cluster_name[col] - col_start;
	}
	__syncthreads();

	for (int row = 0; row < row_end - row_start; row++) {
		changed_status[threadIdx.x] = 0;
		changed_status[threadIdx.x + blockDim.x] = 0;
		int rlabel = row_label[row];
		float rx = row_x[row];
		float ry = row_y[row];
		float rz = row_z[row];
		__syncthreads();

		if (col < col_end && rlabel != clabel && norm3df(cx - rx, cy - ry, cz - rz) < threshold) {
			changed_status[clabel] = 1;
		}
		__syncthreads();

		if (col < col_end && changed_status[clabel] == 1) {
			clabel = rlabel;
			tchanged = true;
		}
		__syncthreads();
	}
	__syncthreads();

	if (tchanged) {
		cluster_name[col_start + threadIdx.x] = cluster_name[row_start + clabel - blockDim.x];
	}
}

__global__ void foreignClustering(int *matrix, int *cluster_list,
									int shift_level,
									int sub_mat_size,
									int sub_mat_offset,
									int cluster_num, bool *changed)
{
	int sub_mat_idx = blockIdx.x / sub_mat_size;
	int col_start = (sub_mat_size + sub_mat_idx * sub_mat_offset + (shift_level + blockIdx.x) % sub_mat_size) * blockDim.x;
	int col_end = (col_start + blockDim.x <= cluster_num) ? col_start + blockDim.x : cluster_num;
	int row_start = (sub_mat_idx * sub_mat_offset + blockIdx.x % sub_mat_size) * blockDim.x;
	int row_end = (row_start + blockDim.x <= cluster_num) ? row_start + blockDim.x : cluster_num;
	int col = col_start + threadIdx.x;
	bool tchanged = false;

	int *tmp = (int*)local_buff;

	bool *bchanged = (bool*)(tmp + blockDim.x * 2);
	int clabel = threadIdx.x;

	if (threadIdx.x == 0)
		*bchanged = false;
	__syncthreads();

	for (int row = row_start; row < row_end; row++) {
		int rlabel = row - row_start;

		tmp[threadIdx.x] = 0;
		tmp[threadIdx.x + blockDim.x] = 0;
		__syncthreads();

		if (col < col_end && matrix[row * cluster_num + col] == 1) {
			tmp[clabel] = 1;
		}
		__syncthreads();

		if (col < col_end && tmp[clabel] == 1) {
			clabel = rlabel + blockDim.x;
			tchanged = true;
		}
		__syncthreads();
	}

	__syncthreads();


	if (tchanged) {
		int new_cluster = cluster_list[row_start + clabel - blockDim.x];
		__syncthreads();
		cluster_list[col] = new_cluster;
	}

	if (tchanged)
		*bchanged = true;

	__syncthreads();
	if (threadIdx.x == 0 && *bchanged)
		*changed = true;
}

/* Iterate through the list of remaining clusters and mark
 * the corresponding location on cluster location array by 1
 */
__global__ void clusterMark(int *cluster_list, int *cluster_location, int cluster_num)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = idx; i < cluster_num; i += blockDim.x * gridDim.x) {
		cluster_location[cluster_list[i]] = 1;
	}
}

void GpuEuclideanCluster2::clusterMarkWrapper(int *cluster_list, int *cluster_location, int cluster_num)
{
	int block_x = (cluster_num > block_size_x_) ? block_size_x_ : cluster_num;
	int grid_x = (cluster_num - 1) / block_x + 1;

	clusterMark<<<grid_x, block_x>>>(cluster_list, cluster_location, cluster_num);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

/* Collect the remaining clusters */
__global__ void clusterCollector(int *new_cluster_list, int new_cluster_num)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = idx; i < new_cluster_num; i += blockDim.x * gridDim.x) {
		new_cluster_list[i] = i;
	}
}

void GpuEuclideanCluster2::clusterCollectorWrapper(int *new_cluster_list, int new_cluster_num)
{
	int block_x = (new_cluster_num > block_size_x_) ? block_size_x_ : new_cluster_num;
	int grid_x = (new_cluster_num - 1) / block_x + 1;

	clusterCollector<<<grid_x, block_x>>>(new_cluster_list, new_cluster_num);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}


// Not working with large matrix because the limitation of int type will lead to overflow
__global__ void buildClusterMatrix(float *x, float *y, float *z, int *cluster_name, int *matrix, int point_num, int cluster_num, float threshold)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int pid = idx; pid < point_num; pid += stride) {
		float tmp_x = x[pid];
		float tmp_y = y[pid];
		float tmp_z = z[pid];
		int col = cluster_name[pid];

		for (int pid2 = blockIdx.y; pid2 < point_num; pid2 += gridDim.y) {
			float tmp_x2 = tmp_x - x[pid2];
			float tmp_y2 = tmp_y - y[pid2];
			float tmp_z2 = tmp_z - z[pid2];
			int row = cluster_name[pid2];


			if (row < col && norm3df(tmp_x2, tmp_y2, tmp_z2) < threshold) {

				matrix[row * cluster_num + col] = 1;
			}
			__syncthreads();
		}
		__syncthreads();
	}
}

void GpuEuclideanCluster2::buildClusterMatrixWrapper(float *x, float *y, float *z,
													int *cluster_name, int *cluster_location,
													int *matrix, int point_num,
													int cluster_num, float threshold)
{
	dim3 grid_size, block_size;

	block_size.x = (point_num > block_size_x_) ? block_size_x_ : point_num;
	block_size.y = block_size.z = 1;
	grid_size.x = (point_num - 1) / block_size.x + 1;
	grid_size.y = (cluster_num > GRID_SIZE_Y) ? GRID_SIZE_Y : cluster_num;
	grid_size.z = 1;

	buildClusterMatrix<<<grid_size, block_size>>>(x, y, z, cluster_name, matrix,
													point_num, cluster_num,
													threshold);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}


/* Merge clusters that belong to a same block of threads */
__global__ void mergeLocalClusters(int *cluster_list, int *matrix, int cluster_num, bool *changed)
{
	int row_start = blockIdx.x * blockDim.x;
	int row_end = (row_start + blockDim.x <= cluster_num) ? row_start + blockDim.x : cluster_num;
	int col = row_start + threadIdx.x;

	int *local_cluster_idx = (int*)local_buff;
	int *cluster_changed = local_cluster_idx + blockDim.x;
	bool tchanged = false;
	bool *bchanged = (bool*)(cluster_changed + blockDim.x);

	if (threadIdx.x == 0)
		*bchanged = false;
	__syncthreads();

	if(col < cluster_num && row_start < row_end) {
		local_cluster_idx[threadIdx.x] = threadIdx.x;
		__syncthreads();

		for (int row = row_start; row < row_end; row++) {
			int col_cluster = local_cluster_idx[threadIdx.x];
			int row_cluster = local_cluster_idx[row - row_start];

			cluster_changed[threadIdx.x] = 0;
			__syncthreads();

			if (row - row_start < threadIdx.x && row_cluster != col_cluster && matrix[row * cluster_num + col] == 1) {
				cluster_changed[col_cluster] = 1;
				tchanged = true;
			}
			__syncthreads();

			local_cluster_idx[threadIdx.x] = (cluster_changed[col_cluster] == 1) ? row_cluster : col_cluster;
			__syncthreads();
		}

		__syncthreads();
		int new_cluster = cluster_list[row_start + local_cluster_idx[threadIdx.x]];
		__syncthreads();

		cluster_list[col] = new_cluster;

		if (tchanged)
			*bchanged = true;
	}

	__syncthreads();
	if (threadIdx.x == 0 && *bchanged)
		*changed = true;
}

void GpuEuclideanCluster2::mergeLocalClustersWrapper(int *cluster_list, int *matrix, int cluster_num, bool *changed)
{
	int block_x = (cluster_num > block_size_x_) ? block_size_x_ : cluster_num;
	int grid_x = (cluster_num - 1) / block_x + 1;

	mergeLocalClusters<<<grid_x, block_x, sizeof(int) * 2 * block_size_x_ + sizeof(bool)>>>(cluster_list, matrix, cluster_num, changed);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	bool hchanged = false;

	checkCudaErrors(cudaMemcpy(&hchanged, changed, sizeof(bool), cudaMemcpyDeviceToHost));

	if (cluster_num > block_size_x_) {
		int sub_mat_offset = 2;
		int sub_mat_num = (cluster_num - 1) / block_size_x_ + 1;
		int active_block_num = (sub_mat_num - 1) / sub_mat_offset + 1;

		block_x = block_size_x_;
		grid_x = active_block_num;

		foreignClustering<<<grid_x, block_x, sizeof(int) * 2 * block_size_x_ + sizeof(bool)>>>(matrix, cluster_list, 0, 1, 2, cluster_num, changed);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}
}

/* Merge clusters that belong to different block of threads */
__global__ void mergeForeignClusters(int *matrix, int *cluster_list,
										int shift_level,
										int sub_mat_size,
										int sub_mat_offset,
										int cluster_num, bool *changed)
{
	int sub_mat_idx = blockIdx.x / sub_mat_size;
	int col_start = (sub_mat_size + sub_mat_idx * sub_mat_offset + (shift_level + blockIdx.x) % sub_mat_size) * blockDim.x;
	int col_end = (col_start + blockDim.x <= cluster_num) ? col_start + blockDim.x : cluster_num;
	int row_start = (sub_mat_idx * sub_mat_offset + blockIdx.x % sub_mat_size) * blockDim.x;
	int row_end = (row_start + blockDim.x <= cluster_num) ? row_start + blockDim.x : cluster_num;
	int col = col_start + threadIdx.x;
	bool tchanged = false;

	int *tmp = (int*)local_buff;

	bool *bchanged = (bool*)(tmp + blockDim.x * 2);
	int clabel = threadIdx.x;

	if (threadIdx.x == 0)
		*bchanged = 0;
	__syncthreads();

	for (int row = row_start; row < row_end; row++) {
		int rlabel = row - row_start;

		tmp[threadIdx.x] = 0;
		tmp[threadIdx.x + blockDim.x] = 0;
		__syncthreads();

		if (col < col_end && matrix[row * cluster_num + col] == 1) {
			tmp[clabel] = 1;
		}
		__syncthreads();

		if (col < col_end && tmp[clabel] == 1) {
			clabel = rlabel + blockDim.x;
			tchanged = true;
		}
		__syncthreads();
	}

	__syncthreads();

	if (tchanged) {
		int new_cluster = cluster_list[row_start + clabel - blockDim.x];
		cluster_list[col] = new_cluster;
		*bchanged = true;
	}
	__syncthreads();

	if (threadIdx.x == 0 && *bchanged) {
		*changed = true;
	}
}

void GpuEuclideanCluster2::mergeForeignClustersWrapper(int *matrix, int *cluster_list,
														int shift_level,
														int sub_mat_size,
														int sub_mat_offset,
														int cluster_num, bool *changed)
{
	int block_x = (cluster_num > block_size_x_) ? block_size_x_ : cluster_num;
	int grid_x = (cluster_num - 1) / block_x + 1;

	mergeForeignClusters<<<grid_x, block_x, sizeof(int) * 2 * block_size_x_ + sizeof(bool)>>>(matrix, cluster_list, shift_level,
																								sub_mat_size, sub_mat_offset,
																								cluster_num, changed);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

/* Check if are there any 1 element in the adjacency matrix */
__global__ void clusterIntersecCheck(int *matrix, int *changed_diag, int sub_mat_size, int sub_mat_offset, int cluster_num)
{
	int col_idx = ((blockIdx.x / sub_mat_size) * sub_mat_offset + blockIdx.x % sub_mat_size);
	int row_idx = ((blockIdx.x / sub_mat_size) * sub_mat_offset + blockIdx.y % sub_mat_size);

	int col_start = (sub_mat_size + col_idx) * blockDim.x;
	int col_end = (col_start + blockDim.x <= cluster_num) ? col_start + blockDim.x : cluster_num;

	int row_start = row_idx * blockDim.x;
	int row_end = (row_start + blockDim.x <= cluster_num) ? row_start + blockDim.x : cluster_num;

	int col = col_start + threadIdx.x;
	int diag_offset = (sub_mat_size + col_idx - row_idx) % sub_mat_size;
	__shared__ int schanged_diag;

	if (threadIdx.x == 0)
		schanged_diag = -1;
	__syncthreads();

	if (col < col_end && col_start < col_end && row_start < row_end) {
		for (int row = row_start; row < row_end; row++) {
			if (matrix[row * cluster_num + col] != 0) {
				schanged_diag = diag_offset;
				break;
			}
		}
	}
	__syncthreads();

	if (threadIdx.x == 0 && schanged_diag >= 0)
		*changed_diag = schanged_diag;
}

void GpuEuclideanCluster2::clusterIntersecCheckWrapper(int *matrix, int *changed_diag, int *hchanged_diag, int sub_mat_size, int sub_mat_offset, int cluster_num)
{
	int tiny_matrix_num = (cluster_num - 1) / block_size_x_ + 1;
	int sub_mat_num = (tiny_matrix_num - 1) / sub_mat_offset + 1;

	dim3 block_size, grid_size;

	block_size.x = block_size_x_;
	block_size.y = block_size.z = 1;
	grid_size.x = sub_mat_size * sub_mat_num;
	grid_size.y = sub_mat_size;
	grid_size.z = 1;

	*hchanged_diag = -1;

	checkCudaErrors(cudaMemcpy(changed_diag, hchanged_diag, sizeof(int), cudaMemcpyHostToDevice));

	clusterIntersecCheck<<<grid_size, block_size>>>(matrix, changed_diag, sub_mat_size, sub_mat_offset, cluster_num);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(hchanged_diag, changed_diag, sizeof(int), cudaMemcpyDeviceToHost));
}

/* Rename the cluster name of each point after some clusters are joined together */
__global__ void applyClusterChanged(int *cluster_name, int *cluster_list, int *cluster_location, int point_num)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = idx; i < point_num; i += blockDim.x * gridDim.x) {
		int old_cluster = cluster_name[i];

		cluster_name[i] = cluster_list[cluster_location[old_cluster]];
	}
}

void GpuEuclideanCluster2::applyClusterChangedWrapper(int *cluster_name, int *cluster_list, int *cluster_location, int point_num)
{
	int block_x = (point_num > block_size_x_) ? block_size_x_ : point_num;
	int grid_x = (point_num - 1) / block_x + 1;

	applyClusterChanged<<<grid_x, block_x>>>(cluster_name, cluster_list, cluster_location, point_num);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

/* Rebuild the adjacency matrix after some clusters are joined together */
__global__ void rebuildMatrix(int *old_matrix, int *merged_cluster_list, int *new_matrix, int *new_cluster_location, int old_cluster_num, int new_cluster_num)
{
	for (int col = threadIdx.x + blockIdx.x * blockDim.x; col < old_cluster_num; col += blockDim.x * gridDim.x) {
		int new_col = new_cluster_location[merged_cluster_list[col]];

		for (int row = blockIdx.y; row < col; row += gridDim.y) {
			int new_row = new_cluster_location[merged_cluster_list[row]];

			if (old_matrix[row * old_cluster_num + col] != 0) {
				new_matrix[new_row * new_cluster_num + new_col] = 1;
				new_matrix[new_col * new_cluster_num + new_row] = 1;
			}
		}
	}
}

__global__ void rebuildMatrix2(int *old_matrix, int *merged_cluster_list, int *new_matrix, int *new_cluster_location, int old_cluster_num, int new_cluster_num)
{
	for (int col = threadIdx.x + blockIdx.x * blockDim.x; col < old_cluster_num; col += blockDim.x * gridDim.x) {
		int new_col = new_cluster_location[merged_cluster_list[col]];

		for (int row = 0; row < col; row++) {
			int new_row = new_cluster_location[merged_cluster_list[row]];

			if (old_matrix[row * old_cluster_num + col] != 0) {
				new_matrix[new_row * new_cluster_num + new_col] = 1;
				new_matrix[new_col * new_cluster_num + new_row] = 1;	// Added 2019.02
			}
		}
	}
}

void GpuEuclideanCluster2::rebuildMatrixWrapper(int *old_matrix, int *merged_cluster_list,
												int *new_matrix, int *new_cluster_location,
												int old_cluster_num, int new_cluster_num)
{
	dim3 grid, block;

	block.x = (old_cluster_num > block_size_x_) ? block_size_x_ : old_cluster_num;
	block.y = block.z = 1;
	grid.x = (old_cluster_num - 1) / block.x + 1;
	grid.y = GRID_SIZE_Y;
	grid.z = 1;

	//rebuildMatrix<<<grid, block>>>(old_matrix, merged_cluster_list, new_matrix, new_cluster_location, old_cluster_num, cluster_num_);
	rebuildMatrix2<<<grid.x, block.x>>>(old_matrix, merged_cluster_list, new_matrix, new_cluster_location, old_cluster_num, cluster_num_);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}


void GpuEuclideanCluster2::blockClusteringWrapper(float *x, float *y, float *z, int point_num, int *cluster_name, float threshold)
{
	int block_x = (point_num > block_size_x_) ? block_size_x_ : point_num;
	int grid_x = (point_num - 1) / block_x + 1;

	blockClustering<<<grid_x, block_x, (sizeof(float) * 3 + sizeof(int) * 2) * block_size_x_>>>(x, y, z, point_num, cluster_name, threshold);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	if (point_num > block_size_x_) {
		foreignBlockClustering<<<grid_x, block_x, (sizeof(float) * 3 + sizeof(int) * 3) * block_size_x_>>>(x, y, z, point_num, cluster_name, threshold, 0, 1, 2);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

}

void GpuEuclideanCluster2::extractClusters(long long &total_time, long long &initial_time, long long &build_matrix, long long &clustering_time, int &iteration_num)
{
	std::cout << "MATRIX-BASED 1" << std::endl;
	total_time = initial_time = build_matrix = clustering_time = 0;

	struct timeval start, end;

	// Initialize names of clusters
	initClusters();

	bool *check;
	bool hcheck = false;

	checkCudaErrors(cudaMalloc(&check, sizeof(bool)));
	checkCudaErrors(cudaMemcpy(check, &hcheck, sizeof(bool), cudaMemcpyHostToDevice));

	gettimeofday(&start, NULL);
	blockClusteringWrapper(x_, y_, z_, point_num_, cluster_name_, threshold_);
	gettimeofday(&end, NULL);

	initial_time = timeDiff(start, end);
	total_time += timeDiff(start, end);
	//std::cout << "blockClustering = " << timeDiff(start, end) << std::endl;

	// Collect the remaining clusters
	// Locations of clusters in the cluster list
	int *cluster_location;

	gettimeofday(&start, NULL);
	checkCudaErrors(cudaMalloc(&cluster_location, sizeof(int) * (point_num_ + 1)));
	checkCudaErrors(cudaMemset(cluster_location, 0, sizeof(int) * (point_num_ + 1)));

	clusterMarkWrapper(cluster_name_, cluster_location, point_num_);

	int new_cluster_num = 0;
	GUtilities::exclusiveScan(cluster_location, point_num_ + 1, &new_cluster_num);

	int *cluster_list;

	checkCudaErrors(cudaMalloc(&cluster_list, sizeof(int) * new_cluster_num));

	clusterCollectorWrapper(cluster_list, new_cluster_num);

	// Added 2019.02.
	applyClusterChangedWrapper(cluster_name_, cluster_list, cluster_location, point_num_);


//	gettimeofday(&end, NULL);
//
//	total_time += timeDiff(start, end);
//	std::cout << "Collect remaining clusters: " << timeDiff(start, end) << std::endl;

	cluster_num_ = new_cluster_num;

	std::cout << "Number of remaining clusters: " << cluster_num_ << std::endl;

//	gettimeofday(&start, NULL);
	// Build relation matrix which describe the current relationship between clusters
	int *matrix;

	checkCudaErrors(cudaMalloc(&matrix, sizeof(int) * cluster_num_ * cluster_num_));
	checkCudaErrors(cudaMemset(matrix, 0, sizeof(int) * cluster_num_ * cluster_num_));
	checkCudaErrors(cudaDeviceSynchronize());

//	gettimeofday(&end, NULL);

	//std::cout << "Malloc and memset = " << timeDiff(start, end) << std::endl;

//	gettimeofday(&start, NULL);
	buildClusterMatrixWrapper(x_, y_, z_, cluster_name_,
								cluster_location, matrix,
								point_num_, cluster_num_,
								threshold_);
	gettimeofday(&end, NULL);

	build_matrix = timeDiff(start, end);
	total_time += timeDiff(start, end);
//	std::cout << "Build RC and Matrix = " << timeDiff(start, end) << std::endl;

	int *changed_diag;
	int hchanged_diag;
	checkCudaErrors(cudaMalloc(&changed_diag, sizeof(int)));

	int *new_cluster_list;

	gettimeofday(&start, NULL);
	int itr = 0;

	do {
		hcheck = false;
		hchanged_diag = -1;

		checkCudaErrors(cudaMemcpy(check, &hcheck, sizeof(bool), cudaMemcpyHostToDevice));

		mergeLocalClustersWrapper(cluster_list, matrix, cluster_num_, check);

		int sub_matrix_size = 2;
		int sub_matrix_offset = 4;

		checkCudaErrors(cudaMemcpy(&hcheck, check, sizeof(bool), cudaMemcpyDeviceToHost));

		int inner_itr_num = 0;

		while (!(hcheck) && sub_matrix_size * block_size_x_ < cluster_num_ && cluster_num_ > block_size_x_) {

			clusterIntersecCheckWrapper(matrix, changed_diag, &hchanged_diag, sub_matrix_size, sub_matrix_offset, cluster_num_);

			if (hchanged_diag >= 0) {
#ifdef DEBUG_
				std::cout << "Merge foreign clusters" << std::endl;
#endif
				mergeForeignClustersWrapper(matrix, cluster_list, hchanged_diag, sub_matrix_size, sub_matrix_offset, cluster_num_, check);

				checkCudaErrors(cudaMemcpy(&hcheck, check, sizeof(bool), cudaMemcpyDeviceToHost));
			}

			sub_matrix_size *= 2;
			sub_matrix_offset *= 2;
			inner_itr_num++;
		}

		/* If some changes in the cluster list are recorded (some clusters are merged together),
		 * rebuild the matrix, the cluster location, and apply those changes to the cluster_name array
		 */
		if (hcheck) {
			// Apply changes to the cluster_name array
			applyClusterChangedWrapper(cluster_name_, cluster_list, cluster_location, point_num_);

			checkCudaErrors(cudaMemset(cluster_location, 0, sizeof(int) * (point_num_ + 1)));

			// Remake the cluster location
			clusterMarkWrapper(cluster_list, cluster_location, cluster_num_);

			int old_cluster_num = cluster_num_;

			GUtilities::exclusiveScan(cluster_location, point_num_ + 1, &cluster_num_);

			checkCudaErrors(cudaMalloc(&new_cluster_list, sizeof(int) * cluster_num_));

			clusterCollectorWrapper(new_cluster_list, cluster_num_);

			// Rebuild matrix
			int *new_matrix;

#ifdef DEBUG_
			std::cout << "cluster_num = " << cluster_num_ << std::endl;
#endif

			checkCudaErrors(cudaMalloc(&new_matrix, sizeof(int) * cluster_num_ * cluster_num_));
			checkCudaErrors(cudaMemset(new_matrix, 0, sizeof(int) * cluster_num_ * cluster_num_));

			rebuildMatrixWrapper(matrix, cluster_list, new_matrix, cluster_location, old_cluster_num, cluster_num_);

			checkCudaErrors(cudaFree(cluster_list));
			cluster_list = new_cluster_list;

			checkCudaErrors(cudaFree(matrix));
			matrix = new_matrix;
		}

		itr++;
	} while (hcheck);


	gettimeofday(&end, NULL);

	clustering_time = timeDiff(start, end);
	total_time += timeDiff(start, end);
	iteration_num = itr;

#ifdef DEBUG_
	std::cout << "Iteration = " << timeDiff(start, end) << " itr_num = " << itr << std::endl;
#endif

	gettimeofday(&start, NULL);
//	renamingClusters(cluster_name_, cluster_location, point_num_);
	applyClusterChangedWrapper(cluster_name_, cluster_list, cluster_location, point_num_);

	checkCudaErrors(cudaMemcpy(cluster_name_host_, cluster_name_, point_num_ * sizeof(int), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(matrix));
	checkCudaErrors(cudaFree(cluster_list));
	checkCudaErrors(cudaFree(cluster_location));
	checkCudaErrors(cudaFree(check));
	checkCudaErrors(cudaFree(changed_diag));
	gettimeofday(&end, NULL);

	total_time += timeDiff(start, end);

#ifdef DEBUG_
	std::cout << "FINAL CLUSTER NUM = " << cluster_num_ << std::endl << std::endl;
#endif
}

void GpuEuclideanCluster2::extractClusters()
{
	long long total_time, initial_time, build_matrix, clustering_time;
	int iteration_num;

	extractClusters(total_time, initial_time, build_matrix, clustering_time, iteration_num);
}
