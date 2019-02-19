#include "include/euclidean_cluster.h"
#include "include/utilities.h"
#include <cuda.h>


GpuEuclideanCluster2::GpuEuclideanCluster2()
{
	x_ = y_ = z_ = NULL;

	point_num_ = 0;
	padded_num_ = 0;
	threshold_ = 0;
	cluster_name_ = NULL;
	cluster_name_host_ = NULL;
	min_cluster_pts_ = 0;
	max_cluster_pts_ = INT_MAX;
	cluster_num_ = 0;

	block_size_x_ = 1024;
}

void GpuEuclideanCluster2::setThreshold(double threshold)
{
	threshold_ = threshold;
}

void GpuEuclideanCluster2::setMinClusterPts(int min_cluster_pts)
{
	min_cluster_pts_ = min_cluster_pts;
}

void GpuEuclideanCluster2::setMaxClusterPts(int max_cluster_pts)
{
	max_cluster_pts_ = max_cluster_pts;
}

void GpuEuclideanCluster2::setBlockSizeX(int block_size)
{
	if (block_size > BLOCK_SIZE_X) {
		block_size_x_ = BLOCK_SIZE_X;
	} else {
		block_size_x_ = block_size;
	}
}

__global__ void convertFormat(pcl::PointXYZ *input, float *out_x, float *out_y, float *out_z, int point_num)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = idx; i < point_num; i += blockDim.x * gridDim.x) {
		pcl::PointXYZ tmp_input = input[i];
		out_x[i] = tmp_input.x;
		out_y[i] = tmp_input.y;
		out_z[i] = tmp_input.z;
		// Convert to 2d cloud
		//out_z[i] = 0;
	}
}

void GpuEuclideanCluster2::setInputPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr input)
{
	if (x_ != NULL)
		checkCudaErrors(cudaFree(x_));

	if (y_ != NULL)
		checkCudaErrors(cudaFree(y_));

	if (z_ != NULL)
		checkCudaErrors(cudaFree(z_));

	if (input->size() > 0) {
		point_num_ = input->size();
		padded_num_ = ((point_num_ - 1) / block_size_x_ + 1) * block_size_x_;
		checkCudaErrors(cudaMalloc(&x_, sizeof(float) * padded_num_));
		checkCudaErrors(cudaMalloc(&y_, sizeof(float) * padded_num_));
		checkCudaErrors(cudaMalloc(&z_, sizeof(float) * padded_num_));

		pcl::PointXYZ *dev_tmp_input;

		checkCudaErrors(cudaMalloc(&dev_tmp_input, sizeof(pcl::PointXYZ) * point_num_));
		checkCudaErrors(cudaMemcpy(dev_tmp_input, input->points.data(), sizeof(pcl::PointXYZ) * point_num_, cudaMemcpyHostToDevice));

		int block_x = (point_num_ > block_size_x_) ? block_size_x_ : point_num_;
		int grid_x = (point_num_ - 1) / block_x + 1;

		convertFormat<<<grid_x, block_x>>>(dev_tmp_input, x_, y_, z_, point_num_);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaFree(dev_tmp_input));


		checkCudaErrors(cudaMalloc(&cluster_name_, point_num_ * sizeof(int)));
		cluster_name_host_ = (int*)malloc(point_num_ * sizeof(int));
	}
}

__global__ void initClusterNames(int *cluster_names, int point_num)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < point_num; i += blockDim.x * gridDim.x) {
		cluster_names[i] = i;
	}
}

void GpuEuclideanCluster2::initClusters()
{
	int block_x, grid_x;

	block_x = (point_num_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : point_num_;
	grid_x = (point_num_ - 1) / block_x + 1;

	initClusterNames<<<grid_x, block_x>>>(cluster_name_, point_num_);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

std::vector<GpuEuclideanCluster2::GClusterIndex> GpuEuclideanCluster2::getOutput()
{
	std::vector<GpuEuclideanCluster2::GClusterIndex> output(cluster_num_);

	if (cluster_num_ == 0)
		return output;

	for (int i = 0; i < cluster_num_; i++) {
		output[i].index_value = i;
	}

	for (int i = 0; i < point_num_; i++) {
		GClusterIndex &cluster = output[cluster_name_host_[i]];

		cluster.points_in_cluster.push_back(i);
	}


	int point_num_test = 0;

	for (unsigned int i = 0; i < output.size();) {
		int number_of_pts = output[i].points_in_cluster.size();

		point_num_test += number_of_pts;

		if (number_of_pts < min_cluster_pts_ || number_of_pts > max_cluster_pts_)
			output.erase(output.begin() + i);
		else
			i++;
	}

	return output;
}

/* Re-indexing the cluster name array to make all cluster indexes starts from 0 */
__global__ void renameClusters(int *cluster_name, int *cluster_location, int point_num)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < point_num; i += blockDim.x * gridDim.x) {
		int old_name = cluster_name[i];

		cluster_name[i] = cluster_location[old_name];
	}
}

void GpuEuclideanCluster2::renamingClusters(int *cluster_names, int *cluster_location, int point_num) {
	int block_x = (point_num > block_size_x_) ? block_size_x_ : point_num;
	int grid_x = (point_num - 1) / block_x + 1;

	renameClusters<<<grid_x, block_x>>>(cluster_names, cluster_location, point_num);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}


GpuEuclideanCluster2::~GpuEuclideanCluster2()
{
	if (x_ != NULL) {
		checkCudaErrors(cudaFree(x_));
	}

	if (y_ != NULL) {
		checkCudaErrors(cudaFree(y_));
	}

	if (z_ != NULL) {
		checkCudaErrors(cudaFree(z_));
	}

	if (cluster_name_ != NULL) {
		checkCudaErrors(cudaFree(cluster_name_));
	}

	if (cluster_name_host_ != NULL) {
		free(cluster_name_host_);
	}
}

__global__ void edgeCountCommon(float *x, float *y, float *z, int point_num, int *edge_count, float threshold)
{
	__shared__ float local_x[BLOCK_SIZE_X];
	__shared__ float local_y[BLOCK_SIZE_X];
	__shared__ float local_z[BLOCK_SIZE_X];
	int pid;
	int last_point = (point_num / blockDim.x) * blockDim.x;	// Exclude the last block
	float dist;

	for (pid = threadIdx.x + blockIdx.x * blockDim.x; pid < last_point; pid += blockDim.x * gridDim.x) {
		float tmp_x = x[pid];
		float tmp_y = y[pid];
		float tmp_z = z[pid];
		int count = 0;

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

		edge_count[pid] = count;
		__syncthreads();
	}
	__syncthreads();


	// Handle last block
	if (pid >= last_point) {
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
}

float GpuEuclideanCluster2::density(){

	float result = 0;

	int *edge_count;

	checkCudaErrors(cudaMalloc(&edge_count, sizeof(int) * (point_num_ + 1)));

	int block_x = (point_num_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : point_num_;
	int grid_x = (point_num_ - 1) / block_x + 1;

	edgeCountCommon<<<grid_x, block_x>>>(x_, y_, z_, point_num_, edge_count, threshold_);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	int edge_num = 0;

	GUtilities::exclusiveScan(edge_count, point_num_ + 1, &edge_num);

	std::cout << "Edge num = " << edge_num << std::endl;

	float fpnum = static_cast<float>(point_num_);

	result = static_cast<float>(edge_num) / (fpnum * (fpnum - 1));

	checkCudaErrors(cudaFree(edge_count));

	return result;
}


