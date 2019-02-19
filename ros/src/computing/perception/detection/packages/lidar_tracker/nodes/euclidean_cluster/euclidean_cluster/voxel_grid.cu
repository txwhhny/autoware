#include "include/voxel_grid.h"

#define BLOCK_SIZE_X (1024)

GVoxelGrid::GVoxelGrid():
	x_(NULL),
	y_(NULL),
	z_(NULL),
	point_num_(0),
	voxel_num_(0),
	max_x_(FLT_MIN),
	max_y_(FLT_MIN),
	max_z_(FLT_MIN),
	min_x_(FLT_MAX),
	min_y_(FLT_MAX),
	min_z_(FLT_MAX),
	voxel_x_(0),
	voxel_y_(0),
	voxel_z_(0),
	max_b_x_(0),
	max_b_y_(0),
	max_b_z_(0),
	min_b_x_(0),
	min_b_y_(0),
	min_b_z_(0),
	vgrid_x_(0),
	vgrid_y_(0),
	vgrid_z_(0),
	points_per_voxel_(NULL),
	starting_point_ids_(NULL),
	point_ids_(NULL)
{
}

// x, y, z are GPU buffers
GVoxelGrid::GVoxelGrid(float *x, float *y, float *z, int point_num)
{
	x_ = x;
	y_ = y;
	z_ = z;
	point_num_ = point_num;

	findBoundaries();

	// Allocate empty voxel grid
	voxel_num_ = vgrid_x_ * vgrid_y_ * vgrid_z_;

	checkCudaErrors(cudaMalloc(&points_per_voxel_, sizeof(int) * voxel_num_));
	checkCudaErrors(cudaMemset(points_per_voxel_, 0, sizeof(int) * voxel_num_));
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMalloc(&starting_point_ids_, sizeof(int) * (voxel_num_ + 1)));

	insertPointsToVoxels();

}

__global__ void findMax(float *x, float *y, float *z, int full_size, int half_size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < half_size; i += stride) {
		x[i] = (i + half_size < full_size) ? ((x[i] >= x[i + half_size]) ? x[i] : x[i + half_size]) : x[i];
		y[i] = (i + half_size < full_size) ? ((y[i] >= y[i + half_size]) ? y[i] : y[i + half_size]) : y[i];
		z[i] = (i + half_size < full_size) ? ((z[i] >= z[i + half_size]) ? z[i] : z[i + half_size]) : z[i];
	}
}

__global__ void findMin(float *x, float *y, float *z, int full_size, int half_size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < half_size; i += stride) {
		x[i] = (i + half_size < full_size) ? ((x[i] <= x[i + half_size]) ? x[i] : x[i + half_size]) : x[i];
		y[i] = (i + half_size < full_size) ? ((y[i] <= y[i + half_size]) ? y[i] : y[i + half_size]) : y[i];
		z[i] = (i + half_size < full_size) ? ((z[i] <= z[i + half_size]) ? z[i] : z[i + half_size]) : z[i];
	}
}


void GVoxelGrid::findBoundaries()
{
	float *max_x, *max_y, *max_z, *min_x, *min_y, *min_z;

	checkCudaErrors(cudaMalloc(&max_x, sizeof(float) * point_num_));
	checkCudaErrors(cudaMalloc(&max_y, sizeof(float) * point_num_));
	checkCudaErrors(cudaMalloc(&max_z, sizeof(float) * point_num_));
	checkCudaErrors(cudaMalloc(&min_x, sizeof(float) * point_num_));
	checkCudaErrors(cudaMalloc(&min_y, sizeof(float) * point_num_));
	checkCudaErrors(cudaMalloc(&min_z, sizeof(float) * point_num_));

	checkCudaErrors(cudaMemcpy(max_x, x_, sizeof(float) * point_num_, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(max_y, y_, sizeof(float) * point_num_, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(max_z, z_, sizeof(float) * point_num_, cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaMemcpy(min_x, x_, sizeof(float) * point_num_, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(min_y, y_, sizeof(float) * point_num_, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(min_z, z_, sizeof(float) * point_num_, cudaMemcpyDeviceToDevice));

	int points_num = point_num_;

	while (points_num > 1) {
		int half_points_num = (points_num - 1) / 2 + 1;
		int block_x = (half_points_num > BLOCK_SIZE_X) ? BLOCK_SIZE_X : half_points_num;
		int grid_x = (half_points_num - 1) / block_x + 1;

		findMax<<<grid_x, block_x>>>(max_x, max_y, max_z, points_num, half_points_num);
		checkCudaErrors(cudaGetLastError());

		findMin<<<grid_x, block_x>>>(min_x, min_y, min_z, points_num, half_points_num);
		checkCudaErrors(cudaGetLastError());

		points_num = half_points_num;
	}

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(&max_x_, max_x, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&max_y_, max_y, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&max_z_, max_z, sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(&min_x_, min_x, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&min_y_, min_y, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&min_z_, min_z, sizeof(float), cudaMemcpyDeviceToHost));

	max_b_x_ = static_cast<int> (floor(max_x_ / voxel_x_));
	max_b_y_ = static_cast<int> (floor(max_y_ / voxel_y_));
	max_b_z_ = static_cast<int> (floor(max_z_ / voxel_z_));

	min_b_x_ = static_cast<int> (floor(min_x_ / voxel_x_));
	min_b_y_ = static_cast<int> (floor(min_y_ / voxel_y_));
	min_b_z_ = static_cast<int> (floor(min_z_ / voxel_z_));

	vgrid_x_ = max_b_x_ - min_b_x_ + 1;
	vgrid_y_ = max_b_y_ - min_b_y_ + 1;
	vgrid_z_ = max_b_z_ - min_b_z_ + 1;

	checkCudaErrors(cudaFree(max_x));
	checkCudaErrors(cudaFree(max_y));
	checkCudaErrors(cudaFree(max_z));

	checkCudaErrors(cudaFree(min_x));
	checkCudaErrors(cudaFree(min_y));
	checkCudaErrors(cudaFree(min_z));
}

__device__ int voxelId(float x, float y, float z,
							float voxel_x, float voxel_y, float voxel_z,
							int min_b_x, int min_b_y, int min_b_z,
							int vgrid_x, int vgrid_y, int vgrid_z)
{
	int id_x = static_cast<int>(floorf(x / voxel_x) - static_cast<float>(min_b_x));
	int id_y = static_cast<int>(floorf(y / voxel_y) - static_cast<float>(min_b_y));
	int id_z = static_cast<int>(floorf(z / voxel_z) - static_cast<float>(min_b_z));

	return (id_x + id_y * vgrid_x + id_z * vgrid_x * vgrid_y);
}

__global__ void countPointsPerVoxel(float *x, float *y, float *z, int point_num, int *points_per_voxel, int voxel_num,
										int vgrid_x, int vgrid_y, int vgrid_z,
										float voxel_x, float voxel_y, float voxel_z,
										int min_b_x, int min_b_y, int min_b_z)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < point_num; i += blockDim.x * gridDim.x) {
		int voxel_id = voxelId(x[i], y[i], z[i], voxel_x, voxel_y, voxel_z, min_b_x, min_b_y, min_b_z, vgrid_x, vgrid_y, vgrid_z);

		atomicAdd(points_per_voxel + voxel_id, 1);
	}
}

__global__ void insertPointsToGrid(float *x, float *y, float *z, int point_num, int voxel_num,
									int vgrid_x, int vgrid_y, int vgrid_z,
									float voxel_x, float voxel_y, float voxel_z,
									int min_b_x, int min_b_y, int min_b_z,
									int *writing_location, int *point_ids)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < points_num; i += blockDim.x * gridDim.x) {
		int voxel_id = voxelId(x[i], y[i], z[i], voxel_x, voxel_y, voxel_z,
								min_b_x, min_b_y, min_b_z, vgrid_x, vgrid_y, vgrid_z);

		int loc = atomicAdd(writing_locations + voxel_id, 1);

		point_ids[loc] = i;
	}
}


void GVoxelGrid::insertPointsToVoxels()
{
	if (starting_point_ids_ != NULL) {
		checkCudaErrors(cudaFree(starting_point_ids_));
		starting_point_ids_ = NULL;
	}

	if (point_ids_ != NULL) {
		checkCudaErrors(cudaFree(point_ids_));
		point_ids_ = NULL;
	}

	int block_x = (point_num_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : point_num_;
	int grid_x = (point_num_ - 1) / block_x + 1;

	countPointsPerVoxel<<<grid_x, block_x>>>(x_, y_, z_, point_num_, points_per_voxel_, voxel_num_,
												vgrid_x_, vgrid_y_, vgrid_z_,
												voxel_x_, voxel_y_, voxel_z_,
												min_b_x_, min_b_y_, min_b_z_);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(starting_point_ids_, points_per_voxel_, sizeof(int) * voxel_num_, cudaMemcpyDeviceToDevice));

	GUtilities::exclusiveScan(starting_point_ids_, voxel_num_ + 1);

	int *writing_location;

	checkCudaErrors(cudaMalloc(&writing_location, sizeof(int) * voxel_num_));
	checkCudaErrors(cudaMemcpy(writing_location, starting_point_ids_, sizeof(int) * voxel_num_, cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaMalloc(&point_ids_, sizeof(int) * point_num_));

	insertPointsToGrid<<<grid_x, block_x>>>(x_, y_, z_, point_num_, voxel_num_,
												vgrid_x_, vgrid_y_, vgrid_z_,
												voxel_x_, voxel_y_, voxel_z_,
												min_b_x_, min_b_y_, min_b_z_,
												writing_location, point_ids_);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(writing_location));
}

void GVoxelGrid::radiusSearch(int2 *nearest_neighbors)
{
	int block_x = (point_num_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : point_num_;
	int grid_x = (point_num_ - 1) / block_x + 1;

	int *max_vid_x, *max_vid_y, *max_vid_z;
	int *min_vid_x, *min_vid_y, *min_vid_z;

	checkCudaErrors(cudaMalloc(&max_vid_x, sizeof(int) * point_num_));
	checkCudaErrors(cudaMalloc(&max_vid_y, sizeof(int) * point_num_));
	checkCudaErrors(cudaMalloc(&max_vid_z, sizeof(int) * point_num_));

	checkCudaErrors(cudaMalloc(&min_vid_x, sizeof(int) * point_num_));
	checkCudaErrors(cudaMalloc(&min_vid_y, sizeof(int) * point_num_));
	checkCudaErrors(cudaMalloc(&min_vid_z, sizeof(int) * point_num_));
}
