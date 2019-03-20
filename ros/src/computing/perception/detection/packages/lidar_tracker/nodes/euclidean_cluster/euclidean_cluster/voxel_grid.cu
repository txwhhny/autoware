#include "include/voxel_grid.h"
#include "include/utilities.h"
#include <cfloat>
#include <iostream>

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
	voxel_ids_(NULL),
	starting_point_ids_(NULL),
	point_ids_(NULL)
{
}

__global__ void initPointIds(int *point_ids, int point_num)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < point_num; i += blockDim.x * gridDim.x) {
		point_ids[i] = i;
	}
}

// x, y, z are GPU buffers
GVoxelGrid::GVoxelGrid(float *x, float *y, float *z, int point_num, float voxel_x, float voxel_y, float voxel_z)
{
	x_ = x;
	y_ = y;
	z_ = z;
	point_num_ = point_num;
	voxel_x_ = voxel_x;
	voxel_y_ = voxel_y;
	voxel_z_ = voxel_z;
	voxel_ids_ = NULL;
	starting_point_ids_ = NULL;
	point_ids_ = NULL;

	checkCudaErrors(cudaMalloc(&point_ids_, sizeof(int) * point_num_));

	int block_x = (point_num_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : point_num_;
	int grid_x = (point_num_ - 1) / block_x + 1;

	initPointIds<<<grid_x, block_x>>>(point_ids_, point_num_);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#ifdef DEBUG_
	struct timeval start, end;

	gettimeofday(&start, NULL);
#endif

	findBoundaries();
#ifdef DEBUG_
	gettimeofday(&end, NULL);

	std::cout << "FindBoundaries = " << timeDiff(start, end) << std::endl;
#endif

	// Allocate empty voxel grid

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
#ifdef DEBUG_
	struct timeval start, end;
#endif

	float *max_x, *max_y, *max_z, *min_x, *min_y, *min_z;

#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif
	checkCudaErrors(cudaMalloc(&max_x, sizeof(float) * point_num_));
	checkCudaErrors(cudaMalloc(&max_y, sizeof(float) * point_num_));
	checkCudaErrors(cudaMalloc(&max_z, sizeof(float) * point_num_));
	checkCudaErrors(cudaMalloc(&min_x, sizeof(float) * point_num_));
	checkCudaErrors(cudaMalloc(&min_y, sizeof(float) * point_num_));
	checkCudaErrors(cudaMalloc(&min_z, sizeof(float) * point_num_));

#ifdef DEBUG_
	gettimeofday(&end, NULL);

	std::cout << "malloc min and maxxyz = " << timeDiff(start, end) << std::endl;
#endif

#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif
	checkCudaErrors(cudaMemcpy(max_x, x_, sizeof(float) * point_num_, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(max_y, y_, sizeof(float) * point_num_, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(max_z, z_, sizeof(float) * point_num_, cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaMemcpy(min_x, x_, sizeof(float) * point_num_, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(min_y, y_, sizeof(float) * point_num_, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(min_z, z_, sizeof(float) * point_num_, cudaMemcpyDeviceToDevice));

#ifdef DEBUG_
	gettimeofday(&end, NULL);

	std::cout << "memcpy min and maxxyz = " << timeDiff(start, end) << std::endl;
#endif

	int points_num = point_num_;

#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif
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

#ifdef DEBUG_
	gettimeofday(&end, NULL);

	std::cout << "findMax and min = " << timeDiff(start, end) << std::endl;
#endif

#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif
	checkCudaErrors(cudaMemcpy(&max_x_, max_x, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&max_y_, max_y, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&max_z_, max_z, sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(&min_x_, min_x, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&min_y_, min_y, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&min_z_, min_z, sizeof(float), cudaMemcpyDeviceToHost));

#ifdef DEBUG_
	gettimeofday(&end, NULL);

	std::cout << "copy max and minxyz = " << timeDiff(start, end) << std::endl;
#endif

	max_b_x_ = static_cast<int> (floor(max_x_ / voxel_x_));
	max_b_y_ = static_cast<int> (floor(max_y_ / voxel_y_));
	max_b_z_ = static_cast<int> (floor(max_z_ / voxel_z_));

	min_b_x_ = static_cast<int> (floor(min_x_ / voxel_x_));
	min_b_y_ = static_cast<int> (floor(min_y_ / voxel_y_));
	min_b_z_ = static_cast<int> (floor(min_z_ / voxel_z_));

	vgrid_x_ = max_b_x_ - min_b_x_ + 1;
	vgrid_y_ = max_b_y_ - min_b_y_ + 1;
	vgrid_z_ = max_b_z_ - min_b_z_ + 1;


#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif

	checkCudaErrors(cudaFree(max_x));
	checkCudaErrors(cudaFree(max_y));
	checkCudaErrors(cudaFree(max_z));

	checkCudaErrors(cudaFree(min_x));
	checkCudaErrors(cudaFree(min_y));
	checkCudaErrors(cudaFree(min_z));

#ifdef DEBUG_
	gettimeofday(&end, NULL);

	std::cout << "Free max and minxyz = " << timeDiff(start, end) << std::endl;
#endif
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

__global__ void computeVoxelId(float *x, float *y, float *z, int point_num,
								int *vid_of_point,
								int vgrid_x, int vgrid_y, int vgrid_z,
								float voxel_x, float voxel_y, float voxel_z,
								int min_b_x, int min_b_y, int min_b_z)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < point_num; i += blockDim.x * gridDim.x) {
		vid_of_point[i] = voxelId(x[i], y[i], z[i], voxel_x, voxel_y, voxel_z, min_b_x, min_b_y, min_b_z, vgrid_x, vgrid_y, vgrid_z);
	}
}

__global__ void markNonEmptyVoxels(int *vid_of_point, int point_num, int *mark)
{
	int i;

	for (i = threadIdx.x + blockIdx.x * blockDim.x; i < point_num - 1; i += blockDim.x * gridDim.x) {
		mark[i] = (vid_of_point[i] < vid_of_point[i + 1]) ? 1 : 0;
	}

	if (i == point_num - 1) {
		mark[i] = 1;
	}
}

__global__ void collectNonEmptyVoxels(int *vid_of_point, int point_num, int *writing_location,
										int *voxel_ids, int *starting_point_ids)
{
	int i, loc;

	for (i = threadIdx.x + blockIdx.x * blockDim.x; i < point_num - 1; i += blockDim.x * gridDim.x) {
		if (vid_of_point[i] < vid_of_point[i + 1]) {
			loc = writing_location[i];

			voxel_ids[loc] = vid_of_point[i];
			starting_point_ids[loc + 1] = i + 1;
		}
	}

	if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
		starting_point_ids[0] = 0;
	}

	if (i == point_num - 1) {
		loc = writing_location[i];

		voxel_ids[loc] = vid_of_point[i];
		starting_point_ids[loc + 1] = i + 1;
	}
}


void GVoxelGrid::insertPointsToVoxels()
{
#ifdef DEBUG_
	struct timeval start, end;
#endif
	int block_x = (point_num_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : point_num_;
	int grid_x = (point_num_ - 1) / block_x + 1;

	int *vid_of_point;

#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif
	checkCudaErrors(cudaMalloc(&vid_of_point, sizeof(int) * point_num_));

	computeVoxelId<<<grid_x, block_x>>>(x_, y_, z_, point_num_,	vid_of_point,
											vgrid_x_, vgrid_y_, vgrid_z_,
											voxel_x_, voxel_y_, voxel_z_,
											min_b_x_, min_b_y_, min_b_z_);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
#ifdef DEBUG_
	gettimeofday(&end, NULL);

	std::cout << "ComputeVoxelId = " << timeDiff(start, end) << std::endl;
#endif



#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif
	GUtilities::sortByKey(vid_of_point, point_ids_, point_num_);
#ifdef DEBUG_
	gettimeofday(&end, NULL);

	std::cout << "sortbyKey = " << timeDiff(start, end) << std::endl;
#endif

	int *mark;

#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif

	checkCudaErrors(cudaMalloc(&mark, sizeof(int) * (point_num_ + 1)));

	markNonEmptyVoxels<<<grid_x, block_x>>>(vid_of_point, point_num_, mark);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#ifdef DEBUG_
	gettimeofday(&end, NULL);

	std::cout << "markNonEmpty = " << timeDiff(start, end) << std::endl;
#endif

#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif
	GUtilities::exclusiveScan(mark, point_num_ + 1, &voxel_num_);

	if (voxel_ids_ != NULL) {
		checkCudaErrors(cudaFree(voxel_ids_));
		voxel_ids_ = NULL;
	}
	checkCudaErrors(cudaMalloc(&voxel_ids_, sizeof(int) * voxel_num_));

	if (starting_point_ids_ != NULL) {
		checkCudaErrors(cudaFree(starting_point_ids_));
		starting_point_ids_ = NULL;
	}
	checkCudaErrors(cudaMalloc(&starting_point_ids_, sizeof(int) * (voxel_num_ + 1)));

	collectNonEmptyVoxels<<<grid_x, block_x>>>(vid_of_point, point_num_, mark, voxel_ids_, starting_point_ids_);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#ifdef DEBUG_
	gettimeofday(&end, NULL);

	std::cout << "collectNonEmpty = " << timeDiff(start, end) << std::endl;
#endif

	checkCudaErrors(cudaFree(vid_of_point));
	checkCudaErrors(cudaFree(mark));

#ifdef DEBUG_
	std::cout << "Voxel num = " << voxel_num_ << std::endl;
#endif
}


// Return the index of vid in the voxel id list. Return -1 if not found.
__device__ int voxelSearch(int *voxel_ids, int voxel_num, int vid)
{
	int left = 0, right = voxel_num - 1;
	int middle;

	while (left <= right) {
		middle = (left + right) / 2;
		int candidate = voxel_ids[middle];

		if (vid == candidate) {
			return middle;
		}

		left = (vid > candidate) ? middle + 1 : left;
		right = (vid < candidate) ? middle - 1 : right;
	}

	return -1;
}
__global__ void boundariesSearch(float *x, float *y, float *z, int point_num, float radius,
									int vgrid_x, int vgrid_y, int vgrid_z,
									float voxel_x, float voxel_y, float voxel_z,
									int max_b_x, int max_b_y, int max_b_z,
									int min_b_x, int min_b_y, int min_b_z,
									int *voxel_ids, int voxel_num,
									int *candidate_vids)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < point_num; i += blockDim.x * gridDim.x) {
		float tx = x[i];
		float ty = y[i];
		float tz = z[i];

		int max_id_x = static_cast<int>(floorf((tx + radius) / voxel_x));
		int max_id_y = static_cast<int>(floorf((ty + radius) / voxel_y));
		int max_id_z = static_cast<int>(floorf((tz + radius) / voxel_z));

		int min_id_x = static_cast<int>(floorf((tx - radius) / voxel_x));
		int min_id_y = static_cast<int>(floorf((ty - radius) / voxel_y));
		int min_id_z = static_cast<int>(floorf((tz - radius) / voxel_z));

		max_id_x = (max_id_x > max_b_x) ? max_b_x - min_b_x : max_id_x - min_b_x;
		max_id_y = (max_id_y > max_b_y) ? max_b_y - min_b_y : max_id_y - min_b_y;
		max_id_z = (max_id_z > max_b_z) ? max_b_z - min_b_z : max_id_z - min_b_z;

		min_id_x = (min_id_x < min_b_x) ? 0 : min_id_x - min_b_x;
		min_id_y = (min_id_y < min_b_y) ? 0 : min_id_y - min_b_y;
		min_id_z = (min_id_z < min_b_z) ? 0 : min_id_z - min_b_z;

		// Number of candidate voxels
		int voxel_count = 0;

		for (int j = min_id_x; j <= max_id_x; j++) {
			for (int k = min_id_y; k <= max_id_y; k++) {
				for (int l = min_id_z; l <= max_id_z; l++) {
					int vid = j + k * vgrid_x + l * vgrid_x * vgrid_y;

					candidate_vids[i + point_num * voxel_count] = voxelSearch(voxel_ids, voxel_num, vid);
					voxel_count++;
				}
			}
		}
	}
}

__global__ void initCandidateVids(int *candidate_vids, int size)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x) {
		candidate_vids[i] = -1;
	}
}

__global__ void neighborsCountEdge(float *x, float *y, float *z, int point_num,
									int *candidate_vids,
									int *starting_point_ids, int *point_ids,
									int *neighbor_count,
									float threshold)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < point_num; i += blockDim.x * gridDim.x) {
		float tx = x[i];
		float ty = y[i];
		float tz = z[i];
		int count = 0;

		for (int candidate_num = 0; candidate_num < 27; candidate_num++) {
			int vid = candidate_vids[i + candidate_num * point_num];

			if (vid >= 0) {
				int pid_start = starting_point_ids[vid];
				int pid_end = starting_point_ids[vid + 1];

				for (int k = pid_start; k < pid_end; k++) {
					int pid = point_ids[k];

					if (i < pid && norm3df(tx - x[pid], ty - y[pid], tz - z[pid]) < threshold) {
						count++;
					}
				}
			}
		}

		__syncthreads();

		neighbor_count[i] = count;
	}
}

__global__ void neighborsCountVertex(float *x, float *y, float *z, int point_num,
										int *candidate_vids,
										int *starting_point_ids, int *point_ids,
										int *neighbor_count,
										float threshold)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < point_num; i += blockDim.x * gridDim.x) {
		float tx = x[i];
		float ty = y[i];
		float tz = z[i];
		int count = 0;

		for (int candidate_num = 0; candidate_num < 27; candidate_num++) {
			int vid = candidate_vids[i + candidate_num * point_num];

			if (vid >= 0) {
				int pid_start = starting_point_ids[vid];
				int pid_end = starting_point_ids[vid + 1];

				for (int k = pid_start; k < pid_end; k++) {
					int pid = point_ids[k];

					if (i != pid && norm3df(tx - x[pid], ty - y[pid], tz - z[pid]) < threshold) {
						count++;
					}
				}
			}
		}

		__syncthreads();

		neighbor_count[i] = count;
	}
}

__global__ void neighborsSearchVertex(float *x, float *y, float *z, int point_num,
										int *candidate_vids,
										int *starting_point_ids, int *point_ids,
										int *starting_neighbor_ids, int *neighbor_ids,
										float threshold)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < point_num; i += blockDim.x * gridDim.x) {
		float tx = x[i];
		float ty = y[i];
		float tz = z[i];
		int location = starting_neighbor_ids[i];

		for (int candidate_num = 0; candidate_num < 27; candidate_num++) {
			int vid = candidate_vids[i + candidate_num * point_num];

			if (vid >= 0) {
				int pid_start = starting_point_ids[vid];
				int pid_end = starting_point_ids[vid + 1];

				for (int k = pid_start; k < pid_end; k++) {
					int pid = point_ids[k];


					if (i != pid && norm3df(tx - x[pid], ty - y[pid], tz - z[pid]) < threshold) {
						neighbor_ids[location++] = pid;
					}
				}
			}
		}
	}
}

void GVoxelGrid::radiusSearch(int **starting_neighbor_ids, int **neighbor_ids, int *neighbor_num, float radius)
{
#ifdef DEBUG_
	struct timeval start, end;
#endif
	int block_x, grid_x;


	int *candidate_vids;

#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif
	checkCudaErrors(cudaMalloc(&candidate_vids, sizeof(int) * (point_num_ * 27)));

#ifdef DEBUG_
	gettimeofday(&end, NULL);

	std::cout << "Malloc candidateVids = " << timeDiff(start, end) << std::endl;
#endif

	block_x = (point_num_ * 27 > BLOCK_SIZE_X) ? BLOCK_SIZE_X : point_num_ * 27;
	grid_x = (point_num_ * 27 - 1) / block_x + 1;

#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif

	initCandidateVids<<<grid_x, block_x>>>(candidate_vids, point_num_ * 27);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#ifdef DEBUG_
	gettimeofday(&end, NULL);

	std::cout << "init candidate vids = " << timeDiff(start, end) << std::endl;
#endif


	block_x = (point_num_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : point_num_;
	grid_x = (point_num_ - 1) / block_x + 1;

#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif
	boundariesSearch<<<grid_x, block_x>>>(x_, y_, z_, point_num_, radius,
											vgrid_x_, vgrid_y_, vgrid_z_,
											voxel_x_, voxel_y_, voxel_z_,
											max_b_x_, max_b_y_, max_b_z_,
											min_b_x_, min_b_y_, min_b_z_,
											voxel_ids_, voxel_num_,
											candidate_vids);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#ifdef DEBUG_
	gettimeofday(&end, NULL);

	std::cout << "boundariesSearch = " << timeDiff(start, end) << std::endl;
#endif


#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif

	checkCudaErrors(cudaMalloc(starting_neighbor_ids, sizeof(int) * (point_num_ + 1)));

	neighborsCountVertex<<<grid_x, block_x>>>(x_, y_, z_, point_num_,
												candidate_vids,
												starting_point_ids_, point_ids_,
												*starting_neighbor_ids, radius);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#ifdef DEBUG_
	gettimeofday(&end, NULL);

	std::cout << "neighborCount = " << timeDiff(start, end) << std::endl;
#endif

#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif

	GUtilities::exclusiveScan(*starting_neighbor_ids, point_num_ + 1, neighbor_num);

#ifdef DEBUG_
	gettimeofday(&end, NULL);

	std::cout << "exclusiveScan = " << timeDiff(start, end) << std::endl;
#endif

	if (*neighbor_num == 0) {
		checkCudaErrors(cudaFree(candidate_vids));
		checkCudaErrors(cudaFree(*starting_neighbor_ids));
		*starting_neighbor_ids = NULL;
		*neighbor_ids = NULL;

		return;
	}

#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif

#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif
	checkCudaErrors(cudaMalloc(neighbor_ids, sizeof(int) * (*neighbor_num)));
#ifdef DEBUG_
	gettimeofday(&end, NULL);

	std::cout << "neighborids malloc = " << timeDiff(start, end) << std::endl;
#endif

#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif
	neighborsSearchVertex<<<grid_x, block_x>>>(x_, y_, z_, point_num_,
												candidate_vids,
												starting_point_ids_, point_ids_,
												*starting_neighbor_ids,
												*neighbor_ids,
												radius);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#ifdef DEBUG_
	gettimeofday(&end, NULL);

	std::cout << "neighborrSearches = " << timeDiff(start, end) << std::endl;
#endif


	checkCudaErrors(cudaFree(candidate_vids));
}


__global__ void edgeSetBuild(float *x, float *y, float *z, int point_num,
								int *candidate_vids,
								int *starting_point_ids, int *point_ids,
								int *starting_neighbor_ids, int2 *edge_set,
								float threshold)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < point_num; i += blockDim.x * gridDim.x) {
		float tx = x[i];
		float ty = y[i];
		float tz = z[i];
		int location = starting_neighbor_ids[i];
		int2 val;

		val.x = i;

		for (int candidate_num = 0; candidate_num < 27; candidate_num++) {
			int vid = candidate_vids[i + candidate_num * point_num];

			if (vid >= 0) {
				int pid_start = starting_point_ids[vid];
				int pid_end = starting_point_ids[vid + 1];

				for (int k = pid_start; k < pid_end; k++) {
					int pid = point_ids[k];

					if (i < pid && norm3df(tx - x[pid], ty - y[pid], tz - z[pid]) < threshold) {
						val.y = pid;
						edge_set[location++] = val;
					}
				}
			}
		}
	}
}


/* Basically perform a radius search and
 * record the result into edge_set
 */
void GVoxelGrid::createEdgeSet(int2 **edge_set, int *edge_num, float radius)
{
#ifdef DEBUG_
	struct timeval start, end;
#endif

	int block_x, grid_x;


	int *candidate_vids;

#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif
	checkCudaErrors(cudaMalloc(&candidate_vids, sizeof(int) * (point_num_ * 27)));

#ifdef DEBUG_
	gettimeofday(&end, NULL);
	std::cout << "malloc candidate_vids = " << timeDiff(start, end) << std::endl;
#endif


	block_x = (point_num_ * 27 > BLOCK_SIZE_X) ? BLOCK_SIZE_X : point_num_ * 27;
	grid_x = (point_num_ * 27 - 1) / block_x + 1;

#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif

	initCandidateVids<<<grid_x, block_x>>>(candidate_vids, point_num_ * 27);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#ifdef DEBUG_
	gettimeofday(&end, NULL);

	std::cout << "init candidate vids = " << timeDiff(start, end) << std::endl;
#endif

	block_x = (point_num_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : point_num_;
	grid_x = (point_num_ - 1) / block_x + 1;

#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif
	boundariesSearch<<<grid_x, block_x>>>(x_, y_, z_, point_num_, radius,
											vgrid_x_, vgrid_y_, vgrid_z_,
											voxel_x_, voxel_y_, voxel_z_,
											max_b_x_, max_b_y_, max_b_z_,
											min_b_x_, min_b_y_, min_b_z_,
											voxel_ids_, voxel_num_,
											candidate_vids);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#ifdef DEBUG_
	gettimeofday(&end, NULL);

	std::cout << "boundariesSearch = " << timeDiff(start, end) << std::endl;
#endif


	int *starting_neighbor_ids;

#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif
	checkCudaErrors(cudaMalloc(&starting_neighbor_ids, sizeof(int) * (point_num_ + 1)));

	neighborsCountEdge<<<grid_x, block_x>>>(x_, y_, z_, point_num_,
												candidate_vids,
												starting_point_ids_, point_ids_,
												starting_neighbor_ids, radius);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#ifdef DEBUG_
	gettimeofday(&end, NULL);

	std::cout << "NeighborCount = " << timeDiff(start, end) << std::endl;
#endif

#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif

	GUtilities::exclusiveScan(starting_neighbor_ids, point_num_ + 1, edge_num);

#ifdef DEBUG_
	gettimeofday(&end, NULL);
	std::cout << "exclusiveScan = " << timeDiff(start, end) << std::endl;
#endif

	if (*edge_num == 0) {
		checkCudaErrors(cudaFree(candidate_vids));
		checkCudaErrors(cudaFree(starting_neighbor_ids));
		*edge_set = NULL;

		return;
	}

#ifdef DEBUG_
	gettimeofday(&start, NULL);
#endif

	checkCudaErrors(cudaMalloc(edge_set, sizeof(int2) * (*edge_num)));

	edgeSetBuild<<<grid_x, block_x>>>(x_, y_, z_, point_num_, candidate_vids,
										starting_point_ids_, point_ids_,
										starting_neighbor_ids, *edge_set,
										radius);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#ifdef DEBUG_
	gettimeofday(&end, NULL);

	std::cout << "edgeSetBuild = " << timeDiff(start, end) << std::endl;
#endif


	checkCudaErrors(cudaFree(candidate_vids));
	checkCudaErrors(cudaFree(starting_neighbor_ids));
}

void GVoxelGrid::createAdjacentList(int **starting_neighbor_ids, int **adjacent_list, int *list_size, float radius)
{
	radiusSearch(starting_neighbor_ids, adjacent_list, list_size, radius);
}

__global__ void matrixBuild(float *x, float *y, float *z, int point_num,
							int *candidate_vids,
							int *starting_point_ids, int *point_ids,
							int *point_labels, int label_num,
							int *matrix, bool *is_zero, float threshold)
{
	bool t_is_zero = true;

	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < point_num; i += blockDim.x * gridDim.x) {
		float tx = x[i];
		float ty = y[i];
		float tz = z[i];
		int col = point_labels[i];

		for (int candidate_num = 0; candidate_num < 27; candidate_num++) {
			int vid = candidate_vids[i + candidate_num * point_num];

			if (vid >= 0) {
				int pid_start = starting_point_ids[vid];
				int pid_end = starting_point_ids[vid + 1];

				for (int k = pid_start; k < pid_end; k++) {
					int pid = point_ids[k];
					int row = point_labels[pid];

					if (row < col && norm3df(tx - x[pid], ty - y[pid], tz - z[pid]) < threshold) {
						matrix[row * label_num + col] = 1;
						t_is_zero = false;
					}
				}
			}
		}
	}

	if (!t_is_zero) {
		*is_zero = false;
	}
}

//#define DEBUG2_ 1
void GVoxelGrid::createLabeledMatrix(int *point_labels, int label_num, int **matrix, bool *is_zero, float radius)
{
	if (label_num == 0) {
		*matrix = NULL;
		return;
	}

#ifdef DEBUG2_
	struct timeval start, end;

	gettimeofday(&start, NULL);
#endif

	checkCudaErrors(cudaMalloc(matrix, sizeof(int) * label_num * label_num));
	checkCudaErrors(cudaMemset(*matrix, 0, sizeof(int) * label_num * label_num));

#ifdef DEBUG2_
	gettimeofday(&end, NULL);

	std::cout << "Matrix memset = " << timeDiff(start, end) << std::endl;
#endif

	// neighbor search and set to matrix
	int block_x, grid_x;

	int *candidate_vids;

	checkCudaErrors(cudaMalloc(&candidate_vids, sizeof(int) * (point_num_ * 27)));

	block_x = (point_num_ * 27 > BLOCK_SIZE_X) ? BLOCK_SIZE_X : point_num_ * 27;
	grid_x = (point_num_ * 27 - 1) / block_x + 1;

#ifdef DEBUG2_
	gettimeofday(&start, NULL);
#endif

	initCandidateVids<<<grid_x, block_x>>>(candidate_vids, point_num_ * 27);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#ifdef DEBUG2_
	gettimeofday(&end, NULL);

	std::cout << "initCandidateVids = " << timeDiff(start, end) << std::endl;
#endif


	block_x = (point_num_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : point_num_;
	grid_x = (point_num_ - 1) / block_x + 1;

#ifdef DEBUG2_
	gettimeofday(&start, NULL);
#endif

	boundariesSearch<<<grid_x, block_x>>>(x_, y_, z_, point_num_, radius,
											vgrid_x_, vgrid_y_, vgrid_z_,
											voxel_x_, voxel_y_, voxel_z_,
											max_b_x_, max_b_y_, max_b_z_,
											min_b_x_, min_b_y_, min_b_z_,
											voxel_ids_, voxel_num_,
											candidate_vids);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#ifdef DEBUG2_
	gettimeofday(&end, NULL);

	std::cout << "boundariesSearch = " << timeDiff(start, end) << std::endl;
#endif

	bool *g_is_zero;

	checkCudaErrors(cudaMalloc(&g_is_zero, sizeof(bool)));

	*is_zero = true;

	checkCudaErrors(cudaMemcpy(g_is_zero, is_zero, sizeof(bool), cudaMemcpyHostToDevice));

#ifdef DEBUG2_
	gettimeofday(&start, NULL);
#endif

	matrixBuild<<<grid_x, block_x>>>(x_, y_, z_, point_num_,
										candidate_vids,
										starting_point_ids_, point_ids_,
										point_labels, label_num, *matrix, g_is_zero, radius);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#ifdef DEBUG2_
	gettimeofday(&end, NULL);

	std::cout << "matrixBuild = " << timeDiff(start, end) << std::endl;
#endif

	checkCudaErrors(cudaMemcpy(is_zero, g_is_zero, sizeof(bool), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(candidate_vids));
	checkCudaErrors(cudaFree(g_is_zero));
}


GVoxelGrid::~GVoxelGrid()
{
	if (voxel_ids_ != NULL) {
		checkCudaErrors(cudaFree(voxel_ids_));
	}

	if (starting_point_ids_ != NULL) {
		checkCudaErrors(cudaFree(starting_point_ids_));
	}

	if (point_ids_ != NULL) {
		checkCudaErrors(cudaFree(point_ids_));
	}

	voxel_ids_ = starting_point_ids_ = point_ids_ = NULL;

	x_ = y_ = z_ = NULL;
}
