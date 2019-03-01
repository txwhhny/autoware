#include "include/euclidean_cluster.h"
#include "include/utilities.h"
#include "include/voxel_grid.h"
#include <cuda.h>


void GpuEuclideanCluster2::extractClusters4(long long &total_time, long long &initial_time, long long &build_matrix, long long &clustering_time, int &iteration_num)
{
	std::cout << "MATRIX-BASED 2: Use octree" << std::endl;
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

#ifdef DEBUG_
	std::cout << "blockClustering = " << timeDiff(start, end) << std::endl;
#endif

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

	applyClusterChangedWrapper(cluster_name_, cluster_list, cluster_location, point_num_);
	gettimeofday(&end, NULL);

	total_time += timeDiff(start, end);

#ifdef DEBUG_
	std::cout << "Collect remaining clusters: " << timeDiff(start, end) << std::endl;
#endif

	cluster_num_ = new_cluster_num;

#ifdef DEBUG_
	std::cout << "Number of remaining cluste: " << cluster_num_ << std::endl;
#endif

	gettimeofday(&start, NULL);
	// Build relation matrix which describe the current relationship between clusters
	int *matrix;
	bool is_zero = false;

	GVoxelGrid voxel_grid(x_, y_, z_, point_num_, threshold_, threshold_, threshold_);

	voxel_grid.createLabeledMatrix(cluster_name_, cluster_num_, &matrix, &is_zero, threshold_);

	gettimeofday(&end, NULL);

	build_matrix = timeDiff(start, end);
	total_time += timeDiff(start, end);

#ifdef DEBUG_
	std::cout << "Build RC and Matrix = " << timeDiff(start, end) << std::endl;
#endif

	if (is_zero) {
		checkCudaErrors(cudaFree(matrix));
		checkCudaErrors(cudaFree(cluster_location));
		checkCudaErrors(cudaFree(cluster_list));
		checkCudaErrors(cudaFree(check));

		std::cout << "FINAL CLUSTER NUM = " << cluster_num_ << std::endl << std::endl;

		return;
	}

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

#ifdef DEBUG_
			std::cout << "Check intersection " << std::endl;
#endif
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

			std::cout << "New cluster num = " << cluster_num_ << std::endl;

			// Rebuild matrix
			int *new_matrix;

			//std::cout << "cluster_num = " << cluster_num_ << std::endl;
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

	std::cout << "FINAL CLUSTER NUM = " << cluster_num_ << std::endl << std::endl;

	//exit(1);
}

void GpuEuclideanCluster2::extractClusters4()
{
	long long total_time, initial_time, build_matrix, clustering_time;
	int iteration_num;

	extractClusters4(total_time, initial_time, build_matrix, clustering_time, iteration_num);
}
