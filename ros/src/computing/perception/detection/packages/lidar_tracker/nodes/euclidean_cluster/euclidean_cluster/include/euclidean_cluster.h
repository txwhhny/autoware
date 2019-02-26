#ifndef GPU_EUCLIDEAN_CLUSTER_H_
#define GPU_EUCLIDEAN_CLUSTER_H_

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>


class GpuEuclideanCluster2 {
public:
	typedef struct {
		int index_value;
		std::vector<int> points_in_cluster;
	} GClusterIndex;

	GpuEuclideanCluster2();

	void setInputPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr input);
	void setThreshold(double threshold);
	void setMinClusterPts(int min_cluster_pts);
	void setMaxClusterPts(int max_cluster_pts);
	void setBlockSizeX(int block_size);

	// Matrix-based: Use adjacency matrix
	void extractClusters();
	void extractClusters(long long &total_time, long long &initial_time, long long &build_matrix, long long &clustering_time, int &iteration_num);

	// Vertex-based: full graph, use adjacent list and atomic operations
	void extractClusters2();
	void extractClusters2(long long &total_time, long long &graph_build_time, long long &clustering_time, int &iteration_num);

	// Edge-based: Full graph, use edge set and atomic operations
	void extractClusters3();
	void extractClusters3(long long &total_time, long long &graph_build_time, long long &clustering_time, int &iteration_num);

	// Matrix-based: Use octree to fasten matrix build
	void extractClusters4();
	void extractClusters4(long long &total_time, long long &initial_time, long long &build_matrix, long long &clustering_time, int &iteration_num);

	// Edge-based: Full graph, use octree
	void extractClusters5();
	void extractClusters5(long long &total_time, long long &graph_build_time, long long &clustering_time, int &iteration_num);

	// Vertex-based: Full graph, use octree
	void extractClusters6();
	void extractClusters6(long long &total_time, long long &graph_build_time, long long &clustering_time, int &iteration_num);

	std::vector<GClusterIndex> getOutput();

	~GpuEuclideanCluster2();

	// Measure the graph density
	float density();

private:
	void initClusters();

	void renamingClusters(int *cluster_names, int *cluster_location, int point_num);

	// Common

	// For matrix-based method
	void blockClusteringWrapper(float *x, float *y, float *z, int point_num, int *cluster_name, float threshold);

	void clusterMarkWrapper(int *cluster_list, int *cluster_location, int cluster_num);

	void clusterCollectorWrapper(int *new_cluster_list, int new_cluster_num);

	void buildClusterMatrixWrapper(float *x, float *y, float *z, int *cluster_name, int *cluster_location, int *matrix, int point_num, int cluster_num, float threshold);

	void buildClusterMatrixWrapper(int *starting_neighbor_ids, int *neighbor_ids, int *cluster_name, int *cluster_location, int *matrix, int point_num, int cluster_num);

	void mergeLocalClustersWrapper(int *cluster_list, int *matrix, int cluster_num, bool *changed);

	void clusterIntersecCheckWrapper(int *matrix, int *changed_diag, int *hchanged_diag, int sub_mat_size, int sub_mat_offset, int cluster_num);

	void mergeForeignClustersWrapper(int *matrix, int *cluster_list, int shift_level, int sub_mat_size, int sub_mat_offset, int cluster_num, bool *changed);

	void resetClusterListWrapper(int *cluster_list, int cluster_num);

	void applyClusterChangedWrapper(int *cluster_name, int *cluster_list, int *cluster_location, int point_num);

	void rebuildMatrixWrapper(int *old_matrix, int *merged_cluster_list,
								int *new_matrix, int *new_cluster_location,
								int old_cluster_num, int new_cluster_num);


	float *x_, *y_, *z_;
	int point_num_;
	int padded_num_;
	double threshold_;
	int *cluster_name_;
	int *cluster_name_host_;
	int min_cluster_pts_;
	int max_cluster_pts_;
	int cluster_num_;

	int block_size_x_;
};

#ifndef BLOCK_SIZE_X
#define BLOCK_SIZE_X (1024)
#endif

#ifndef GRID_SIZE_Y
#define GRID_SIZE_Y (1024)
#endif

#endif
