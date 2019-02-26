#ifndef EUCLIDEAN_VGRID_H_
#define EUCLIDEAN_VGRID_H_

#include <cuda.h>


class GVoxelGrid {
public:
	GVoxelGrid();

	GVoxelGrid(float *x, float *y, float *z, int point_num,
				float voxel_x, float voxel_y, float voxel_z);

	void buildGrid();

	/* Customized radiusSearch():
	 * For each point pid, search for its neighbors.
	 * If search_all = true, search for all of its neighbors.
	 * If search_all = false, only search for neighbors whose indexes is larger than pid.
	 * By default, search_all = false.
	 */
	void radiusSearch(int **starting_neighbor_ids, int **neighbor_ids, int *neighbor_num, float radius, bool search_all = false);

	/* Produce a graph in the form of an edge set from the cloud.
	 * Each vertex is a point.
	 * An edge connects two points whose distance is less than radius.
	 */
	void createEdgeSet(int2 **edge_set, int *edge_num, float radius);

	/* Produce a graph in the form of an adjacent list from the cloud.
	 * Each vertex is a point.
	 * An edge connects two points whose distance is less than radius.
	 */
	void createAdjacentList(int **starting_neighbor_ids, int **adjacent_list, int *list_size, float radius);

	/* Produce a matrix that describes the relationship
	 * between clusters of points. is_zero is set if
	 * the matrix is zero, so we may not have to traverse through
	 * the matrix for clustering.
	 */
	void createLabeledMatrix(int *point_labels, int label_num, int **matrix, bool *is_zero, float radius);

	~GVoxelGrid();

private:

	void findBoundaries();

	void insertPointsToVoxels();

	float *x_, *y_, *z_;
	int point_num_;
	int voxel_num_;
	float max_x_, max_y_, max_z_;
	float min_x_, min_y_, min_z_;
	float voxel_x_, voxel_y_, voxel_z_;
	int max_b_x_, max_b_y_, max_b_z_;
	int min_b_x_, min_b_y_, min_b_z_;
	int vgrid_x_, vgrid_y_, vgrid_z_;

	int *voxel_ids_;
	int *starting_point_ids_;
	int *point_ids_;
};

#endif
