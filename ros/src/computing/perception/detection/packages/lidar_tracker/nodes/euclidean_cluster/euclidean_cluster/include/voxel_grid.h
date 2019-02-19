#include <cuda.h>


class GVoxelGrid {
public:
	GVoxelGrid();

	GVoxelGrid(float *x, float *y, float *z, int point_num);

	void buildGrid();

	/* Customized radiusSearch():
	 * For every point q, search for two nearest neighbors.
	 * One has index larger than q, one has index smaller than q.
	 * If no such point exist, record -1
	 */
	void radiusSearch(int2 *nearest_neighbors);

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

	int *points_per_voxel_;
	int *starting_point_ids_;
	int *point_ids_;
};
