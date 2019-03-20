#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>
#include <sys/time.h>

#include "gpuectest.h"
#include "euclidean_cluster/include/euclidean_cluster.h"

#define SAMPLE_DIST_ (1024.0)
#define SAMPLE_RAND_ (1024)
#define SAMPLE_SIZE_ (32768)
#define SAMPLE_SIZE_F_ (32768.0)
#define JOINT_DIST_FACTOR_ (0.5)


void GPUECTest::worstCaseEdgeBased()
{
	// Chained graph
	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), SAMPLE_SIZE_, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);

	float d_th = 1.0;

	for (unsigned int i = 0; i < sample_cloud->points.size(); i++) {
		sample_point.x += d_th / 2;
		sample_cloud->points[i] = sample_point;
	}

	GpuEuclideanCluster2 test_sample;

	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);
	test_sample.setInputPoints(sample_cloud);

	struct timeval start, end;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Worst case edge-based - Edge-based: " << timeDiff(start, end) << " usecs" << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Worst case edge-based - Matrix-based: " << timeDiff(start, end) << " usecs" << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Worst case edge-based - Vertex-based: " << timeDiff(start, end) << " usecs" << std::endl << std::endl;
}

void GPUECTest::pointCloudVariationTest()
{
	std::ofstream test_result("/home/anh/euclidean_cluster_test.ods");

	int point_num, disjoint_comp_num, joint_comp_num, point_distance, point_degree;


	// Initialize gpu, produce no output file
	matrixTest();

//	std::cout << "***** Unequal cluster test *****" << std::endl;
//	point_num = 262144;
//	disjoint_comp_num = 128;
//	int point_num_per_joint = 16;
//
//	test_result << "****** Unequal cluster test *****" << std::endl;
//	test_result << "point_num = 262144 disjoint num = 1024 point location is randomized average number of point per joint = 16, point_distance = 4" << std::endl;
//	test_result << "Common difference, E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based, PCL, ";
//	test_result << "E Edge-based,,, E Matrix-based,,,, E Vertex-based,,, RS Edge-based,,, RS Matrix-based,,,, RS Vertex-based,,, PCL,, ";
//	test_result << "E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based" << std::endl;
//
//	for (int common_diff = 0; common_diff < 2 * (point_num - disjoint_comp_num * point_num_per_joint) / (disjoint_comp_num * (disjoint_comp_num - 1)); common_diff++) {
//		std::cout << "Common diff = " << common_diff << std::endl;
//		std::string res = variousSizeClusterTest(point_num, disjoint_comp_num, point_num_per_joint, common_diff);
//
//		test_result << common_diff << "," << res << std::endl;
//	}


	std::cout << "***** POINT CLOUD VARIATION TEST *****" << std::endl;

	//int point_num, disjoint_comp_num, joint_comp_num, point_distance;
	std::cout << "########################### Point Num Variation Test ##################################" << std::endl;
	// Point num variation, fix disjoint_comp_num, joint_comp_num, point_distance
	disjoint_comp_num = 128;
	point_degree = 32;
	point_distance = 4;

	test_result << "****** Point Num Variation Test *****" << std::endl;
	test_result << "point_num varies disjoint num = 128 point degree = 32 point distance = 4" << std::endl;
	test_result << "Point num, E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based, PCL, ";
	test_result << "E Edge-based,,, E Matrix-based,,,, E Vertex-based,,, RS Edge-based,,, RS Matrix-based,,,, RS Vertex-based,,, PCL,, ";
	test_result << "E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based" << std::endl;

	for (point_num = 128 * 32; point_num <= 262144; point_num *= 2) {
		joint_comp_num = (point_num / disjoint_comp_num) / point_degree;
		std::string res = pointCloudVariationTest(point_num, disjoint_comp_num, joint_comp_num, point_distance);
		test_result << point_num << "," << res << std::endl;
	}

	test_result << std::endl << std::endl << std::endl;

	std::cout << "########################### Disjoint Comp Num Variation Test ###########################" << std::endl;

	test_result << "***** Disjoint Comp Num Variation Test *****" << std::endl;
	test_result << "Cluster num varies point num = 262144 point degree = 32 point distance = 4" << std::endl;

	test_result << "Cluster num, E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based, PCL, ";
	test_result << "E Edge-based,,, E Matrix-based,,,, E Vertex-based,,, RS Edge-based,,, RS Matrix-based,,,, RS Vertex-based,,, PCL,, ";
	test_result << "E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based" << std::endl;

	// Disjoint_comp_num variation, point_num fix, joint_comp_num fix, and point_distance
	point_num = 262144;
	point_degree = 32;
	point_distance = 4;

	for (disjoint_comp_num = 16; disjoint_comp_num <= 8192; disjoint_comp_num *= 2) {
		joint_comp_num = (point_num / disjoint_comp_num) / point_degree;
		std::string res = pointCloudVariationTest(point_num, disjoint_comp_num, joint_comp_num, point_distance);
		test_result << disjoint_comp_num << "," << res << std::endl;
	}

	std::cout << "########################### Disjoint Comp Num Variation Test ###########################" << std::endl;

	test_result << "***** Disjoint Comp Num Variation Test *****" << std::endl;
	test_result << "Cluster num varies point num = 32768 point degree = 32 point distance = 4" << std::endl;

	test_result << "Cluster num, E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based, PCL, ";
	test_result << "E Edge-based,,, E Matrix-based,,,, E Vertex-based,,, RS Edge-based,,, RS Matrix-based,,,, RS Vertex-based,,, PCL,, ";
	test_result << "E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based" << std::endl;

	// Disjoint_comp_num variation, point_num fix, joint_comp_num fix, and point_distance
	point_num = 32768;
	point_degree = 32;
	point_distance = 4;

	for (disjoint_comp_num = 16; disjoint_comp_num <= 1024; disjoint_comp_num *= 2) {
		joint_comp_num = (point_num / disjoint_comp_num) / point_degree;
		std::string res = pointCloudVariationTest(point_num, disjoint_comp_num, joint_comp_num, point_distance);
		test_result << disjoint_comp_num << "," << res << std::endl;
	}

	test_result << std::endl << std::endl << std::endl;

	std::cout << "########################### Disjoint Comp Num Variation Test ###########################" << std::endl;

	test_result << "***** Disjoint Comp Num Variation Test *****" << std::endl;
	test_result << "Cluster num varies point num = 4096 point degree = 32 point distance = 4" << std::endl;

	test_result << "Cluster num, E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based, PCL, ";
	test_result << "E Edge-based,,, E Matrix-based,,,, E Vertex-based,,, RS Edge-based,,, RS Matrix-based,,,, RS Vertex-based,,, PCL,, ";
	test_result << "E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based" << std::endl;

	// Disjoint_comp_num variation, point_num fix, joint_comp_num fix, and point_distance
	point_num = 4096;
	point_degree = 32;
	point_distance = 4;

	for (disjoint_comp_num = 16; disjoint_comp_num <= 128; disjoint_comp_num *= 2) {
		joint_comp_num = (point_num / disjoint_comp_num) / point_degree;
		std::string res = pointCloudVariationTest(point_num, disjoint_comp_num, joint_comp_num, point_distance);
		test_result << disjoint_comp_num << "," << res << std::endl;
	}

	test_result << std::endl << std::endl << std::endl;



	std::cout << "########################### Joint Comp Num Variation Test ###########################" << std::endl;
	test_result << "***** Point degree Variation Test *****" << std::endl;
	test_result << "Neighbor num varies point num = 262144 disjoint num = 128 point distance = 4" << std::endl;

	test_result << "Neighbor num, E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based, PCL, ";
	test_result << "E Edge-based,,, E Matrix-based,,,, E Vertex-based,,, RS Edge-based,,, RS Matrix-based,,,, RS Vertex-based,,, PCL,, ";
	test_result << "E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based" << std::endl;

	// Joint_comp_num variation, point_num, disjoint_comp_num, and point_distance are fixed
	point_num = 262144;
	disjoint_comp_num = 128;
	point_distance = 4;

	for (point_degree = 1; point_degree <= 2048; point_degree *= 2) {
		joint_comp_num = (point_num / disjoint_comp_num) / point_degree;
		std::string res = pointCloudVariationTest(point_num, disjoint_comp_num, joint_comp_num, point_distance);
		test_result << point_degree << "," << res << std::endl;
	}
	test_result << std::endl << std::endl << std::endl;



	std::cout << "########################### Joint Comp Num Variation Test ###########################" << std::endl;
	test_result << "***** Joint Comp Num Variation Test *****" << std::endl;
	test_result << "Neighbor num varies point num = 32768 disjoint num = 128 point distance = 4" << std::endl;

	test_result << "Neighbor num, E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based, PCL, ";
	test_result << "E Edge-based,,, E Matrix-based,,,, E Vertex-based,,, RS Edge-based,,, RS Matrix-based,,,, RS Vertex-based,,, PCL,, ";
	test_result << "E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based" << std::endl;

	// Joint_comp_num variation, point_num, disjoint_comp_num, and point_distance are fixed
	point_num = 32768;
	disjoint_comp_num = 128;
	point_distance = 4;

	for (point_degree = 1; point_degree <= 256; point_degree *= 2) {
		joint_comp_num = (point_num / disjoint_comp_num) / point_degree;
		std::string res = pointCloudVariationTest(point_num, disjoint_comp_num, joint_comp_num, point_distance);
		test_result << point_degree << "," << res << std::endl;
	}
	test_result << std::endl << std::endl << std::endl;


	std::cout << "########################### Joint Comp Num Variation Test ###########################" << std::endl;
	test_result << "***** Joint Comp Num Variation Test *****" << std::endl;
	test_result << "Neighbor num varies point num = 4096 disjoint num = 128 point distance = 4" << std::endl;

	test_result << "Neighbor num, E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based, PCL, ";
	test_result << "E Edge-based,,, E Matrix-based,,,, E Vertex-based,,, RS Edge-based,,, RS Matrix-based,,,, RS Vertex-based,,, PCL,, ";
	test_result << "E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based" << std::endl;

	// Joint_comp_num variation, point_num, disjoint_comp_num, and point_distance are fixed
	point_num = 4096;
	disjoint_comp_num = 128;
	point_distance = 4;

	for (point_degree = 1; point_degree <= 32; point_degree *= 2) {
		joint_comp_num = (point_num / disjoint_comp_num) / point_degree;
		std::string res = pointCloudVariationTest(point_num, disjoint_comp_num, joint_comp_num, point_distance);
		test_result << point_degree << "," << res << std::endl;
	}
	test_result << std::endl << std::endl << std::endl;


	std::cout << "########################### Point Distance Variation Test ###########################" << std::endl;
	test_result << "***** Point Distance Variation Test *****" << std::endl;
	test_result << "Point distance varies point num = 65536 disjoint num = 256 joint_num = 32" << std::endl;

	test_result << "Point distance, E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based, PCL, ";
	test_result << "E Edge-based,,, E Matrix-based,,,, E Vertex-based,,, RS Edge-based,,, RS Matrix-based,,,, RS Vertex-based,,, PCL,, ";
	test_result << "E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based" << std::endl;

	// Point distance variation, others are fixed

	point_num = 65536;
	disjoint_comp_num = 256;
	point_degree = 32;

//	for (point_distance = 1; point_distance <= 2048; point_distance += 7) {
//		std::string res = pointCloudVariationTest(point_num, disjoint_comp_num, joint_comp_num, point_distance);
//		test_result << point_distance << "," << res << std::endl;
//	}


	joint_comp_num = (point_num / disjoint_comp_num) / point_degree;
	SampleCloud base_cloud = pointCloudGeneration(point_num, disjoint_comp_num, joint_comp_num);
	for (point_distance = 1; point_distance <= 256; point_distance *= 2) {
		std::string res = pointDistanceTest(base_cloud, point_distance);
		test_result << point_distance << "," << res << std::endl;
	}

	test_result << std::endl << std::endl << std::endl;

	// Point distance variation, others are fixed
	std::cout << "########################### Point Distance Variation Test ###########################" << std::endl;
	test_result << "***** Point Distance Variation Test *****" << std::endl;
	test_result << "Point distance varies point num = 65536 disjoint num = 1024 joint_num = 32" << std::endl;

	test_result << "Point distance, E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based, PCL, ";
	test_result << "E Edge-based,,, E Matrix-based,,,, E Vertex-based,,, RS Edge-based,,, RS Matrix-based,,,, RS Vertex-based,,, PCL,, ";
	test_result << "E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based" << std::endl;


	point_num = 65536;
	disjoint_comp_num = 1024;
	joint_comp_num = 32;

//	for (point_distance = 1; point_distance <= 2048; point_distance += 7) {
//		std::string res = pointCloudVariationTest(point_num, disjoint_comp_num, joint_comp_num, point_distance);
//		test_result << point_distance << "," << res << std::endl;
//	}

	base_cloud = pointCloudGeneration(point_num, disjoint_comp_num, joint_comp_num);
	for (point_distance = 1; point_distance <= 1024; point_distance *= 2) {
		std::string res = pointDistanceTest(base_cloud, point_distance);
		test_result << point_distance << "," << res << std::endl;
	}

	test_result << std::endl << std::endl << std::endl;

	// Point distance variation, others are fixed
	std::cout << "########################### Point Distance Variation Test ###########################" << std::endl;
	test_result << "***** Point Distance Variation Test *****" << std::endl;
	test_result << "Point distance varies point num = 65536 disjoint num = 2048 joint_num = 32" << std::endl;

	test_result << "Point distance, E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based, PCL, ";
	test_result << "E Edge-based,,, E Matrix-based,,,, E Vertex-based,,, RS Edge-based,,, RS Matrix-based,,,, RS Vertex-based,,, PCL,, ";
	test_result << "E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based" << std::endl;

	point_num = 65536;
	disjoint_comp_num = 2048;
	joint_comp_num = 32;

//	for (point_distance = 1; point_distance <= 2048; point_distance += 7) {
//		std::string res = pointCloudVariationTest(point_num, disjoint_comp_num, joint_comp_num, point_distance);
//		test_result << point_distance << "," << res << std::endl;
//	}

	base_cloud = pointCloudGeneration(point_num, disjoint_comp_num, joint_comp_num);
	for (point_distance = 1; point_distance <= 512; point_distance *= 2) {
		std::string res = pointDistanceTest(base_cloud, point_distance);
		test_result << point_distance << "," << res << std::endl;
	}

	test_result << std::endl << std::endl << std::endl;

	std::cout << "END OF POINT CLOUD VARIATION TEST" << std::endl;

	std::cout << "***** LINE TEST *****" << std::endl;

	// Line test
	point_num = 1048576;

	test_result << "***** LINE TEST *****" << std::endl;

	test_result << "Point distance, E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based, PCL, ";
	test_result << "E Edge-based,,, E Matrix-based,,,, E Vertex-based,,, RS Edge-based,,, RS Matrix-based,,,, RS Vertex-based,,, PCL,, ";
	test_result << "E Edge-based, E Matrix-based, E Vertex-based, RS Edge-based, RS Matrix-based, RS Vertex-based" << std::endl;


	test_result << lineTest(point_num);

	std::cout << "END OF POINT CLOUD VARIATION TEST" << std::endl;
}

std::string GPUECTest::pointCloudVariationTest(int point_num, int disjoint_comp_num, int joint_comp_num, int point_distance)
{
	std::stringstream output;

	srand(time(NULL));

	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), point_num, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;
	int points_per_disjoint = point_num / disjoint_comp_num;
	int points_per_joint = points_per_disjoint / joint_comp_num;

	std::vector<bool> status(point_num, false);
	pcl::PointXYZ origin(0, 0, 0);
	float sample_dist;

#ifdef DEBUG_
	std::ofstream out_test("/home/anh/sample_cloud_pid.txt");
#endif

	for (int djcomp_id = 0, base_id = 0; djcomp_id < disjoint_comp_num; djcomp_id++, base_id++) {
		while (status[base_id]) {
			base_id++;
		}
		int offset = point_distance;

//		while (offset * points_per_disjoint + base_id > point_num) {
//			offset--;
//		}

		for (int i = 0; i < points_per_disjoint; i++) {
			int pid = base_id + i * offset;
			int joint_id = i / points_per_joint;

#ifdef DEBUG_
			out_test << "Cluster " << djcomp_id << " pid = " << pid << " ";
#endif

			// Moved to the new disjoint, move the origin
			if (i % points_per_joint == 0) {
				if (joint_id % 3 == 1) {
					origin.x += d_th * JOINT_DIST_FACTOR_;
				} else if (joint_id % 3 == 2) {
					origin.y += d_th * JOINT_DIST_FACTOR_;
				} else {
					origin.z += d_th * JOINT_DIST_FACTOR_;
				}
			}

#ifdef DEBUG_
			out_test << "origin =  " << origin << " ";
#endif

			sample_dist = rand() % SAMPLE_RAND_;

			sample_point = origin;

			if (joint_id % 3 == 0) {
				sample_point.x = origin.x + sample_dist / SAMPLE_DIST_;
			} else if (joint_id % 3 == 1) {
				sample_point.y = origin.y + sample_dist / SAMPLE_DIST_;
			} else {
				sample_point.z = origin.z + sample_dist / SAMPLE_DIST_;
			}

#ifdef DEBUG_
			out_test << "sample point = " << sample_point << std::endl;
#endif

			sample_cloud->points[pid] = sample_point;
			status[pid] = true;
		}

		origin.x += d_th * 10;
		origin.y += d_th * 10;
		origin.z += d_th * 10;
	}

	output << test(sample_cloud, 1024, d_th);

	return output.str();
}

std::string GPUECTest::lineTest(int point_num)
{
	std::ostringstream output;

	srand(time(NULL));

	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), point_num, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;
	std::vector<bool> status(point_num, false);

	for (int i = 0; i < point_num; i++) {
		if (i % 3 == 1) {
			sample_point.x += d_th * JOINT_DIST_FACTOR_;
		} else if (i % 3 == 2) {
			sample_point.y += d_th * JOINT_DIST_FACTOR_;
		} else {
			sample_point.z += d_th * JOINT_DIST_FACTOR_;
		}

		int pid = 0;

		while (status[pid]) {
			pid = rand() % point_num;
		}

		sample_cloud->points[pid] = sample_point;
		status[pid] = true;
	}

	output << test(sample_cloud, 1024, d_th);

	return output.str();
}

GPUECTest::SampleCloud GPUECTest::pointCloudGeneration(int point_num, int disjoint_comp_num, int joint_comp_num)
{
	srand(time(NULL));

	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), point_num, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;
	int points_per_disjoint = point_num / disjoint_comp_num;
	int points_per_joint = points_per_disjoint / joint_comp_num;

	pcl::PointXYZ origin(0, 0, 0);
	float sample_dist;


	int pid = 0;

	std::cout << "Disjoint num = " << disjoint_comp_num << std::endl;
	std::cout << "Joint num = " << joint_comp_num << std::endl;
	std::cout << "Points per joint = " << points_per_joint << std::endl;

	for (int i = 0; i < disjoint_comp_num; i++) {
		for (int j = 0; j < joint_comp_num; j++) {
			for (int k = 0; k < points_per_joint; k++) {
				sample_point = origin;

				sample_dist = rand() % SAMPLE_RAND_;

				if (j % 3 == 1) {
					sample_point.x = origin.x + sample_dist / SAMPLE_DIST_;
				} else if (j % 3 == 2) {
					sample_point.y = origin.y + sample_dist / SAMPLE_DIST_;
				} else {
					sample_point.z = origin.z + sample_dist / SAMPLE_DIST_;
				}

				sample_cloud->points[pid++] = sample_point;
			}

			// Generate the origin of the next disjoint component
			if (j % 3 == 0) {
				origin.x += d_th * JOINT_DIST_FACTOR_;
			} else if (j % 3 == 1) {
				origin.y += d_th * JOINT_DIST_FACTOR_;
			} else {
				origin.z += d_th * JOINT_DIST_FACTOR_;
			}
		}

		origin.x += d_th * 10;
		origin.y += d_th * 10;
		origin.z += d_th * 10;
	}

	SampleCloud output;

	output.cloud_ = sample_cloud;
	output.disjoint_num_ = disjoint_comp_num;
	output.joint_num_ = joint_comp_num;
	output.point_distance_ = 1;

	std::cout << "Base cloud test" << std::endl;
	test(sample_cloud, 1024, d_th);

	return output;
}

// Assume that base_cloud.point_distance is always 1
std::string GPUECTest::pointDistanceTest(SampleCloud base_cloud, int point_distance)
{
	std::stringstream output;

	srand(time(NULL));

	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	int point_num = base_cloud.cloud_->points.size();

	sample_cloud->points.insert(sample_cloud->points.begin(), point_num, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;
	int disjoint_comp_num = base_cloud.disjoint_num_;
	int points_per_disjoint = point_num / disjoint_comp_num;

	std::vector<bool> status(point_num, false);
	pcl::PointXYZ origin(0, 0, 0);
	int source_id = 0;

	for (int djcomp_id = 0, base_id = 0; djcomp_id < disjoint_comp_num; djcomp_id++, base_id++) {
		while (status[base_id]) {
			base_id++;
		}
		int offset = point_distance;

//		while (offset * points_per_disjoint + base_id > point_num) {
//			offset--;
//		}

		for (int i = 0; i < points_per_disjoint; i++) {
			int pid = base_id + i * offset;

			sample_cloud->points[pid] = base_cloud.cloud_->points[source_id++];
			status[pid] = true;
		}
	}

	output << test(sample_cloud, 1024, d_th);

	return output.str();
}

void GPUECTest::matrixTest()
{
	srand(time(NULL));

	int point_num = 1024 * 4;

	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), point_num, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;

	std::vector<bool> status(point_num, false);
	pcl::PointXYZ origin(0, 0, 0);

	sample_cloud->points[513] = sample_point;
	status[513] = true;

	sample_cloud->points[3500] = sample_point;
	status[3500] = true;

	sample_point.x += d_th * 10;
	sample_point.y += d_th * 10;
	sample_point.z += d_th * 10;

	sample_cloud->points[1125] = sample_point;
	status[1125] = true;

	sample_cloud->points[2080] = sample_point;
	status[2080] = true;

	for (int i = 0; i < point_num; i++) {
		if (!status[i]) {
			sample_point.x += d_th * 10;
			sample_point.y += d_th * 10;
			sample_point.z += d_th * 10;

			sample_cloud->points[i] = sample_point;

			status[i] = true;
		}
	}

	GpuEuclideanCluster2 test_sample;

	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);
	test_sample.setInputPoints(sample_cloud);

	struct timeval start, end;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	gettimeofday(&start, NULL);
	test_sample.extractClusters5();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	gettimeofday(&start, NULL);
	test_sample.extractClusters4();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

}


/* The tests above have clusters with the same number of points.
 * In this test, the number of points in clusters are not the same.
 * This is to test the ability of handling skewed workload of gpu algorithms.
 * */
std::string GPUECTest::variousSizeClusterTest(int point_num, int disjoint_comp_num, int point_num_per_joint, int common_diff)
{
	std::ostringstream output;

	srand(time(NULL));

	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), point_num, pcl::PointXYZ(0, 0, 0));

	std::vector<bool> status(point_num, false);
	int point_per_disjoint = point_num / disjoint_comp_num - common_diff * (disjoint_comp_num - 1) / 2;
	pcl::PointXYZ sample_point(0, 0, 0);
	int point_count = 0;
	pcl::PointXYZ origin(0, 0, 0);
	float sample_dist;
	float d_th = 1.0;
	int point_distance = 4;
	int base_id = 0;
	int offset = 0;

	for (int cluster_id = 0; cluster_id < disjoint_comp_num; cluster_id++, point_per_disjoint += common_diff) {
		int joint_comp_num = (point_per_disjoint - 1) / point_num_per_joint + 1;
		int disjoint_point_count = 0;

		for (int joint_id = 0; joint_id < joint_comp_num; joint_id++) {
			if (joint_id % 3 == 1) {
				origin.x += d_th * JOINT_DIST_FACTOR_;
			} else if (joint_id % 3 == 2) {
				origin.y += d_th * JOINT_DIST_FACTOR_;
			} else {
				origin.z += d_th * JOINT_DIST_FACTOR_;
			}

			for (int i = 0; i < point_num_per_joint && disjoint_point_count < point_per_disjoint && point_count < point_num; i++, disjoint_point_count++, point_count++) {
				int pid = base_id + offset * point_distance;

				if (pid >= point_num) {
					base_id++;
					offset = 0;

					pid = base_id + offset * point_distance;
				} else {
					offset++;
				}

				sample_dist = rand() % SAMPLE_RAND_;

				sample_point = origin;

				if (joint_id % 3 == 0) {
					sample_point.x += sample_dist / SAMPLE_DIST_;
				} else if (joint_id % 3 == 1) {
					sample_point.y += sample_dist / SAMPLE_DIST_;
				} else {
					sample_point.z += sample_dist / SAMPLE_DIST_;
				}

				sample_cloud->points[pid] = sample_point;
				status[pid] = true;
			}
		}

		origin.x += d_th * 10;
		origin.y += d_th * 10;
		origin.z += d_th * 10;
	}

	for (; point_count < point_num; point_count++) {
		int pid = 0;

		while (status[pid]) {
			pid = rand() % point_num;
		}

		sample_dist = rand() % SAMPLE_RAND_;

		sample_point.x = origin.x + sample_dist / SAMPLE_DIST_;

		sample_cloud->points[pid] = sample_point;
		status[pid] = true;
	}

	output << test(sample_cloud, 1024, d_th);

	return output.str();
}

// Input the sample cloud and output result to a string
std::string GPUECTest::test(pcl::PointCloud<pcl::PointXYZ>::Ptr input, int block_size, float threshold)
{
	std::ostringstream output;

	struct timeval start, end;

	GpuEuclideanCluster2 test_sample;

	long long gpu_initial;

	gettimeofday(&start, NULL);
	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(threshold);
	test_sample.setInputPoints(input);
	gettimeofday(&end, NULL);

	gpu_initial = timeDiff(start, end);


	long long e_total_time, e_graph_time, e_clustering_time;
	int e_itr_num;

	long long e_total_time2, e_graph_time2, e_clustering_time2;
	int e_itr_num2;

	long long m_total_time, m_initial, m_build_matrix, m_clustering_time;
	int m_itr_num;

	long long m_total_time2, m_initial2, m_build_matrix2, m_clustering_time2;
	int m_itr_num2;

	long long v_total_time, v_graph_time, v_clustering_time;
	int v_itr_num;

	long long v_total_time2, v_graph_time2, v_clustering_time2;
	int v_itr_num2;

	long long c_total_time, c_clustering_time, c_tree_build;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3(e_total_time, e_graph_time, e_clustering_time, e_itr_num);
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	e_total_time += gpu_initial;
	std::cout << "E Edge-based: total exec time = " << gpu_initial + timeDiff(start, end) << std::endl << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters5(e_total_time2, e_graph_time2, e_clustering_time2, e_itr_num2);
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	e_total_time2 += gpu_initial;
	std::cout << "RS Edge-based: total exec time = " << gpu_initial + timeDiff(start, end) << std::endl << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters(m_total_time, m_initial, m_build_matrix, m_clustering_time, m_itr_num);
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	m_total_time += gpu_initial;
	std::cout << "E Matrix-based: total exec time = " << gpu_initial + timeDiff(start, end) << std::endl << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters4(m_total_time2, m_initial2, m_build_matrix2, m_clustering_time2, m_itr_num2);
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	m_total_time2 += gpu_initial;
	std::cout << "RS Matrix-based: total exec time = " << gpu_initial + timeDiff(start, end) << std::endl << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2(v_total_time, v_graph_time, v_clustering_time, v_itr_num);
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	v_total_time += gpu_initial;
	std::cout << "E Vertex-based: total exec time = " << gpu_initial + timeDiff(start, end) << std::endl << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters6(v_total_time2, v_graph_time2, v_clustering_time2, v_itr_num2);
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	v_total_time2 += gpu_initial;
	std::cout << "RS Vertex-based: total exec time = " <<gpu_initial + timeDiff(start, end) << std::endl << std::endl;

	gettimeofday(&start, NULL);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

	tree->setInputCloud (input);
	gettimeofday(&end, NULL);

	c_tree_build = timeDiff(start, end);

	std::vector<pcl::PointIndices> cluster_indices;

	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance (threshold);
	ec.setSearchMethod(tree);
	ec.setInputCloud(input);
	ec.extract (cluster_indices);

	gettimeofday(&end, NULL);

	std::cout << "PCL Cluster num = " << cluster_indices.size() << std::endl;

	c_total_time = timeDiff(start, end);
	c_clustering_time = c_total_time - c_tree_build;

	std::cout << "PCL: total exec time = " << c_total_time << std::endl << std::endl;

	// Total execution time
	output << e_total_time << "," << m_total_time << "," << v_total_time << "," << e_total_time2 << "," << m_total_time2 << "," << v_total_time2 << "," << c_total_time << ",";

	// Breakdown
	output << gpu_initial << "," << e_graph_time << "," << e_clustering_time << ",";
	output << gpu_initial << "," << m_initial << "," << m_build_matrix << "," << m_clustering_time << ",";
	output << gpu_initial << "," << v_graph_time << ","  << v_clustering_time << ",";

	output << gpu_initial << "," << e_graph_time2 << "," << e_clustering_time2 << ",";
	output << gpu_initial << "," << m_initial2 << "," << m_build_matrix2 << "," << m_clustering_time2 << ",";
	output << gpu_initial << "," << v_graph_time2 << ","  << v_clustering_time2 << ",";


	// Cpu-based
	output << c_tree_build << "," << c_clustering_time << ",";

	// Iteration number
	output << e_itr_num << "," << m_itr_num << "," << v_itr_num << "," << e_itr_num2 << "," << m_itr_num2 << "," << v_itr_num2;

	// Speedup rate

	return output.str();
}
