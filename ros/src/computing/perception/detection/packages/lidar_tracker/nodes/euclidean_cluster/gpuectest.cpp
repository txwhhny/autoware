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

void GPUECTest::sparseGraphTest()
{
	std::cout << "********** SPARSE GRAPH TEST *************" << std::endl;

	sparseGraphTest100();

	sparseGraphTest875();

	sparseGraphTest75();

	sparseGraphTest625();

	sparseGraphTest50();

	sparseGraphTest375();

	sparseGraphTest25();

	sparseGraphTest125();

	sparseGraphTest0();

	std::cout << "*********** END OF SPARSE GRAPH TEST **********" << std::endl << std::endl;
}

void GPUECTest::sparseGraphTest100()
{
	GpuEuclideanCluster2 test_sample;

	srand(time(NULL));

	// Fix at SAMPLE_SIZE_ points, density varies from 100% to 0%
	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), SAMPLE_SIZE_, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;
	float sample_dist;
	std::vector<bool> status(SAMPLE_SIZE_, false);
	struct timeval start, end;


	// Set clustering parameters
	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);

	// Density 100%
	for (int i = 0; i < sample_cloud->points.size(); i++) {
		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.x = sample_dist / SAMPLE_DIST_;

		sample_point.y = 0;

		sample_point.z = 0;

		sample_cloud->points[i] = sample_point;
	}

	test_sample.setInputPoints(sample_cloud);

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 100% - Matrix-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 100% - Vertex-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 100% - Edge-based: " << timeDiff(start, end) << std::endl;
}

void GPUECTest::sparseGraphTest875()
{
	GpuEuclideanCluster2 test_sample;

	srand(time(NULL));

	// Fix at SAMPLE_SIZE_ points, density varies from 100% to 0%
	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), SAMPLE_SIZE_, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;
	float sample_dist;
	std::vector<bool> status(SAMPLE_SIZE_, false);
	int part_size;
	struct timeval start, end;


	// Set clustering parameters
	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);

	// Density 87.5%

	part_size = (SAMPLE_SIZE_F_ + sqrt(SAMPLE_SIZE_F_ * SAMPLE_SIZE_F_ - 2 * 0.125 * SAMPLE_SIZE_F_ * (SAMPLE_SIZE_F_ - 1))) / 2;

	for (int i = 0; i < part_size; i++) {
		int pid = 0;

		while (status[pid]) {
			pid = rand() % SAMPLE_SIZE_;
		}

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.x = sample_dist / SAMPLE_DIST_;

		sample_point.y = 0;

		sample_point.z = 0;

		sample_cloud->points[pid] = sample_point;
		status[pid] = true;
	}

	for (int i = 0; i < SAMPLE_SIZE_ - part_size; i++) {
		int pid = 0;

		while (status[pid]) {
			pid = rand() % SAMPLE_SIZE_;
		}

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.x = d_th * 10 + sample_dist / SAMPLE_DIST_;

		sample_point.y = d_th * 10;

		sample_point.z = d_th * 10;

		sample_cloud->points[pid] = sample_point;
		status[pid] = true;
	}

	test_sample.setInputPoints(sample_cloud);

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 87.5% - Matrix-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 87.5% - Vertex-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 87.5% - Edge-based: " << timeDiff(start, end) << std::endl;
}

void GPUECTest::sparseGraphTest75()
{
	GpuEuclideanCluster2 test_sample;

	srand(time(NULL));

	// Fix at SAMPLE_SIZE_ points, density varies from 100% to 0%
	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), SAMPLE_SIZE_, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;
	float sample_dist;
	std::vector<bool> status(SAMPLE_SIZE_, false);
	int part_size;
	struct timeval start, end;


	// Set clustering parameters
	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);

	// Density 75%
	part_size = (SAMPLE_SIZE_F_ + sqrt(SAMPLE_SIZE_F_ * SAMPLE_SIZE_F_ - 2 * 0.25 * SAMPLE_SIZE_F_ * (SAMPLE_SIZE_F_ - 1))) / 2;

	for (int i = 0; i < part_size; i++) {
		int pid = 0;

		while (status[pid]) {
			pid = rand() % SAMPLE_SIZE_;
		}

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.x = sample_dist / SAMPLE_DIST_;

		sample_point.y = 0;

		sample_point.z = 0;

		sample_cloud->points[pid] = sample_point;
		status[pid] = true;
	}

	for (int i = 0; i < SAMPLE_SIZE_ - part_size; i++) {
		int pid = 0;

		while (status[pid]) {
			pid = rand() % SAMPLE_SIZE_;
		}

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.x = d_th * 10 + sample_dist / SAMPLE_DIST_;

		sample_point.y = d_th * 10;

		sample_point.z = d_th * 10;

		sample_cloud->points[pid] = sample_point;
		status[pid] = true;
	}

	test_sample.setInputPoints(sample_cloud);

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 75% - Matrix-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 75% - Vertex-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 75% - Edge-based: " << timeDiff(start, end) << std::endl;

}

void GPUECTest::sparseGraphTest625()
{
	GpuEuclideanCluster2 test_sample;

	srand(time(NULL));

	// Fix at SAMPLE_SIZE_ points, density varies from 100% to 0%
	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), SAMPLE_SIZE_, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;
	float sample_dist;
	std::vector<bool> status(SAMPLE_SIZE_, false);
	int part_size;
	struct timeval start, end;


	// Set clustering parameters
	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);

	// Density 62.5%
	part_size = (SAMPLE_SIZE_F_ + sqrt(SAMPLE_SIZE_F_ * SAMPLE_SIZE_F_ - 2 * 0.375 * SAMPLE_SIZE_F_ * (SAMPLE_SIZE_F_ - 1))) / 2;

	for (int i = 0; i < part_size; i++) {
		int pid = 0;

		while (status[pid]) {
			pid = rand() % SAMPLE_SIZE_;
		}

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.x = sample_dist / SAMPLE_DIST_;

		sample_point.y = 0;

		sample_point.z = 0;

		sample_cloud->points[pid] = sample_point;
		status[pid] = true;
	}

	for (int i = 0; i < SAMPLE_SIZE_ - part_size; i++) {
		int pid = 0;

		while (status[pid]) {
			pid = rand() % SAMPLE_SIZE_;
		}

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.x = d_th * 10 + sample_dist / SAMPLE_DIST_;

		sample_point.y = d_th * 10;

		sample_point.z = d_th * 10;

		sample_cloud->points[pid] = sample_point;
		status[pid] = true;
	}

	test_sample.setInputPoints(sample_cloud);

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 62.5% - Matrix-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 62.5% - Vertex-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 62.5% - Edge-based: " << timeDiff(start, end) << std::endl;
}

void GPUECTest::sparseGraphTest50()
{
	// Density 50%
	GpuEuclideanCluster2 test_sample;

	srand(time(NULL));

	// Fix at SAMPLE_SIZE_ points, density varies from 100% to 0%
	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), SAMPLE_SIZE_, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;
	float sample_dist;
	std::vector<bool> status(SAMPLE_SIZE_, false);
	int part_size;
	struct timeval start, end;


	// Set clustering parameters
	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);

	part_size = (SAMPLE_SIZE_F_ + sqrt(SAMPLE_SIZE_F_ * SAMPLE_SIZE_F_ - 2 * 0.5 * SAMPLE_SIZE_F_ * (SAMPLE_SIZE_F_ - 1))) / 2;

	for (int i = 0; i < part_size; i++) {
		int pid = 0;

		while (status[pid]) {
			pid = rand() % SAMPLE_SIZE_;
		}

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.x = sample_dist / SAMPLE_DIST_;

		sample_point.y = 0;

		sample_point.z = 0;

		sample_cloud->points[pid] = sample_point;
		status[pid] = true;
	}

	for (int i = 0; i < SAMPLE_SIZE_ - part_size; i++) {
		int pid = 0;

		while (status[pid]) {
			pid = rand() % SAMPLE_SIZE_;
		}

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.x = d_th * 10 + sample_dist / SAMPLE_DIST_;

		sample_point.y = d_th * 10;

		sample_point.z = d_th * 10;

		sample_cloud->points[pid] = sample_point;
		status[pid] = true;
	}

	test_sample.setInputPoints(sample_cloud);

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 50% - Matrix-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 50% - Vertex-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 50% - Edge-based: " << timeDiff(start, end) << std::endl << std::endl;

}

void GPUECTest::sparseGraphTest375()
{
	GpuEuclideanCluster2 test_sample;

	srand(time(NULL));

	// Fix at SAMPLE_SIZE_ points, density varies from 100% to 0%
	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), SAMPLE_SIZE_, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;
	float sample_dist;
	std::vector<bool> status(SAMPLE_SIZE_, false);
	int part_size;
	struct timeval start, end;


	// Set clustering parameters
	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);

	// Density 37.5%

	part_size = SAMPLE_SIZE_ / 2;

	for (int i = 0; i < part_size; i++) {
		int pid = 0;

		while (status[pid]) {
			pid = rand() % SAMPLE_SIZE_;
		}

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.x = sample_dist / SAMPLE_DIST_;

		sample_point.y = 0;

		sample_point.z = 0;

		sample_cloud->points[pid] = sample_point;
		status[pid] = true;
	}

	part_size = SAMPLE_SIZE_ / 4;

	for (int i = 1; i <= 2; i++) {
		for (int j = 0; j < part_size; j++) {
			int pid = 0;

			while (status[pid]) {
				pid = rand() % SAMPLE_SIZE_;
			}

			sample_dist = rand() % SAMPLE_RAND_;
			sample_point.x = d_th * i * 10 + sample_dist / SAMPLE_DIST_;

			sample_point.y = d_th * i * 10;

			sample_point.z = d_th * i * 10;

			sample_cloud->points[pid] = sample_point;
			status[pid] = true;
		}
	}

	test_sample.setInputPoints(sample_cloud);

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 37.5% - Matrix-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 37.5% - Vertex-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 37.5% - Edge-based: " << timeDiff(start, end) << std::endl;

}

void GPUECTest::sparseGraphTest25()
{
	GpuEuclideanCluster2 test_sample;

	srand(time(NULL));

	// Fix at SAMPLE_SIZE_ points, density varies from 100% to 0%
	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), SAMPLE_SIZE_, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;
	float sample_dist;
	std::vector<bool> status(SAMPLE_SIZE_, false);
	int part_size;
	struct timeval start, end;


	// Set clustering parameters
	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);

	// Density 25%
	part_size = SAMPLE_SIZE_ / 4;

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < part_size; j++) {
			int pid = 0;

			while (status[pid]) {
				pid = rand() % SAMPLE_SIZE_;
			}

			sample_dist = rand() % SAMPLE_RAND_;
			sample_point.x = d_th * i * 10 + sample_dist / SAMPLE_DIST_;

			sample_point.y = d_th * i * 10;

			sample_point.z = d_th * i * 10;

			sample_cloud->points[pid] = sample_point;
			status[pid] = true;
		}
	}

	test_sample.setInputPoints(sample_cloud);

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 25% - Matrix-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 25% - Vertex-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 25% - Edge-based: " << timeDiff(start, end) << std::endl;

}

void GPUECTest::sparseGraphTest125()
{
	GpuEuclideanCluster2 test_sample;

	srand(time(NULL));

	// Fix at SAMPLE_SIZE_ points, density varies from 100% to 0%
	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), SAMPLE_SIZE_, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;
	float sample_dist;
	std::vector<bool> status(SAMPLE_SIZE_, false);
	int part_size;
	struct timeval start, end;


	// Set clustering parameters
	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);

	// Density 12.5%
	part_size = SAMPLE_SIZE_ / 8;

	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < part_size; j++) {
			int pid = 0;

			while (status[pid]) {
				pid = rand() % SAMPLE_SIZE_;
			}

			sample_dist = rand() % SAMPLE_RAND_;
			sample_point.x = d_th * i * 10 + sample_dist / SAMPLE_DIST_;

			sample_point.y = d_th * i * 10;

			sample_point.z = d_th * i * 10;

			sample_cloud->points[pid] = sample_point;
			status[pid] = true;
		}
	}

	test_sample.setInputPoints(sample_cloud);

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 12.5% - Matrix-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 12.5% - Vertex-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 12.5% - Edge-based: " << timeDiff(start, end) << std::endl;
}

void GPUECTest::sparseGraphTest0()
{
	GpuEuclideanCluster2 test_sample;

	srand(time(NULL));

	// Fix at SAMPLE_SIZE_ points, density varies from 100% to 0%
	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), SAMPLE_SIZE_, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;

	struct timeval start, end;


	// Set clustering parameters
	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);

	// Density 0%
	sample_point.x = sample_point.y = sample_point.z = 0;

	for (int i = 0; i < sample_cloud->points.size(); i++) {
		if (i % 3 == 0)
			sample_point.x += d_th + 1;
		else if (i % 3 == 1)
			sample_point.y += d_th + 1;
		else
			sample_point.z += d_th + 1;

		if (i == 1)
			std::cout << "Point[1]=" << sample_point.x << "," << sample_point.y << "," << sample_point.z << std::endl;
		else if (i == 2)
			std::cout << "Point[2]=" << sample_point.x << "," << sample_point.y << "," << sample_point.z << std::endl;

		sample_cloud->points[i] = sample_point;
	}


	test_sample.setInputPoints(sample_cloud);

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 0% - Matrix-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 0% - Vertex-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 0% - Edge-based: " << timeDiff(start, end) << std::endl;
}


void GPUECTest::clusterNumVariationTest()
{
	std::cout << "********* CLUSTER NUMBER VARIATION TEST ***********" << std::endl;
//	for (int cluster_num = 1; cluster_num <= SAMPLE_SIZE_; cluster_num *= 2) {
//		clusterNumVariationTest(cluster_num);
//	}
	clusterNumVariationTest(1);

	std::cout << "****** END OF CLUSTER NUMBER VARIATION TEST *******" << std::endl;

}

void GPUECTest::clusterNumVariationTest(int cluster_num)
{
	srand(time(NULL));

	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), SAMPLE_SIZE_, pcl::PointXYZ(0, 0, 0));

	int points_per_cluster = SAMPLE_SIZE_ / cluster_num;
	pcl::PointXYZ origin(0, 0, 0);
	float sample_dist;
	pcl::PointXYZ sample_point;
	std::vector<bool> status(SAMPLE_SIZE_, false);
	float d_th = 1.0;

	std::cout << "Points per cluster = " << points_per_cluster << std::endl;

	for (int i = 0; i < cluster_num; i++) {

		for (int j = 0; j < points_per_cluster; j++) {
			int pid = 0;

			while (status[pid]) {
				pid = rand() % SAMPLE_SIZE_;
			}


			sample_dist = rand() % SAMPLE_RAND_;
			sample_point.x = origin.x + sample_dist / SAMPLE_DIST_;

			sample_point.y = origin.y;

			sample_point.z = origin.z;

			sample_cloud->points[pid] = sample_point;
			status[pid] = true;

	}

		origin.x += d_th * 10;
		origin.y += d_th * 10;
		origin.z += d_th * 10;
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

	std::cout << "Cluster num " << cluster_num << " - Edge-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Cluster num " << cluster_num << " - Matrix-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Cluster num " << cluster_num << " - Vertex-based: " << timeDiff(start, end) << std::endl << std::endl;

}

void GPUECTest::worstCaseEdgeBased()
{
	// Chained graph
	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), SAMPLE_SIZE_, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);

	float d_th = 1.0;

	for (int i = 0; i < sample_cloud->points.size(); i++) {
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

	std::cout << "***** POINT CLOUD VARIATION TEST *****" << std::endl;

	int point_num, disjoint_comp_num, joint_comp_num, point_distance;

	std::cout << "***** Point Num Variation Test *****" << std::endl;
	// Point num variation, fix disjoint_comp_num, joint_comp_num, point_distance
	disjoint_comp_num = 128;
	joint_comp_num = 32;
	point_distance = 4;

	std::ofstream test_result("/home/anh/euclidean_cluster_test.csv");

	test_result << "****** Point Num Variation Test *****" << std::endl;
	test_result << "point_num varies disjoint num = 128 joint num = 32 point distance = 4" << std::endl;
	test_result << "Point num, Edge-based, Matrix-based, Vertex-based, CPU, Set input, Edge-based,, Matrix-based,,, Vertex-based,, CPU,, Edge-based, Matrix-based, Vertex-based" << std::endl;

	for (point_num = 128 * 32; point_num <= 262144; point_num *= 2) {
		std::string res = pointCloudVariationTest(point_num, disjoint_comp_num, joint_comp_num, point_distance);
		test_result << point_num << "," << res << std::endl;
	}

	test_result << std::endl << std::endl << std::endl;

	std::cout << "***** Disjoint Comp Num Variation Test *****" << std::endl;

	test_result << "***** Disjoint Comp Num Variation Test *****" << std::endl;
	test_result << "Disjoint num varies point num = 262144 joint num = 32 point distance = 4" << std::endl;
	test_result << "Disjoint num, Edge-based, Matrix-based, Vertex-based, CPU, Set input, Edge-based,, Matrix-based,,, Vertex-based,, CPU,, Edge-based, Matrix-based, Vertex-based" << std::endl;
	// Disjoint_comp_num variation, point_num fix, joint_comp_num fix, and point_distance
	point_num = 262144;
	joint_comp_num = 32;
	point_distance = 4;

	for (disjoint_comp_num = 16; disjoint_comp_num <= 8192; disjoint_comp_num *= 2) {
		std::string res = pointCloudVariationTest(point_num, disjoint_comp_num, joint_comp_num, point_distance);
		test_result << disjoint_comp_num << "," << res << std::endl;
	}

	test_result << std::endl << std::endl << std::endl;

	std::cout << "***** Joint Comp Num Variation Test *****" << std::endl;
	test_result << "***** Joint Comp Num Variation Test *****" << std::endl;
	test_result << "Joint num varies point num = 262144 disjoint num = 128 point distance = 4" << std::endl;
	test_result << "Joint num, Edge-based, Matrix-based, Vertex-based, CPU, Set input, Edge-based,, Matrix-based,,, Vertex-based,, CPU,, Edge-based, Matrix-based, Vertex-based" << std::endl;
	// Joint_comp_num variation, point_num, disjoint_comp_num, and point_distance are fixed
	point_num = 262144;
	disjoint_comp_num = 128;
	point_distance = 4;

	for (joint_comp_num = 1; joint_comp_num <= 2048; joint_comp_num *= 2) {
		std::string res = pointCloudVariationTest(point_num, disjoint_comp_num, joint_comp_num, point_distance);
		test_result << joint_comp_num << "," << res << std::endl;
	}
	test_result << std::endl << std::endl << std::endl;

	std::cout << "***** Point Distance Variation Test *****" << std::endl;
	test_result << "***** Point Distance Variation Test *****" << std::endl;
	test_result << "Point distance varies point num = 65536 disjoint num = 2048 joint_num = 32" << std::endl;
	test_result << "Point distance, Edge-based, Matrix-based, Vertex-based, CPU, Set input, Edge-based,, Matrix-based,,, Vertex-based,, CPU,, Edge-based, Matrix-based, Vertex-based" << std::endl;
	// Point distance variation, others are fixed
	point_num = 65536;
	disjoint_comp_num = 2048;
	joint_comp_num = 32;

//	for (point_distance = 1; point_distance <= 2048; point_distance += 7) {
//		std::string res = pointCloudVariationTest(point_num, disjoint_comp_num, joint_comp_num, point_distance);
//		test_result << point_distance << "," << res << std::endl;
//	}

	SampleCloud base_cloud = pointCloudGeneration(point_num, disjoint_comp_num, joint_comp_num);
	for (point_distance = 1; point_distance <= 2048; point_distance += 7) {
		std::string res = pointDistanceTest(base_cloud, point_distance);
		test_result << point_distance << "," << res << std::endl;
	}

	test_result << std::endl << std::endl << std::endl;

	std::cout << "END OF POINT CLOUD VARIATION TEST" << std::endl;

	std::cout << "***** LINE TEST *****" << std::endl;

	// Line test
	point_num = 1048576;

	lineTest(point_num);


}

void GPUECTest::pointCloudVariationTest(int point_num, int disjoint_comp_num, int joint_comp_num)
{
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


	for (int i = 0; i < disjoint_comp_num; i++) {
		for (int j = 0; j < joint_comp_num; j++) {
			for (int k = 0; k < points_per_joint; k++) {
				int pid = 0;

				while (status[pid]) {
					pid = rand() % point_num;
				}

				sample_point = origin;

				sample_dist = rand() % SAMPLE_RAND_;

				if (j % 3 == 1) {
					sample_point.x = origin.x + sample_dist / SAMPLE_DIST_;
				} else if (j % 3 == 2) {
					sample_point.y = origin.y + sample_dist / SAMPLE_DIST_;
				} else {
					sample_point.z = origin.z + sample_dist / SAMPLE_DIST_;
				}

				sample_cloud->points[pid] = sample_point;

				status[pid] = true;
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
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);


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

	for (int djcomp_id = 0, base_id = 0; djcomp_id < disjoint_comp_num; djcomp_id++, base_id++) {
		while (status[base_id]) {
			base_id++;
		}
		int offset = point_distance;

		while (offset * points_per_disjoint + base_id > point_num) {
			offset--;
		}

		for (int i = 0; i < points_per_disjoint; i++) {
			int pid = base_id + i * offset;
			int joint_id = i / points_per_joint;

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

			sample_dist = rand() % SAMPLE_RAND_;

			sample_point = origin;

			if (joint_id % 3 == 0) {
				sample_point.x = origin.x + sample_dist / SAMPLE_DIST_;
			} else if (joint_id % 3 == 1) {
				sample_point.y = origin.y + sample_dist / SAMPLE_DIST_;
			} else {
				sample_point.z = origin.z + sample_dist / SAMPLE_DIST_;
			}

			sample_cloud->points[pid] = sample_point;
			status[pid] = true;
		}

		origin.x += d_th * 10;
		origin.y += d_th * 10;
		origin.z += d_th * 10;
	}


	struct timeval start, end;

	GpuEuclideanCluster2 test_sample;

	long long gpu_initial;

	gettimeofday(&start, NULL);
	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);
	test_sample.setInputPoints(sample_cloud);
	gettimeofday(&end, NULL);

	gpu_initial = timeDiff(start, end);


	long long e_total_time, e_graph_time, e_clustering_time;
	int e_itr_num;
	long long m_total_time, m_initial, m_build_matrix, m_clustering_time;
	int m_itr_num;
	long long v_total_time, v_graph_time, v_clustering_time;
	int v_itr_num;
	long long c_total_time, c_clustering_time, c_tree_build;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3(e_total_time, e_graph_time, e_clustering_time, e_itr_num);
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	e_total_time = timeDiff(start, end) + gpu_initial;

	gettimeofday(&start, NULL);
	test_sample.extractClusters5();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	gettimeofday(&start, NULL);
	test_sample.extractClusters(m_total_time, m_initial, m_build_matrix, m_clustering_time, m_itr_num);
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	m_total_time = timeDiff(start, end) + gpu_initial;
	gettimeofday(&start, NULL);
	test_sample.extractClusters2(v_total_time, v_graph_time, v_clustering_time, v_itr_num);
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	gettimeofday(&start, NULL);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

	tree->setInputCloud (sample_cloud);
	gettimeofday(&end, NULL);

	c_tree_build = timeDiff(start, end);

	std::vector<pcl::PointIndices> cluster_indices;

	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance (d_th);
	ec.setSearchMethod(tree);
	ec.setInputCloud (sample_cloud);
	ec.extract (cluster_indices);

	gettimeofday(&end, NULL);

	c_total_time = timeDiff(start, end);
	c_clustering_time = c_total_time - c_tree_build;

	output << e_total_time << "," << m_total_time << "," << v_total_time << "," << c_total_time << "," << gpu_initial << "," << e_graph_time << "," << e_clustering_time << "," << m_initial << "," << m_build_matrix << "," << m_clustering_time << "," << v_graph_time << ","  << v_clustering_time << "," << c_tree_build << "," << c_clustering_time << "," << e_itr_num << "," << m_itr_num << "," << v_itr_num;

	return output.str();
}

void GPUECTest::lineTest(int point_num)
{
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

	GpuEuclideanCluster2 test_sample;

	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);
	test_sample.setInputPoints(sample_cloud);

	struct timeval start, end;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Edge-based: " << timeDiff(start, end) << " usecs" << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Matrix-based: " << timeDiff(start, end) << " usecs" << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Vertex-based: " << timeDiff(start, end) << " usecs" << std::endl << std::endl;

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

		while (offset * points_per_disjoint + base_id > point_num) {
			offset--;
		}

		for (int i = 0; i < points_per_disjoint; i++) {
			int pid = base_id + i * offset;

			sample_cloud->points[pid] = base_cloud.cloud_->points[source_id++];
			status[pid] = true;
		}
	}


	struct timeval start, end;

	GpuEuclideanCluster2 test_sample;

	long long gpu_initial;

	gettimeofday(&start, NULL);
	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);
	test_sample.setInputPoints(sample_cloud);
	gettimeofday(&end, NULL);

	gpu_initial = timeDiff(start, end);


	long long e_total_time, e_graph_time, e_clustering_time;
	int e_itr_num;
	long long m_total_time, m_initial, m_build_matrix, m_clustering_time;
	int m_itr_num;
	long long v_total_time, v_graph_time, v_clustering_time;
	int v_itr_num;
	long long c_total_time, c_clustering_time, c_tree_build;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3(e_total_time, e_graph_time, e_clustering_time, e_itr_num);
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	e_total_time = timeDiff(start, end) + gpu_initial;

	gettimeofday(&start, NULL);
	test_sample.extractClusters(m_total_time, m_initial, m_build_matrix, m_clustering_time, m_itr_num);
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	m_total_time = timeDiff(start, end) + gpu_initial;
	gettimeofday(&start, NULL);
	test_sample.extractClusters2(v_total_time, v_graph_time, v_clustering_time, v_itr_num);
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	gettimeofday(&start, NULL);
//	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
//
//	tree->setInputCloud (sample_cloud);
	gettimeofday(&end, NULL);

//	c_tree_build = timeDiff(start, end);
//
//	std::vector<pcl::PointIndices> cluster_indices;
//
//	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
//	ec.setClusterTolerance (d_th);
//	ec.setSearchMethod(tree);
//	ec.setInputCloud (sample_cloud);
//	ec.extract (cluster_indices);

	gettimeofday(&end, NULL);

	c_total_time = timeDiff(start, end);
	c_clustering_time = c_total_time - c_tree_build;

	output << e_total_time << "," << m_total_time << "," << v_total_time << "," << c_total_time << "," << gpu_initial << "," << e_graph_time << "," << e_clustering_time << "," << m_initial << "," << m_build_matrix << "," << m_clustering_time << "," << v_graph_time << ","  << v_clustering_time << "," << c_tree_build << "," << c_clustering_time << "," << e_itr_num << "," << m_itr_num << "," << v_itr_num;

	return output.str();
}
