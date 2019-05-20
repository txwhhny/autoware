/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ********************
 *
 */

#include "object_map/object_map_utils.hpp"

namespace object_map
{
	geometry_msgs::Point TransformPoint(const geometry_msgs::Point &in_point, const tf::Transform &in_tf)
	{
		tf::Point tf_point;
		tf::pointMsgToTF(in_point, tf_point);	// msg类型的点转成tf的点

		tf_point = in_tf * tf_point;		// 变换

		geometry_msgs::Point out_point;			// 转会msg类型的点
		tf::pointTFToMsg(tf_point, out_point);

		return out_point;
	}

	void PublishGridMap(const grid_map::GridMap &in_gridmap, const ros::Publisher &in_publisher)
	{
		grid_map_msgs::GridMap message;
		grid_map::GridMapRosConverter::toMessage(in_gridmap, message);
		in_publisher.publish(message);
	}

	void PublishOccupancyGrid(const grid_map::GridMap &in_gridmap,
	                          const ros::Publisher &in_publisher,
	                          const std::string& in_layer,
	                          double in_min_value,
	                          double in_max_value,
	                          double in_height)
	{
		nav_msgs::OccupancyGrid message;
		grid_map::GridMapRosConverter::toOccupancyGrid(in_gridmap, in_layer, in_min_value, in_max_value, message );
		message.info.origin.position.z = in_height;
		in_publisher.publish(message);
	}

	tf::StampedTransform FindTransform(const std::string &in_target_frame, const std::string &in_source_frame,
	                                   const tf::TransformListener &in_tf_listener)
	{
		tf::StampedTransform transform;

		try
		{
			in_tf_listener.lookupTransform(in_target_frame, in_source_frame, ros::Time(0), transform);
		}
		catch (tf::TransformException ex)
		{
			ROS_ERROR("%s", ex.what());
		}

		return transform;
	}

	std::vector<geometry_msgs::Point>
	SearchAreaPoints(const vector_map::Area &in_area, const vector_map::VectorMap &in_vectormap)	// 从这个area出发，从线找到点
	{
		std::vector<geometry_msgs::Point> area_points;
		std::vector<geometry_msgs::Point> area_points_empty;

		if (in_area.aid == 0)
			return area_points_empty;

		vector_map_msgs::Line line = in_vectormap.findByKey(vector_map::Key<vector_map_msgs::Line>(in_area.slid));
		// must set beginning line
		if (line.lid == 0 || line.blid != 0)
			return area_points_empty;

		// Search all lines in in_area
		while (line.flid != 0)
		{
			vector_map_msgs::Point bp = in_vectormap.findByKey(vector_map::Key<vector_map_msgs::Point>(line.bpid));
			if (bp.pid == 0)
				return area_points_empty;

			vector_map_msgs::Point fp = in_vectormap.findByKey(vector_map::Key<vector_map_msgs::Point>(line.fpid));
			if (fp.pid == 0)
				return area_points_empty;

			// 2 points of line
			area_points.push_back(vector_map::convertPointToGeomPoint(bp));		// 线的两个端点
			area_points.push_back(vector_map::convertPointToGeomPoint(fp));

			line = in_vectormap.findByKey(vector_map::Key<vector_map_msgs::Line>(line.flid));		// 查找下一条线
			if (line.lid == 0)
				return area_points_empty;
		}

		vector_map_msgs::Point bp = in_vectormap.findByKey(vector_map::Key<vector_map_msgs::Point>(line.bpid));
		vector_map_msgs::Point fp = in_vectormap.findByKey(vector_map::Key<vector_map_msgs::Point>(line.fpid));
		if (bp.pid == 0 || fp.pid == 0)
			return area_points_empty;

		area_points.push_back(vector_map::convertPointToGeomPoint(bp));
		area_points.push_back(vector_map::convertPointToGeomPoint(fp));

		return area_points;
	}

	void FillPolygonAreas(grid_map::GridMap &out_grid_map, const std::vector<std::vector<geometry_msgs::Point>> &in_area_points,
		                      const std::string &in_grid_layer_name, const int in_layer_background_value,
		                      const int in_layer_min_value, const int in_fill_color, const int in_layer_max_value,
		                      const std::string &in_tf_target_frame, const std::string &in_tf_source_frame,
		                      const tf::TransformListener &in_tf_listener)
	{
		if(!out_grid_map.exists(in_grid_layer_name))
		{
			out_grid_map.add(in_grid_layer_name);
		}
		out_grid_map[in_grid_layer_name].setConstant(in_layer_background_value);		// 初始化GridMap

		cv::Mat original_image;
		grid_map::GridMapCvConverter::toImage<unsigned char, 1>(out_grid_map,
		                                                        in_grid_layer_name,
		                                                        CV_8UC1,
		                                                        in_layer_min_value,
		                                                        in_layer_max_value,
		                                                        original_image);

		cv::Mat filled_image = original_image.clone();

		tf::StampedTransform tf = FindTransform(in_tf_target_frame, in_tf_source_frame, in_tf_listener);	// 获得从map到lidar的tf

		// calculate out_grid_map position
		grid_map::Position map_pos = out_grid_map.getPosition();	// 得到out_grid_map的中心点坐标
		double origin_x_offset = out_grid_map.getLength().x() / 2.0 - map_pos.x();
		double origin_y_offset = out_grid_map.getLength().y() / 2.0 - map_pos.y();

		for (const auto &points : in_area_points)		// 多条路的area
		{
			std::vector<cv::Point> cv_points;

			for (const auto &p : points)		// 一个area
			{
				// transform to GridMap coordinate
				geometry_msgs::Point tf_point = TransformPoint(p, tf);

				// coordinate conversion for cv image
				double cv_x = (out_grid_map.getLength().y() - origin_y_offset - tf_point.y) / out_grid_map.getResolution();
				double cv_y = (out_grid_map.getLength().x() - origin_x_offset - tf_point.x) / out_grid_map.getResolution();
				cv_points.emplace_back(cv::Point(cv_x, cv_y));
			}

			// 根据顶点绘制多边形，返回到filled_image，顶点指针，顶点个数，颜色
			cv::fillConvexPoly(filled_image, cv_points.data(), cv_points.size(), cv::Scalar(in_fill_color));		// in_fill_color==0,所以是黑色，黑色可通行
		}

		// convert to ROS msg
		grid_map::GridMapCvConverter::addLayerFromImage<unsigned char, 1>(filled_image,
		                                                                  in_grid_layer_name,
		                                                                  out_grid_map,
		                                                                  in_layer_min_value,
		                                                                  in_layer_max_value);
	}

	void LoadRoadAreasFromVectorMap(ros::NodeHandle& in_private_node_handle,
	                                std::vector<std::vector<geometry_msgs::Point>>& out_area_points)
	{
		vector_map::VectorMap vmap;
		vmap.subscribe(in_private_node_handle,	// /vector_map_info/point,/vector_map_info/line,/vector_map_info/area,/vector_map_info/way_area
		               vector_map::Category::POINT | vector_map::Category::LINE | vector_map::Category::AREA |
		               vector_map::Category::WAY_AREA, 10);		// 订阅指定Category相应的话题，事件循环尝试10次

		std::vector<vector_map_msgs::WayArea> way_areas =
				vmap.findByFilter([](const vector_map_msgs::WayArea &way_area)		// lambda恒返回true，表示返回所有WayArea数据
				                  {
					                  return true;
				                  });

		if (way_areas.empty())
		{
			ROS_WARN_STREAM("No WayArea...");
			return;
		}

		for (const auto &way_area : way_areas)
		{
			vector_map_msgs::Area area = vmap.findByKey(vector_map::Key<vector_map::Area>(way_area.aid));		// 根据way_area(有waid和aid)的aid，查找area
			out_area_points.emplace_back(SearchAreaPoints(area, vmap));
		}

	}

} // namespace object_map
