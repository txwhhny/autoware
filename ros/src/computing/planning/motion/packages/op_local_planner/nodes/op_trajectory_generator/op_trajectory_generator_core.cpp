/*
 * @Description: 
 * @version: 
 * @Company: 
 * @Author: hxc
 * @Date: 2019-08-12 12:08:02
 * @LastEditors: hxc
 * @LastEditTime: 2019-08-13 08:35:12
 */
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
 */

#include "op_trajectory_generator_core.h"
#include "op_ros_helpers/op_ROSHelpers.h"


namespace TrajectoryGeneratorNS
{

TrajectoryGen::TrajectoryGen()
{
	bInitPos = false;
	bNewCurrentPos = false;
	bVehicleStatus = false;
	bWayGlobalPath = false;

	ros::NodeHandle _nh;
	UpdatePlanningParams(_nh);

	tf::StampedTransform transform;
	PlannerHNS::ROSHelpers::GetTransformFromTF("map", "world", transform);
	m_OriginPos.position.x  = transform.getOrigin().x();
	m_OriginPos.position.y  = transform.getOrigin().y();
	m_OriginPos.position.z  = transform.getOrigin().z();

	pub_LocalTrajectories = nh.advertise<autoware_msgs::LaneArray>("local_trajectories", 1);
	pub_LocalTrajectoriesRviz = nh.advertise<visualization_msgs::MarkerArray>("local_trajectories_gen_rviz", 1);

	sub_initialpose = nh.subscribe("/initialpose", 1, &TrajectoryGen::callbackGetInitPose, this);
	sub_current_pose = nh.subscribe("/current_pose", 10, &TrajectoryGen::callbackGetCurrentPose, this);

	int bVelSource = 1;
	_nh.getParam("/op_trajectory_generator/velocitySource", bVelSource);
	if(bVelSource == 0)
		sub_robot_odom = nh.subscribe("/odom", 10,	&TrajectoryGen::callbackGetRobotOdom, this);
	else if(bVelSource == 1)
		sub_current_velocity = nh.subscribe("/current_velocity", 10, &TrajectoryGen::callbackGetVehicleStatus, this);
	else if(bVelSource == 2)
		sub_can_info = nh.subscribe("/can_info", 10, &TrajectoryGen::callbackGetCANInfo, this);

	sub_GlobalPlannerPaths = nh.subscribe("/lane_waypoints_array", 1, &TrajectoryGen::callbackGetGlobalPlannerPath, this);
}

TrajectoryGen::~TrajectoryGen()
{
}

void TrajectoryGen::UpdatePlanningParams(ros::NodeHandle& _nh)
{
	_nh.getParam("/op_trajectory_generator/samplingTipMargin", m_PlanningParams.carTipMargin);		// 从车辆中心点到水平采样的起点，这部分的长度决定了车辆切换不同轨迹的平滑程度
	_nh.getParam("/op_trajectory_generator/samplingOutMargin", m_PlanningParams.rollInMargin);	//从水平采样的起点到平行采样的起点，这部分的长度和车辆速度密切相关，车辆速度越快，rollin部分应越长，使得轨迹更加平滑。
	_nh.getParam("/op_trajectory_generator/samplingSpeedFactor", m_PlanningParams.rollInSpeedFactor);	// 速度因子, 该值*当前车速(如ndt_matching给出的速度)+m_PlanningParams.rollInMargin作为真实的rollin进行计算
	_nh.getParam("/op_trajectory_generator/enableHeadingSmoothing", m_PlanningParams.enableHeadingSmoothing);	// 暂未使用,那部分代码被注释掉了

	_nh.getParam("/op_common_params/enableSwerving", m_PlanningParams.enableSwerving);					// 是否可以急转弯, 应该为true, 故意"基本未使用"
	if(m_PlanningParams.enableSwerving)
		m_PlanningParams.enableFollowing = true;
	else
		_nh.getParam("/op_common_params/enableFollowing", m_PlanningParams.enableFollowing);		// 未使用

	_nh.getParam("/op_common_params/enableTrafficLightBehavior", m_PlanningParams.enableTrafficLightBehavior);	// 未使用
	_nh.getParam("/op_common_params/enableStopSignBehavior", m_PlanningParams.enableStopSignBehavior);	// 未使用

	_nh.getParam("/op_common_params/maxVelocity", m_PlanningParams.maxSpeed);		// 未使用
	_nh.getParam("/op_common_params/minVelocity", m_PlanningParams.minSpeed);		// 未使用
	_nh.getParam("/op_common_params/maxLocalPlanDistance", m_PlanningParams.microPlanDistance);		//cartip->rollin->rollout,是rollout部分的距离, 也就是规划出来的候选轨迹到了平行区域之后的距离

	_nh.getParam("/op_common_params/pathDensity", m_PlanningParams.pathDensity);						// tracjectory轨迹点的密集程度,也就是点的间距,处理成均匀的
	_nh.getParam("/op_common_params/rollOutDensity", m_PlanningParams.rollOutDensity);			// rollout部分的间距, 也就是候选轨迹平行线之间的间距
	if(m_PlanningParams.enableSwerving)
		_nh.getParam("/op_common_params/rollOutsNumber", m_PlanningParams.rollOutNumber);			// 规划的路径数量是该值+1
	else
		m_PlanningParams.rollOutNumber = 0;

	_nh.getParam("/op_common_params/horizonDistance", m_PlanningParams.horizonDistance);		// 局部路径最大距离
	_nh.getParam("/op_common_params/minFollowingDistance", m_PlanningParams.minFollowingDistance);	// 未使用
	_nh.getParam("/op_common_params/minDistanceToAvoid", m_PlanningParams.minDistanceToAvoid);		  // 未使用
	_nh.getParam("/op_common_params/maxDistanceToAvoid", m_PlanningParams.maxDistanceToAvoid);			// 未使用
	_nh.getParam("/op_common_params/speedProfileFactor", m_PlanningParams.speedProfileFactor);			// 未使用

	_nh.getParam("/op_common_params/smoothingDataWeight", m_PlanningParams.smoothingDataWeight);		// 平滑处理是迭代多次的, 相当于一个因子, factor*(处理前-处理后)
	_nh.getParam("/op_common_params/smoothingSmoothWeight", m_PlanningParams.smoothingSmoothWeight);// 平滑处理中, 相当于一个因子, factor*((前一点+后一点))-2当前点)

	_nh.getParam("/op_common_params/horizontalSafetyDistance", m_PlanningParams.horizontalSafetyDistancel);	// 未使用
	_nh.getParam("/op_common_params/verticalSafetyDistance", m_PlanningParams.verticalSafetyDistance);	// 未使用

	_nh.getParam("/op_common_params/enableLaneChange", m_PlanningParams.enableLaneChange);	// 未使用

	_nh.getParam("/op_common_params/width", m_CarInfo.width);
	_nh.getParam("/op_common_params/length", m_CarInfo.length);
	_nh.getParam("/op_common_params/wheelBaseLength", m_CarInfo.wheel_base);
	_nh.getParam("/op_common_params/turningRadius", m_CarInfo.turning_radius);
	_nh.getParam("/op_common_params/maxSteerAngle", m_CarInfo.max_steer_angle);
	_nh.getParam("/op_common_params/maxAcceleration", m_CarInfo.max_acceleration);
	_nh.getParam("/op_common_params/maxDeceleration", m_CarInfo.max_deceleration);

	m_CarInfo.max_speed_forward = m_PlanningParams.maxSpeed;
	m_CarInfo.min_speed_forward = m_PlanningParams.minSpeed;

}

void TrajectoryGen::callbackGetInitPose(const geometry_msgs::PoseWithCovarianceStampedConstPtr &msg)		// 这里的pose是world坐标系下的
{
	if(!bInitPos)
	{
		m_InitPos = PlannerHNS::WayPoint(msg->pose.pose.position.x+m_OriginPos.position.x,
				msg->pose.pose.position.y+m_OriginPos.position.y,
				msg->pose.pose.position.z+m_OriginPos.position.z,
				tf::getYaw(msg->pose.pose.orientation));
		m_CurrentPos = m_InitPos;
		bInitPos = true;
	}
}

void TrajectoryGen::callbackGetCurrentPose(const geometry_msgs::PoseStampedConstPtr& msg)		// 这里的pose是map坐标系下的
{
	m_CurrentPos = PlannerHNS::WayPoint(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z, tf::getYaw(msg->pose.orientation));
	m_InitPos = m_CurrentPos;
	bNewCurrentPos = true;
	bInitPos = true;
}

void TrajectoryGen::callbackGetVehicleStatus(const geometry_msgs::TwistStampedConstPtr& msg)
{
	m_VehicleStatus.speed = msg->twist.linear.x;
	m_CurrentPos.v = m_VehicleStatus.speed;
	if(fabs(msg->twist.linear.x) > 0.25)
		m_VehicleStatus.steer = atan(m_CarInfo.wheel_base * msg->twist.angular.z/msg->twist.linear.x);		// 轴距 / 转弯半径, 转弯角度?
	UtilityHNS::UtilityH::GetTickCount(m_VehicleStatus.tStamp);
	bVehicleStatus = true;
}

void TrajectoryGen::callbackGetCANInfo(const autoware_can_msgs::CANInfoConstPtr &msg)
{
	m_VehicleStatus.speed = msg->speed/3.6;
	m_VehicleStatus.steer = msg->angle * m_CarInfo.max_steer_angle / m_CarInfo.max_steer_value;
	UtilityHNS::UtilityH::GetTickCount(m_VehicleStatus.tStamp);
	bVehicleStatus = true;
}

void TrajectoryGen::callbackGetRobotOdom(const nav_msgs::OdometryConstPtr& msg)
{
	m_VehicleStatus.speed = msg->twist.twist.linear.x;
	m_VehicleStatus.steer += atan(m_CarInfo.wheel_base * msg->twist.twist.angular.z/msg->twist.twist.linear.x);		// (轴距 / 半径)-相当于车的航向?
	UtilityHNS::UtilityH::GetTickCount(m_VehicleStatus.tStamp);
	bVehicleStatus = true;
}

void TrajectoryGen::callbackGetGlobalPlannerPath(const autoware_msgs::LaneArrayConstPtr& msg)
{
	if(msg->lanes.size() > 0)
	{
		bool bOldGlobalPath = m_GlobalPaths.size() == msg->lanes.size();

		m_GlobalPaths.clear();

		for(unsigned int i = 0 ; i < msg->lanes.size(); i++)
		{
			PlannerHNS::ROSHelpers::ConvertFromAutowareLaneToLocalLane(msg->lanes.at(i), m_temp_path);	// 只是换个数据格式

			PlannerHNS::PlanningHelpers::CalcAngleAndCost(m_temp_path);
			m_GlobalPaths.push_back(m_temp_path);

			if(bOldGlobalPath)
			{
				bOldGlobalPath = PlannerHNS::PlanningHelpers::CompareTrajectories(m_temp_path, m_GlobalPaths.at(i));	// 比较各航点的速度,坐标,经纬度是否相同, 用来判断航线是否变了
			}
		}

		if(!bOldGlobalPath)
		{
			bWayGlobalPath = true;
			std::cout << "Received New Global Path Generator ! " << std::endl;
		}
		else
		{
			m_GlobalPaths.clear();
		}
	}
}

void TrajectoryGen::MainLoop()
{
	ros::Rate loop_rate(100);

	PlannerHNS::WayPoint prevState, state_change;

	while (ros::ok())
	{
		ros::spinOnce();

		if(bInitPos && m_GlobalPaths.size()>0)		// 设置了初始位置(或者有current_pose)和得到全局路径集合
		{
			m_GlobalPathSections.clear();

			for(unsigned int i = 0; i < m_GlobalPaths.size(); i++)		// 遍历全局路径集合
			{
				t_centerTrajectorySmoothed.clear();
				PlannerHNS::PlanningHelpers::ExtractPartFromPointToDistanceDirectionFast(m_GlobalPaths.at(i), m_CurrentPos, m_PlanningParams.horizonDistance ,
						m_PlanningParams.pathDensity ,t_centerTrajectorySmoothed);	// 找出最接近m_CurrentPos的航点,往后方10m,往前(m_PlanningParams.horizonDistance - 10)m,以m_PlanningParams.pathDensity为距离均匀处理,生成新的路径t_centerTrajectorySmoothed

				m_GlobalPathSections.push_back(t_centerTrajectorySmoothed);		// 对处理的片段保存到m_GlobalPathSections
			}

			std::vector<PlannerHNS::WayPoint> sampledPoints_debug;
			m_Planner.GenerateRunoffTrajectory(m_GlobalPathSections, m_CurrentPos,
								m_PlanningParams.enableLaneChange,
								m_VehicleStatus.speed,
								m_PlanningParams.microPlanDistance,
								m_PlanningParams.maxSpeed,
								m_PlanningParams.minSpeed,
								m_PlanningParams.carTipMargin,
								m_PlanningParams.rollInMargin,
								m_PlanningParams.rollInSpeedFactor,
								m_PlanningParams.pathDensity,
								m_PlanningParams.rollOutDensity,
								m_PlanningParams.rollOutNumber,
								m_PlanningParams.smoothingDataWeight,
								m_PlanningParams.smoothingSmoothWeight,
								m_PlanningParams.smoothingToleranceError,			// 公差默认为0.05
								m_PlanningParams.speedProfileFactor,
								m_PlanningParams.enableHeadingSmoothing,
								-1 , -1,
								m_RollOuts, sampledPoints_debug);

			autoware_msgs::LaneArray local_lanes;
			for(unsigned int i=0; i < m_RollOuts.size(); i++)
			{
				for(unsigned int j=0; j < m_RollOuts.at(i).size(); j++)
				{
					autoware_msgs::Lane lane;		// PredictConstantTimeCostForTrajectory:根据当m_CurrentPos的当前车速,估计轨迹上各个航点的timecost
					PlannerHNS::PlanningHelpers::PredictConstantTimeCostForTrajectory(m_RollOuts.at(i).at(j), m_CurrentPos, m_PlanningParams.minSpeed, m_PlanningParams.microPlanDistance);
					PlannerHNS::ROSHelpers::ConvertFromLocalLaneToAutowareLane(m_RollOuts.at(i).at(j), lane);		// 转换成ros消息, 发布出来
					lane.closest_object_distance = 0;
					lane.closest_object_velocity = 0;
					lane.cost = 0;
					lane.is_blocked = false;
					lane.lane_index = i;
					local_lanes.lanes.push_back(lane);
				}
			}
			pub_LocalTrajectories.publish(local_lanes);
		}
		else
			sub_GlobalPlannerPaths = nh.subscribe("/lane_waypoints_array", 	1,		&TrajectoryGen::callbackGetGlobalPlannerPath, 	this);

		visualization_msgs::MarkerArray all_rollOuts;
		PlannerHNS::ROSHelpers::TrajectoriesToMarkers(m_RollOuts, all_rollOuts);
		pub_LocalTrajectoriesRviz.publish(all_rollOuts);

		loop_rate.sleep();
	}
}

}
