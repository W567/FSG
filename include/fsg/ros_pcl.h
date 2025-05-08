#ifndef FSG_ROS_PCL_H_
#define FSG_ROS_PCL_H_

#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

template <typename T>
void PcMsg2Pcl(const sensor_msgs::PointCloud2ConstPtr& input,
               const typename pcl::PointCloud<T>::Ptr& output) {
    fromROSMsg(*input, *output);
}

template <typename T>
void Transform(const typename pcl::PointCloud<T>::ConstPtr& input,
               const typename pcl::PointCloud<T>::Ptr& output,
               const std::string& target_frame_id_,
               const tf::TransformListener& tf_listener_) {
    try {
        if (!pcl_ros::transformPointCloud(target_frame_id_, *input, *output, tf_listener_)) {
            ROS_ERROR("Failed to transform point cloud");
        }
    } catch (tf2::ConnectivityException &e) {
        ROS_ERROR("Transform error: %s", e.what());
    } catch (...) {
        ROS_ERROR("Unknown transform error");
    }
}

#endif  // FSG_ROS_PCL_H_
