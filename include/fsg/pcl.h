#ifndef FSG_PCL_H_
#define FSG_PCL_H_

#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/copy_point.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>


template<typename T>
void PassThrough(const typename pcl::PointCloud<T>::Ptr& input, const typename pcl::PointCloud<T>::Ptr& output,
                 float min, float max, const std::string& axis, bool state=false);

template<typename T>
void PassWithNormal(const typename pcl::PointCloud<T>::Ptr& input, const typename pcl::PointCloud<T>::Ptr& output,
                    float min, float max, const std::string& axis,
                    const Eigen::Vector3d& normal, float normal_threshold);

template<typename T, typename pT>
int ObtainNearest(const typename pcl::KdTreeFLANN<T>::Ptr& tree, const pT* query_point, float& dist);

template<typename T>
float GetMeanX(const typename pcl::PointCloud<T>::Ptr& input);

template<typename T>
void GetCentroid(const typename pcl::PointCloud<T>::Ptr& input, Eigen::Vector3d& centroid);

template<typename T>
void GetMeanNormal(const typename pcl::PointCloud<T>::Ptr& input, Eigen::Vector3d& normal);

template<typename T>
void GetMaxR(const typename pcl::PointCloud<T>::Ptr& input, const Eigen::Vector3d& centroid, float& max_r);

template<typename T>
bool IsExist(const typename pcl::PointCloud<T>::Ptr& input, std::vector<float>& min_max, int num_threshold = 0);

#include "impl/pcl_impl.hpp"

#endif  // FSG_PCL_H_