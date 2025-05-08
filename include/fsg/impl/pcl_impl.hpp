#pragma once

// state: true: remove points in the range
//        false: remove points out of the range (default)
template<typename T>
void PassThrough(const typename pcl::PointCloud<T>::Ptr& input,
                 const typename pcl::PointCloud<T>::Ptr& output,
                 const float min, const float max, const std::string& axis, const bool state) {
    pcl::PassThrough<T> pass;
    pass.setInputCloud(input);
    pass.setFilterFieldName(axis);
    pass.setFilterLimits(min, max);
    pass.setNegative(state);
    pass.filter(*output);
}

// passthrough filter with normal comparison
// points outside the range or
// inside the range but with normal intersection angle larger than normal_threshold (0) will be kept
template<typename T>
void PassWithNormal(const typename pcl::PointCloud<T>::Ptr& input, const typename pcl::PointCloud<T>::Ptr& output,
                    const float min, const float max, const std::string& axis,
                    const Eigen::Vector3d& normal, const float normal_threshold) {
    if (axis != "x" && axis != "y" && axis != "z") {
        std::cerr << "Error: axis must be x, y, or z." << std::endl;
        return;
    }

    typename pcl::PointCloud<T>::Ptr tmp(new pcl::PointCloud<T>);
    for (const auto& point : input->points) {
        const float coordinate = (axis == "x") ? point.x : (axis == "y") ? point.y : point.z;
        const float angle_cosine = normal.dot(point.getNormalVector3fMap().template cast<double>());
        if (coordinate < min || coordinate > max || angle_cosine < normal_threshold) {
            tmp->points.push_back(point);
        }
    }
    *output = *tmp;
}

template <typename T, typename pT>
int ObtainNearest(const typename pcl::KdTreeFLANN<T>::Ptr& tree, const pT* query_point, float& dist) {
    T point;
    pcl::copyPoint<pT, T> (*query_point, point);
    std::vector<int> idx(1);
    std::vector<float> sq_dist(1);
    tree->nearestKSearch(point, 1, idx, sq_dist);
    dist = sq_dist[0];
    return idx[0];
}

template<typename T>
float GetMeanX(const typename pcl::PointCloud<T>::Ptr& input) {
    float sum = 0;
    for (const auto& point : input->points) {
        sum += point.x;
    }
    return sum / input->size();
}

template<typename T>
void GetCentroid(const typename pcl::PointCloud<T>::Ptr& input, Eigen::Vector3d& centroid) {
    Eigen::Vector4d centroid_d;
    pcl::compute3DCentroid(*input, centroid_d);
    centroid = centroid_d.head<3>();
}

template<typename T>
void GetMeanNormal(const typename pcl::PointCloud<T>::Ptr& input, Eigen::Vector3d& normal) {
    Eigen::Vector3f normal_f = Eigen::Vector3f::Zero();
    for (const auto& point : input->points) {
        normal_f += point.getNormalVector3fMap();
    }
    normal_f.normalize();
    normal = normal_f.cast<double>();
}

template<typename T>
void GetMaxR(const typename pcl::PointCloud<T>::Ptr& input, const Eigen::Vector3d& centroid, float& max_r) {
    max_r = 0;
    const Eigen::Vector3f centroid_f = centroid.cast<float>();
    for (const auto& point : input->points) {
        Eigen::Vector3f point_f = point.getVector3fMap();
        const float r = (point_f - centroid_f).norm();
        max_r = std::max(max_r, r);
    }
}

template<typename T>
bool IsExist(const typename pcl::PointCloud<T>::Ptr& input, std::vector<float>& min_max, const int num_threshold) {
    typename pcl::PointCloud<T>::Ptr left (new pcl::PointCloud<T>);
    PassThrough<T>(input, left, min_max[0], min_max[1], "x");
    PassThrough<T>( left, left, min_max[2], min_max[3], "y");
    PassThrough<T>( left, left, min_max[4], min_max[5], "z");
    return left->size() > num_threshold;
}