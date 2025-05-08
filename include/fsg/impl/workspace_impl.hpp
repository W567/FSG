#pragma once

#include <ros/package.h>

namespace fsg {

    template <typename PointT>
    int Workspace<PointT>::InitWorkspace(
            const std::vector<std::string>& ws_paths, const int finger_num,
            const double tip_interval, const double fin_collision_thre, const std::vector<float>& palm_min_max) {
        finger_num_ = finger_num;
        tip_interval_ = tip_interval;
        fin_collision_threshold_ = fin_collision_thre;
        palm_min_max_ = palm_min_max;

        return GetWorkspaceClouds(ws_paths);
    }

    template <typename PointT>
    int Workspace<PointT>::GetWorkspaceClouds(const std::vector<std::string>& ws_paths) {
        if (ws_paths.size() != finger_num_) {
            CERR_RED("[initWorkspace] Size of ws_paths not equal to finger_num_")
            return -1;
        }
        tips_trees_.clear();
        tips_original_.clear();
        valid_finger_list_.clear();
        valid_finger_num_ = finger_num_;
        const std::string package_path = ros::package::getPath("fsg") + "/";
        for (const auto& path : ws_paths) {
            CloudPtr fin (new CloudT);
            if (pcl::io::loadPCDFile(package_path + path, *fin) == -1) {
                // if failed to load workspace cloud, disable the finger
                CERR_RED("[initWorkspace] Failed to load " << path)
                if (valid_finger_list_.empty()) {
                    CERR_RED("[initWorkspace] Primary finger workspace cloud invalid")
                    return -1;
                }
                valid_finger_num_--;
                valid_finger_list_.emplace_back(false);
            } else {
                valid_finger_list_.emplace_back(true);
            }
            tips_original_.emplace_back(fin);

            TreePtr tree (new TreeT);
            tips_trees_.emplace_back(tree);
        }
        WsXOrder();
        face_type_ = GetFaceType();
        return face_type_;
    }

    template <typename PointT>
    void Workspace<PointT>::GetLinkCol(const std::vector<std::string>& link_col_paths) {
        if (link_col_paths.size() != finger_num_) {
            CERR_RED("[getLinkCol] Size of link_col_paths not equal to finger_num_")
            return;
        }
        link_original_.clear();
        const std::string package_path = ros::package::getPath("fsg") + "/";
        for (const auto& path : link_col_paths) {
            CloudPtr link (new CloudT);
            if (pcl::io::loadPCDFile(package_path + path, *link) == -1) {
                CERR_RED("[getLinkCol] Failed to load link_col pc " << path)
            }
            link_original_.emplace_back(link);
        }
    }

    template <typename PointT>
    bool Workspace<PointT>::CheckLinkCollision(const CloudPnPtr& obj_transformed, std::vector<float>& joint_angles) {
        const TreePnPtr obj_tree(new TreePn);
        obj_tree->setInputCloud(obj_transformed);
        const int angle_num_ = joint_angles.size() / finger_num_;
#ifdef BUILD_WITH_VISUALIZER
        link_skeletons_.clear();
#endif
        for (int i = 0; i < finger_num_; i++) {
            if (!valid_finger_list_[i]) continue;
            if (required_finger_list_[i] == false) continue;
            if (link_original_[i]->points.empty()) continue;
            const CloudPnPtr link_col_cloud(new CloudPn);
            std::vector<float> finger_joint_angles(joint_angles.begin() + i * angle_num_, joint_angles.begin() + (i + 1) * angle_num_);
            for (const auto& link_point : link_original_[i]->points) {
                if (std::equal(link_point.angle, link_point.angle + angle_num_, finger_joint_angles.begin())) {
                    pcl::PointNormal point;
                    point.x = link_point.x;
                    point.y = link_point.y;
                    point.z = link_point.z;
                    link_col_cloud->points.push_back(point);
                }
            }
#ifdef BUILD_WITH_VISUALIZER
            link_skeletons_.emplace_back(link_col_cloud);
#endif
            float dist;
            const int first_idx = ObtainNearest<pcl::PointNormal, pcl::PointNormal>(obj_tree, &link_col_cloud->points[0], dist);
            V3f prev_obj2link = link_col_cloud->points[0].getVector3fMap() - obj_transformed->points[first_idx].getVector3fMap();
            V3f obj_normal = obj_transformed->points[first_idx].getNormalVector3fMap();
            for (auto& point : link_col_cloud->points) {
                const int idx = ObtainNearest<pcl::PointNormal, pcl::PointNormal>(obj_tree, &point, dist);
                V3f obj2link = point.getVector3fMap() - obj_transformed->points[idx].getVector3fMap();
                if (obj2link.dot(obj_normal) * prev_obj2link.dot(obj_normal) < 0) {
                    std::cerr << "Collision detected" << std::endl;
                    return true;
                }
                prev_obj2link = obj2link;
                obj_normal = obj_transformed->points[idx].getNormalVector3fMap();
            }
        }
        return false;
    }

    template <typename PointT>
    int Workspace<PointT>::GetFaceType() {
        int index = valid_finger_list_[0];
        if (tips_original_[index]->points[0].xx == 0 && tips_original_[index]->points[0].xy == 0 && tips_original_[index]->points[0].xz == 0 &&
            tips_original_[index]->points[0].zx == 0 && tips_original_[index]->points[0].zy == 0 && tips_original_[index]->points[0].zz == 0) {
            return 0;  // 0 - point
        }
        if (tips_original_[index]->points[0].xx != 0 && tips_original_[index]->points[0].xy == 0 && tips_original_[index]->points[0].xz == 0 &&
            tips_original_[index]->points[0].zx == 0 && tips_original_[index]->points[0].zy == 0 && tips_original_[index]->points[0].zz == 0) {
            return 1;  // 1 - circle, xx is the radius
        }
        return 2; // 2 - rectangle
    }

// sort finger clouds based on x coordinate of centroid (primary side and opposite side separately)
    template <typename PointT>
    void Workspace<PointT>::WsXOrder() {
        ws_centroid_x_descent_indices_primary_ = {0};
        ws_centroid_x_descent_indices_opposite_.clear();

        // clustering based on normal direction
        V3d normal, new_normal;
        GetMeanNormal<PointT>(tips_original_[0], normal);
        for (int i = 1; i < finger_num_; i++) {
            if (!valid_finger_list_[i]) continue;
            GetMeanNormal<PointT>(tips_original_[i], new_normal);
            if (normal.dot(new_normal) > 0.0) {
                // same direction with primary finger
                ws_centroid_x_descent_indices_primary_.emplace_back(i);
            } else {
                // opposite direction with primary finger
                ws_centroid_x_descent_indices_opposite_.emplace_back(i);
            }
        }

        // x coordinate of centroid of each tip cloud
        std::vector<float> xs(tips_original_.size());
        std::transform(tips_original_.begin(), tips_original_.end(), xs.begin(),
                       [](const auto& ws) { return GetMeanX<PointT>(ws); });

        // indices of tip clouds in descending order of x coordinate of centroid
        auto sort_by_mean_x_desc = [&xs](const size_t a, const size_t b) { return xs[a] > xs[b]; };
        std::sort(ws_centroid_x_descent_indices_primary_.begin(), ws_centroid_x_descent_indices_primary_.end(), sort_by_mean_x_desc);
        std::sort(ws_centroid_x_descent_indices_opposite_.begin(), ws_centroid_x_descent_indices_opposite_.end(), sort_by_mean_x_desc);
    }

    template <typename PointT>
    void Workspace<PointT>::UpdateTree() {
        for (int i = 0; i < finger_num_; i++) {
            if (!valid_finger_list_[i]) continue;
            tips_trees_[i]->setInputCloud(tips_original_[i]);
        }
    }

// reduce finger list by gradually disable current last active finger
// at least 2 fingers are required
    template <typename PointT>
    void Workspace<PointT>::ReduceFinger() {
        if (required_finger_num_ <= 2) {
            return;
        }
        // disable the last active finger
        required_finger_num_--;
        required_finger_list_[required_finger_num_] = 0;
    }

// Get the fingertip with the least number of points in its workspace cloud
    template <typename PointT>
    void Workspace<PointT>::GetMinorTip() {
        int size_min = INT_MAX;
        for (int i = 0; i < finger_num_; i++) {
            if (required_finger_list_[i] == 0) continue;
            if (!valid_finger_list_[i]) continue;
            const int size_tmp = tips_original_[i]->size();
            // required cloud is empty
            if (size_tmp == 0) {
                CERR_RED("[getMinorTip] Empty cloud in tips_original_[" << i << "]")
                return;
            }
            if (size_tmp < size_min) {
                size_min = size_tmp;
                minor_tip_id_ = i;
            }
        }

        V3f averaged_minor_normal_f = V3f::Zero();
        for (const auto& point : tips_original_[minor_tip_id_]->points) {
            averaged_minor_normal_f += point.getNormalVector3fMap();
        }
        averaged_minor_normal_f.normalize();
        averaged_minor_normal_ = averaged_minor_normal_f.cast<double>();
    }

// Set required_finger_num_ based on fingertip interval and object max length
    template <typename PointT>
    void Workspace<PointT>::SetMaxFinList(const float obj_max_len) {
        if (tip_interval_ > 0) {
            const int max_finger_num = static_cast<int>(std::ceil(obj_max_len / tip_interval_) + 1);
            required_finger_num_ = std::min(max_finger_num, valid_finger_num_);
        } else {
            required_finger_num_ = valid_finger_num_;
        }

        required_finger_list_.resize(finger_num_, 0);
        int count = 0;
        for (size_t i = 0; i < finger_num_; ++i) {
            if (count < required_finger_num_ && valid_finger_list_[i]) {
                required_finger_list_[i] = 1;
                count++;
            }
        }
        if (count != required_finger_num_) {
            CERR_RED("[setMaxFinList] required_finger_num_ is not satisfied")
        }
    }

    template <typename PointT>
    bool Workspace<PointT>::CheckHandCollision(
            const V3d&  obj_origin, const V3d&  obj_y_axis, const CloudPnPtr& obj_transformed,
            const float height_threshold, const int num_threshold) {
        // Hand & desk collision checking, palm position above object bottom
        if (-obj_origin.dot(obj_y_axis) < height_threshold) { return true; }
        // Hand & object collision checking
        return IsExist<pcl::PointNormal>(obj_transformed, palm_min_max_, num_threshold);
    }

// To avoid collision between tips and desk
// Check position of points from tips under transformed object coordinate system
    template <typename PointT>
    bool Workspace<PointT>::CheckFinCollision(
            const V3d& obj_origin, const V3d& obj_y_axis, const std::vector<CloudPnPtr>& tip_faces) const {
        const V3f obj_origin_f = obj_origin.cast<float>();
        const V3f obj_y_axis_f = obj_y_axis.cast<float>();
        for (const auto& face : tip_faces) {
            for (const auto& point : face->points) {
                if ((point.getVector3fMap() - obj_origin_f).dot(obj_y_axis_f) < fin_collision_threshold_) {
                    return true;
                }
            }
        }
        return false;  // No collision detected
    }

// To get the face (corner points) of the fingertip
    template <typename PointT>
    void Workspace<PointT>::GetRectangleFace(const PointT& center, std::vector<CloudPnPtr>& tip_faces) {
        const CloudPnPtr face_tmp (new CloudPn);
        pcl::PointNormal point_tmp;
        point_tmp.x = center.x + center.xx + center.zx;
        point_tmp.y = center.y + center.xy + center.zy;
        point_tmp.z = center.z + center.xz + center.zz;
        point_tmp.normal_x = center.normal_x;
        point_tmp.normal_y = center.normal_y;
        point_tmp.normal_z = center.normal_z;
        face_tmp->points.push_back(point_tmp);

        point_tmp.x = center.x - center.xx + center.zx;
        point_tmp.y = center.y - center.xy + center.zy;
        point_tmp.z = center.z - center.xz + center.zz;
        face_tmp->points.push_back(point_tmp);

        point_tmp.x = center.x - center.xx - center.zx;
        point_tmp.y = center.y - center.xy - center.zy;
        point_tmp.z = center.z - center.xz - center.zz;
        face_tmp->points.push_back(point_tmp);

        point_tmp.x = center.x + center.xx - center.zx;
        point_tmp.y = center.y + center.xy - center.zy;
        point_tmp.z = center.z + center.xz - center.zz;
        face_tmp->points.push_back(point_tmp);

        tip_faces.push_back(face_tmp);
    }

    template <typename PointT>
    void Workspace<PointT>::GetCircleFace(const PointT& center, std::vector<CloudPnPtr>& tip_faces) {
        const CloudPnPtr face_tmp (new CloudPn);
        pcl::PointNormal point_tmp;
        point_tmp.x = center.x;
        point_tmp.y = center.y;
        point_tmp.z = center.z;
        point_tmp.normal_x = center.normal_x;
        point_tmp.normal_y = center.normal_y;
        point_tmp.normal_z = center.normal_z;
        point_tmp.curvature = center.xx; // curvature is used to store radius
        face_tmp->points.push_back(point_tmp);
        tip_faces.push_back(face_tmp);
    }

#ifdef BUILD_WITH_VISUALIZER
    template <typename PointT>
    void Workspace<PointT>::GetCircleFace(const PointT& center, std::vector<CloudPnPtr>& tip_faces, std::vector<CloudPnPtr>& tip_real_faces) {
        GetCircleFace(center, tip_faces);

        float radius = center.xx;
        Eigen::Vector3f x_axis, y_axis, tmp;
        tmp << 1, 0, 0;
        x_axis = center.getNormalVector3fMap().cross(tmp);
        x_axis.normalize();
        y_axis = center.getNormalVector3fMap().cross(x_axis);
        y_axis.normalize();

        pcl::PointNormal point_tmp;
        const CloudPnPtr real_face_tmp (new CloudPn);
        for (int i = 0; i < 360; i += 45) {
            point_tmp.x = center.x + radius * std::cos(i * M_PI / 180) * x_axis[0] + radius * std::sin(i * M_PI / 180) * y_axis[0];
            point_tmp.y = center.y + radius * std::cos(i * M_PI / 180) * x_axis[1] + radius * std::sin(i * M_PI / 180) * y_axis[1];
            point_tmp.z = center.z + radius * std::cos(i * M_PI / 180) * x_axis[2] + radius * std::sin(i * M_PI / 180) * y_axis[2];
            real_face_tmp->points.push_back(point_tmp);
        }
        tip_real_faces.push_back(real_face_tmp);
    }
#endif

// Get index of nearest point in finger cloud to the point in object cloud
// with distance and normal difference inside thresholds
    template <typename PointT>
    int Workspace<PointT>::GetIdx(
            const TreePtr tip_tree, const CloudPtr tip, const CloudPnPtr& obj, std::vector<CloudPnPtr>& obj_points,
            const float dist_threshold, const float norm_threshold) {
        float dist;
        const CloudPnPtr obj_point (new CloudPn);
        for (const auto& point : obj->points) {
            int idx = ObtainNearest<PointT, pcl::PointNormal>(tip_tree, &point, dist);
            if (dist < dist_threshold &&
                point.getNormalVector3fMap().dot(tip->points[idx].getNormalVector3fMap()) > norm_threshold &&
                // In grasp mode, choose points inside object surface (<0); In dataset mode, choose points outside object surface (>0)
                mode_ * (tip->points[idx].getVector3fMap() - point.getVector3fMap()).dot(point.getNormalVector3fMap()) < 0) {
                obj_point->points.push_back(point);
                obj_points.push_back(obj_point);
                return idx;
            }
        }
        obj_points.push_back(obj_point);
        return -1;
    }

// Filter object points based on the interval of fingertip
    template <typename PointT>
    void Workspace<PointT>::IntervalFilter(
            const int finger_id, const int point_id, const V3d& finger_normal, const CloudPnPtr& obj) {
        PointT point = tips_original_[finger_id]->points[point_id];
        if (finger_normal[2] != 0) {
            float delta = tip_interval_ * std::fabs(point.normal_z) /
                          std::sqrt(point.normal_x * point.normal_x + point.normal_z * point.normal_z);
            PassWithNormal<pcl::PointNormal>(obj, obj, point.x - delta, point.x + delta, "x",
                                             finger_normal, 0);
        } else {
            PassWithNormal<pcl::PointNormal>(obj, obj, point.z - tip_interval_, point.z + tip_interval_, "z",
                                             finger_normal, 0);
        }
    }

    template <typename PointT>
    bool Workspace<PointT>::GetTipFaces(
            const int minor_point_id, const V3d& finger_normal, const CloudPnPtr& obj, V3d* finger_normals,
            std::vector<int>& point_indices, std::vector<CloudPnPtr>& tip_faces, std::vector<CloudPnPtr>& obj_points) {
        // Filter object points based on known minor_tip_id_ to avoid collision between fingertips
        IntervalFilter(minor_tip_id_, minor_point_id, finger_normal, obj);
#ifdef BUILD_WITH_VISUALIZER
        CloudXyzPtr first_tmp (new CloudXyz);
        pcl::copyPointCloud(*obj, *first_tmp);
        passed_obj_.clear();
        real_tip_faces_.clear();
#endif

        tip_faces.clear();
        point_indices.clear();
        std::vector<float> fin_x_positions;
        const CloudPnPtr empty_face (new CloudPn);

        // last finger is not required to perform intervalFilter, so start from 2
        int filtered_fin_num = 2;
        for (int finger_id = 0; finger_id < finger_num_; finger_id++) {
            if (!valid_finger_list_[finger_id] || required_finger_list_[finger_id] == 0) {
                point_indices.push_back(-1); // -1 means not valid
                tip_faces.push_back(empty_face);
                fin_x_positions.push_back(0.0);
#ifdef BUILD_WITH_VISUALIZER
                real_tip_faces_.push_back(empty_face);
#endif
                continue;
            }
            if (finger_id != minor_tip_id_) {
                int point_id = GetIdx(tips_trees_[finger_id], tips_original_[finger_id], obj, obj_points);
                // required finger not satisfied
                if (point_id == -1) {
                    return false;
                }
                point_indices.push_back(point_id);
                PointT point = tips_original_[finger_id]->points[point_id];
                finger_normals[finger_id] << point.normal_x, point.normal_y, point.normal_z;
            } else { // if (i == minor_tip_id_)
                point_indices.push_back(minor_point_id);
                finger_normals[finger_id] = finger_normal;
                PointT point = tips_original_[finger_id]->points[minor_point_id];
#ifdef BUILD_WITH_VISUALIZER
                passed_obj_.push_back(first_tmp);
#endif
            }
            fin_x_positions.push_back(tips_original_[finger_id]->points[point_indices.back()].x);
            // Get face of fingertip
            if (face_type_ == 2) { // rectangle
                GetRectangleFace(tips_original_[finger_id]->points[point_indices.back()], tip_faces);
#ifdef BUILD_WITH_VISUALIZER
                real_tip_faces_ = tip_faces;
#endif
            } else { // circle or point
#ifndef BUILD_WITH_VISUALIZER
                GetCircleFace(tips_original_[finger_id]->points[point_indices.back()], tip_faces);
#else
                GetCircleFace(tips_original_[finger_id]->points[point_indices.back()], tip_faces, real_tip_faces_);
#endif
            }
            // Filter object points if not the last finger
            if (finger_id != minor_tip_id_ && filtered_fin_num != required_finger_num_) {
                IntervalFilter(finger_id, point_indices.back(), finger_normals[finger_id], obj);
                filtered_fin_num++;

#ifdef BUILD_WITH_VISUALIZER
                CloudXyzPtr tmp (new CloudXyz);
                pcl::copyPointCloud(*obj, *tmp);
                passed_obj_.push_back(tmp);
#endif
            }
        }

        // Check finger crossing
        for (const auto& descent_indices : {ws_centroid_x_descent_indices_primary_,
                                            ws_centroid_x_descent_indices_opposite_ }) {
            for (size_t k = 1; k < descent_indices.size(); ++k) {
                const int idx0 = descent_indices[k - 1];
                const int idx1 = descent_indices[k];
                if (required_finger_list_[idx0] == 0 || required_finger_list_[idx1] == 0) continue;
                if (fin_x_positions[idx0] < fin_x_positions[idx1]) return false;
            }
        }
        return true;
    }

    template <typename PointT>
    void Workspace<PointT>::GetOverlap(
            const CloudPnPtr& tip_face, const CloudPnPtr& tar, const CloudPnPtr& contacts, const float dist_threshold) {
        const pcl::PointNormal& point = tip_face->points[0];
        const V3f base = point.getVector3fMap();
        const V3f normal = point.getNormalVector3fMap();

        for (const auto& tar_point : tar->points) {
            if (std::fabs(normal.dot(tar_point.getVector3fMap() - base)) < dist_threshold)
                contacts->points.push_back(tar_point);
        }
    }

// Get contact points of all fingers
    template <typename PointT>
    bool Workspace<PointT>::GetContacts(
            const std::vector<CloudPnPtr>& tip_faces, const std::vector<CloudPnPtr>& tar2tip,
            const CloudPnPtr& contacts)
    {
#ifdef BUILD_WITH_VISUALIZER
        other_contacts_.clear();
#endif
        for (int finger_idx = 0; finger_idx < finger_num_; finger_idx++) {
            CloudPnPtr single_contact(new CloudPn);
            if (!valid_finger_list_[finger_idx]) {
#ifdef BUILD_WITH_VISUALIZER
                other_contacts_.push_back(single_contact);
#endif
                continue;
            }
            if (required_finger_list_[finger_idx] == 1) {
                const std::size_t init_size = contacts->size();
                GetOverlap(tip_faces[finger_idx], tar2tip[finger_idx], single_contact);

#ifdef BUILD_WITH_VISUALIZER
                if (finger_idx == minor_tip_id_) {
                    CloudPnPtr tmp(new CloudPn);
                    copyPointCloud(*single_contact, *tmp);
                    first_contacts_ = tmp;
                } else {
                    other_contacts_.push_back(single_contact);
                }
#endif

                *contacts += *single_contact;
                if (contacts->size() == init_size) {
                    // If no contact points found for the required finger, clear all contacts and return
                    contacts->clear();
                    return false;
                }
            } else {
#ifdef BUILD_WITH_VISUALIZER
                other_contacts_.push_back(single_contact);
#endif
            }
        }
        return true;
    }

    template <typename PointT>
    bool Workspace<PointT>::CheckPrimaryFingerPos(const std::vector<int>& tip_point_indices) {
        int max_id = -1;
        for (int i = 0; i < finger_num_; i++) {
            if (tip_point_indices[i] != -1) max_id = i;
        }
        int min_id = 1;
        for (int i = finger_num_ - 1; i > 0; i--) {
            if (tip_point_indices[i] != -1) min_id = i;
        }

        const V3f primary_fin_pos = tips_original_[0]->points[tip_point_indices[0]].getVector3fMap();
        const V3f left_fin_pos = tips_original_[min_id]->points[tip_point_indices[min_id]].getVector3fMap();
        // rough antipodal checking
        if (max_id == 1) {
            const V3f primary_nor = -tips_original_[0]->points[tip_point_indices[0]].getNormalVector3fMap();
            const V3f left_nor = tips_original_[1]->points[tip_point_indices[1]].getNormalVector3fMap();
            const V3f primary2left = (left_fin_pos - primary_fin_pos).normalized();
            const float angle_sum = std::acos(primary2left.dot(primary_nor)) + std::acos(primary2left.dot(left_nor));
            return angle_sum < M_PI / 10;
        }

        // check if the primary finger normal is pointing inside the valid range
        const V3f right_fin_pos = tips_original_[max_id]->points[tip_point_indices[max_id]].getVector3fMap();
        V3f left = (left_fin_pos - primary_fin_pos);
        // project to x-o-z plane
        left[1] = 0;
        left = left.normalized();
        V3f right = right_fin_pos - primary_fin_pos;
        right[1] = 0;
        right = right.normalized();
        V3f nor = -tips_original_[0]->points[tip_point_indices[0]].getNormalVector3fMap();
        nor[1] = 0;
        nor = nor.normalized();
        const double angle = std::acos(left.dot(nor)) + std::acos(right.dot(nor));
        const double range = std::acos(left.dot(right));
        return std::abs(range - angle) < 1e-4;
    }

}  // namespace fsg
