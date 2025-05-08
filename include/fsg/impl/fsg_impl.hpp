#pragma once

namespace fsg {

    template <typename PointT>
    FSG<PointT>::FSG(const int angle_num, const int finger_num)
            : period_{0}, t_start_{}, t_end_{} {
        // Number of fingers of the robot hand
        finger_num_ = finger_num;
        // Maximum number of angles for each finger
        angle_num_ = angle_num;

        // standalone for grasp pose generation, without considering the robot arm/base
        nh_.param("standalone", standalone_, false);

        nh_.param("outer_iter", outer_iteration_, 20);
        nh_.param("inner_iter", inner_iteration_, 15);
        nh_.param("thread_num", thread_num_,
                  static_cast<int>(std::thread::hardware_concurrency())); // C++11

        nh_.param("with_objPrepro", with_obj_prepro_, true);
        nh_.param("with_palmFilter", with_palm_filter_, true);
        nh_.param("with_finIter", with_fin_iteration_, false);
        nh_.param("with_linkColFilter", with_link_col_filter_, false);

        nh_.param("length", max_num_output_, 100);
        nh_.param("gws_thre", gws_threshold_, 0.01);
        nh_.param("palm_h_thre", palm_h_threshold_, 0.05);

        nh_.param<std::string>("tip_prefix", tip_prefix_, "tip_");
        nh_.param<std::string>("con_prefix", contact_prefix_, "con_");
        nh_.param<std::string>("aff_ext", aff_ext_, "_aff0");
        nh_.param<std::string>("robot_base_frame", robot_base_frame_, "base_link");

        if (with_palm_filter_) {
            // Orientation of the robot base frame
            std::vector<float> forward_tmp = {1.0, 0.0, 0.0};
            if (!nh_.getParam("base_forward", forward_tmp)) {
                ROS_WARN("[FSG] base_forward invalid, using 1.0 0.0 0.0");
            }
            base_forward_ = V3d(forward_tmp[0], forward_tmp[1], forward_tmp[2]);

            std::vector<float> up_tmp = {0.0, 0.0, 1.0};
            if (!nh_.getParam("base_upward", up_tmp)) {
                ROS_WARN("[FSG] base_upward invalid, using 0.0 0.0 1.0");
            }
            base_upward_ = V3d(up_tmp[0], up_tmp[1], up_tmp[2]);
        }

        GetWorkspace();

#ifdef BUILD_WITH_VISUALIZER
        thread_num_ = 1;
#endif

        curv_filter_srv_ = nh_.serviceClient<pcl_interface::curvFilter>("curv_filter");
        ROS_WARN_STREAM("\n [FSG] Server on" <<
                                                  "\n    stand_alone:     " << standalone_ <<
                                                  "\n    finger_num:      " << finger_num_ <<
                                                  "\n    angle_num:       " << angle_num_ <<
                                                  "\n    outer_iter:      " << outer_iteration_ <<
                                                  "\n    inner_iter:      " << inner_iteration_ <<
                                                  "\n    thread_num:      " << thread_num_ <<
                                                  "\n    gws_threshold:   " << gws_threshold_ <<
                                                  "\n    max_num_output:  " << max_num_output_ <<
                                                  "\n    palm_h_threshold " << palm_h_threshold_ <<std::boolalpha <<
                                                  "\n    with_objPrepro:  " << with_obj_prepro_ <<
                                                  "\n    with_palmFilter: " << with_palm_filter_ <<
                                                  "\n    with_finIter:    " << with_fin_iteration_);
    }

    template <typename PointT>
    void FSG<PointT>::GetWorkspace() {
        // get workspace point cloud file paths from rosparam
        std::vector<std::string> workspace_pc_paths;
        if (!nh_.getParam("ws_pc_paths", workspace_pc_paths)) {
            ROS_ERROR("[FSG] ws_pc_paths not found");
            exit(1);
        }

        // get configuration on palm and tip
        std::vector<float> palm_min_max;
        double tip_interval, fin_collision_thre;
        nh_.getParam("tip_interval", tip_interval);
        nh_.param("fin_collision_thre", fin_collision_thre, 0.005);
        nh_.getParam("palm_min_max", palm_min_max);

        // Initialize workspace
        face_type_ = ws_.InitWorkspace(workspace_pc_paths, finger_num_, tip_interval, fin_collision_thre, palm_min_max);
        const std::vector<std::string> tip_types = {"point", "circle", "rectangle"};
        ROS_WARN_STREAM("[FSG] Tip type: " << tip_types.at(face_type_));
        if (face_type_ == 0) {
            least_finger_num_ = 3;
        } else {
            least_finger_num_ = 2;
        }
        ROS_WARN_STREAM("[FSG] Least finger number: " << least_finger_num_);

        int mode;
        nh_.param("mode", mode, 1);
        if (mode == 1) ROS_WARN("[FSG] Mode: grasp");
        else if (mode == -1) ROS_WARN("[FSG] Mode: dataset");
        else {
            ROS_ERROR("[FSG] Invalid mode, using [default] 1 - grasp or -1 - dataset");
            mode = 1;
        }
        ws_.SetMode(mode);

        if (with_link_col_filter_) {
            std::vector<std::string> link_col_pc_paths;
            if (!nh_.getParam("link_pc_paths", link_col_pc_paths)) {
                ROS_ERROR("[FSG] link_pc_paths not found");
                exit(1);
            }
            ws_.GetLinkCol(link_col_pc_paths);
        }
    }

    template <typename PointT>
    void FSG<PointT>::ObjPcCallback(const sensor_msgs::PointCloud2ConstPtr& input) {
        // Get object point cloud
        obj_pc_subscriber_.shutdown();
        const CloudPnPtr pc_tmp (new CloudPn);
        PcMsg2Pcl<pcl::PointNormal>(input, pc_tmp);
        obj_.SetWhole(pc_tmp);
        obj_pc_obtained_ = true;
        // Get object centroid and max radius
        obj_.GetWholeInfo();

        if (!with_aff_) {
            if (with_obj_prepro_) {
                // preprocess object point cloud
                ObjPcPrepro(input);
            } else {
                obj_.SetPart(pc_tmp);
            }
            obj_.ShufflePart();
            obj_part_pc_obtained_ = true;
        }
    }

    template <typename PointT>
    void FSG<PointT>::ObjPartPcCallback(const sensor_msgs::PointCloud2ConstPtr& input) {
        // Get object partial point cloud (e.g. object part with affordance)
        obj_part_pc_subscriber_.shutdown();
        const CloudPnPtr pc_tmp (new CloudPn);
        PcMsg2Pcl<pcl::PointNormal>(input, pc_tmp);
        obj_.SetPart(pc_tmp);
        obj_.ShufflePart();
        obj_part_pc_obtained_ = true;
    }

    template <typename PointT>
    void FSG<PointT>::ObjPcPrepro(const sensor_msgs::PointCloud2ConstPtr& input) {
        // preprocess object point cloud based on curvature
        pcl_interface::curvFilter srv;
        srv.request.input_pcd = *input;
        double threshold;
        nh_.param("curv_filter_threshold", threshold, 0.7);
        srv.request.threshold = float(threshold);
        srv.request.input_pc_topic = obj_pc_topic_;
        if (curv_filter_srv_.call(srv)) {
            const CloudPnPtr obj_pc_tmp (new CloudPn);
            PcMsg2Pcl<pcl::PointNormal>(boost::make_shared<sensor_msgs::PointCloud2>(srv.response.output_pcd),
                                        obj_pc_tmp);
            obj_.SetPart(obj_pc_tmp);
        } else {
            ROS_ERROR("[FSG] Failed to call service curv_filter, directly using original object cloud");
            obj_.SetPart(obj_.GetWhole());
        }
    }

    template <typename PointT>
    bool FSG<PointT>::GetObjBottomPose() {
        if (standalone_) {
            // If standalone, object pose is set to identity
            obj_pose_ = A3d::Identity();
            return true;
        }
        try {
            // Get object bottom pose under robot base frame from tf
            tf::StampedTransform transform;
            tf_listener_.waitForTransform(robot_base_frame_, obj_bottom_frame_,
                                          ros::Time(0), ros::Duration(2.0));
            tf_listener_.lookupTransform(robot_base_frame_, obj_bottom_frame_,
                                         ros::Time(0), transform);
            transformTFToEigen(transform, obj_pose_);
        } catch (tf::TransformException &ex) {
            ROS_ERROR("[FSG] failed to get obj_bot pose under base_link. %s",ex.what());
            return false;
        }
        return true;
    }

    template <typename PointT>
    bool FSG<PointT>::FSGCallback(const GraspPose::Request&  req,
                                            GraspPose::Response& res) {
        gettimeofday(&t_start_, nullptr);
        obj_pc_topic_ = req.obj_pc_topic;
        obj_bottom_frame_ = req.obj_bottom_frame;
        with_aff_ = req.with_aff;

        obj_pc_obtained_ = false;
        obj_part_pc_obtained_ = false;
        obj_pc_subscriber_ = nh_.subscribe(obj_pc_topic_, 1, &FSG::ObjPcCallback, this);
        if (with_aff_) {
            obj_part_pc_topic_ = obj_pc_topic_ + aff_ext_;
            obj_part_pc_subscriber_ = nh_.subscribe(obj_part_pc_topic_, 1, &FSG::ObjPartPcCallback, this);
        }

        // Get object bottom pose under robot base frame, object point cloud and partial point cloud
        if (!GetObjBottomPose()) { return false; }
        while (!obj_pc_obtained_ || !obj_part_pc_obtained_) { ros::spinOnce(); }

        // Set finger number and list based on object max length
        ws_.SetMaxFinList(obj_.GetObjMaxR() * 2);
        // Update kdtree
        ws_.UpdateTree();
        // get fingertip with the least number of points in its cloud
        ws_.GetMinorTip();
        if (standalone_) {
            // All points inside partial cloud can be candidates for the minor finger
            obj_.SetPartMinor(obj_.GetPart());
        } else {
            // Get points inside partial cloud for minor finger
            obj_.GetCloud4Minor(ws_.GetMinorNormal());
        }

        // body function
        Estimate();
        int valid_pose_num = 0;
        int size = candidates_list_.size();
        valid_pose_num = CandidateFilter(size, valid_pose_num);
        if (with_fin_iteration_) {
            while (valid_pose_num < 10) {
                ROS_WARN_STREAM("[FSG] Found " << valid_pose_num << " candidates with " << ws_.GetRequiredFingerNum() << " fingers ... Reduce Finger");
                if (ws_.GetRequiredFingerNum() == least_finger_num_) break;
                candidates_list_.clear();
                ws_.ReduceFinger();
                ws_.GetMinorTip();
                if (!standalone_) {
                    obj_.GetCloud4Minor(ws_.GetMinorNormal());
                }
                Estimate();
                size = candidates_list_.size();
                valid_pose_num = CandidateFilter(size, valid_pose_num);
            }
        }

        gettimeofday(&t_end_, nullptr);
        period_ = t_end_.tv_sec - t_start_.tv_sec + (t_end_.tv_usec - t_start_.tv_usec) / 1000000.0;
        ROS_WARN_STREAM("[FSG] Total time: " << period_);

        // Return with time cost, number of results, palm poses, gws scores and hand joint angles
        res.time = period_;
        res.num = palm_poses_.size();
        for (int i = 0; i < palm_poses_.size(); i++) {
            res.poses.poses.push_back(palm_poses_[i]);
            res.scores.push_back(gws_scores_[i]);
        }
        for (int i = 0; i < joint_angle_lists_.size(); i++) {
            std_msgs::MultiArrayDimension mad;
            mad.label = std::to_string(i);
            mad.size = angle_num_ * finger_num_;
            mad.stride = angle_num_ * finger_num_;
            res.angles.layout.dim.push_back(mad);
            res.angles.data.insert(res.angles.data.end(),
                                   joint_angle_lists_[i].begin(), joint_angle_lists_[i].end());
        }

        gws_scores_.clear();
        palm_poses_.clear();
        candidates_list_.clear();
        joint_angle_lists_.clear();
        return true;
    }

    template <typename PointT>
    int FSG<PointT>::CandidateFilter(const int size, int valid_pose_num) {
        if (size == 0) {
            ROS_WARN("[FSG] No valid grasp pose found");
            return valid_pose_num;
        }
        int init_pose_num = valid_pose_num;

        A3d palm_pose;
        geometry_msgs::Pose palm_pose_msg;
        for (const auto& candidate : candidates_list_) {
            // Filtering for single-joint parallel grippers
            if (candidate->angles.size() == 2 && abs(candidate->angles[0] - candidate->angles[1]) > 15) {
                continue;
            }
            palm_pose = obj_pose_ * candidate->obj_transform.inverse();
            if (with_palm_filter_) {
                // filtering based on palm pose
                V3d palm_z_axis = palm_pose.linear().col(2);
                // cos(130deg)
                if (palm_z_axis.dot(base_upward_) < -0.64) continue;
                // hand pointing forward cos(120deg)
                if (palm_z_axis.dot(base_forward_) < -0.5) continue;

                // palm roll
                // cos(135deg)
                V3d palm_x_axis = palm_pose.linear().col(0);
                if (palm_x_axis.dot(base_upward_) < -0.7) continue;
                V3d palm_y_axis = palm_pose.linear().col(1);
                if (palm_y_axis.dot(base_upward_) < -0.7) continue;
            }

            CloudPnPtr result (new CloudPn);
            transformPointCloudWithNormals(*(obj_.GetWhole()), *result, candidate->obj_transform);
            std::vector<float> joint_angles(candidate->angles.begin(), candidate->angles.end());
            if (with_link_col_filter_ && ws_.CheckLinkCollision(result, joint_angles)) continue;

            PoseEigen2Msg(palm_pose, palm_pose_msg);
            palm_poses_.push_back(palm_pose_msg);
            gws_scores_.push_back(candidate->gws_score);

            ROS_INFO_STREAM("[FSG] No. " << std::setw(2) << valid_pose_num << " - gws_score: " << candidate->gws_score);
            candidate->tips->width = candidate->tips->size();
            candidate->tips->height = 1;
            candidate->contacts->width = candidate->contacts->size();
            candidate->contacts->height = 1;
            writer_.write<pcl::PointNormal>("final_" + std::to_string(valid_pose_num) + ".pcd", *result);
            writer_.write<pcl::PointNormal>(tip_prefix_ + std::to_string(valid_pose_num) + ".pcd", *(candidate->tips));
            writer_.write<pcl::PointNormal>(contact_prefix_ + std::to_string(valid_pose_num) + ".pcd", *(candidate->contacts));

            joint_angle_lists_.push_back(joint_angles);
            valid_pose_num++;
        }
        ROS_WARN_STREAM("[FSG] Filter grasp poses, size: " << valid_pose_num - init_pose_num << " out of " << size);
        return valid_pose_num;
    }

    template <typename PointT>
    void FSG<PointT>::Estimate() {
        omp_set_num_threads(thread_num_);
#pragma omp parallel default(none)
        {
            int idx_tip;
            CloudPnPtr obj_transformed (new CloudPn);
            CloudPnPtr full_obj_transformed (new CloudPn);
            CloudPnPtr obj_transformed_tmp (new CloudPn);
            A3d translate = A3d::Identity(), rotation, transform;
            V3d finger_normal, obj_normal;
            std::vector<CloudPnPtr> tip_faces, obj2tip;

#pragma omp parallel for
            for (int o = 0; o < outer_iteration_; ++o) {
                MatchPointPair(idx_tip, finger_normal, obj_normal, transform, translate);
                for (int i = 0; i < inner_iteration_; ++i) {
                    std::vector<CloudPnPtr> obj_points;
                    V3d tip_face_normals[finger_num_];
                    std::vector<int> tip_point_indices;
                    // rotate around self axis
                    rotation = Eigen::AngleAxisd(2 * M_PI * i / inner_iteration_, obj_normal);
                    A3d obj_transform;
                    obj_transform = transform * rotation * translate;
                    transformPointCloudWithNormals(*(obj_.GetPart()), *obj_transformed, obj_transform);
                    transformPointCloudWithNormals(*(obj_.GetWhole()), *full_obj_transformed, obj_transform);

                    V3d obj_y_axis;
                    obj_y_axis = obj_transform.linear().col(1);
                    V3d obj_origin;
                    obj_origin = obj_transform.translation();
                    if (ws_.CheckHandCollision(obj_origin, obj_y_axis, full_obj_transformed, palm_h_threshold_)) continue;
                    copyPointCloud(*obj_transformed, *obj_transformed_tmp);
                    if (!ws_.GetTipFaces(idx_tip, finger_normal, obj_transformed_tmp, tip_face_normals,
                                         tip_point_indices, tip_faces, obj_points)) {
                        continue;
                    }
                    if (!standalone_ && ws_.CheckFinCollision(obj_origin, obj_y_axis, tip_faces)) continue;
                    // Get target face inside the area of contact face
                    obj2tip.clear();
                    for (int j = 0; j < finger_num_; ++j) {
                        CloudPnPtr left_part (new CloudPn);
                        if (!tip_faces[j]->empty()) {
                            if (face_type_ == 2) {
                                Object::GetRectangleContactCandidates(tip_faces[j], tip_face_normals[j],
                                                                      obj_transformed, left_part);
                            } else if (face_type_ == 1) {
                                Object::GetCircleContactCandidates(tip_faces[j], tip_faces[j]->points[0].curvature,
                                                                   obj_transformed, left_part);
                            }
                        }
                        obj2tip.push_back(left_part);
                    }
                    // get contacts
                    CloudPnPtr contacts (new CloudPn);
                    if (face_type_ == 2 || face_type_ == 1) {
                        if (!ws_.GetContacts(tip_faces, obj2tip, contacts)) continue;
                    } else { // point
                        for (const auto& obj_point : obj_points) {
                            *contacts += *obj_point;
                        }
                    }
                    if (contacts->size() < 3) continue;
                    // check if primary finger contact position is valid
                    if (!ws_.CheckPrimaryFingerPos(tip_point_indices)) continue;
#ifdef BUILD_WITH_VISUALIZER
                    std::vector<float> joint_angles;
                    for (int fin_id = 0; fin_id < finger_num_; fin_id++) {
                        if (tip_point_indices[fin_id] == -1) {
                            joint_angles.insert(joint_angles.end(), angle_num_, 0);
                        } else {
                            for (int j = 0; j < angle_num_; j++) {
                                joint_angles.push_back(ws_.GetTipPointAngle(fin_id, tip_point_indices[fin_id], j));
                            }
                        }
                    }

                    if (ws_.CheckLinkCollision(full_obj_transformed, joint_angles)) continue;

                    std::vector<CloudPnPtr> real_contacts;
                    real_contacts.push_back(ws_.GetFirstContacts());

                    CloudPnPtr first_contact (new CloudPn);
                    *first_contact += *ws_.GetFirstContacts();
                    for (auto& contact : ws_.GetOtherContacts()) {
                        real_contacts.push_back(contact);
                    }

                    VisProcedure(tip_point_indices, ws_.GetRealTipFaces(), obj_transformed, obj_transform, real_contacts, ws_.GetLinkSkeletons());
#endif

                    auto candidate = std::make_shared<GraspCandidate>();
                    candidate->gws_score = GwsScore(contacts, obj_transform);
#pragma omp critical
                    {
                        PushToList(candidate, contacts, tip_faces, obj_transform, tip_point_indices);
                    }
                }
            }
        }
    }

    template <typename PointT>
    void FSG<PointT>::MatchPointPair(int& idx_tip, V3d& finger_normal, V3d& obj_normal,
                                          A3d& transform, A3d& translate) {
        // Randomly select a point on the object and minor fingertip cloud,
        thread_local std::random_device rd;
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        int obj_id = rd() % obj_.GetPartMinorSize();
        idx_tip = rd() % ws_.GetMinorTipSize();
#ifdef BUILD_WITH_VISUALIZER
        obj_point_id_ = obj_id;
#endif

        const float* tarPtr_ = obj_.GetPartMinorPointXPtr(obj_id);
        const float* finPtr_ = ws_.GetMinorTipPointXPtr(idx_tip);
        obj_normal << *(tarPtr_ + 4), *(tarPtr_ + 5), *(tarPtr_ + 6);
        finger_normal << *(finPtr_ + 4), *(finPtr_ + 5), *(finPtr_ + 6);
        // Move point on obj to origin
        translate.translation() << -*tarPtr_, -*(tarPtr_ + 1), -*(tarPtr_ + 2);

        // Calculate the transformation matrix to have two points coincide in position and normal
        V3d tip_origin(*(finPtr_), *(finPtr_ + 1), *(finPtr_ + 2));
        if (face_type_ == 2) { // rectangle
            const V3d tip_x_axis(*(finPtr_ + 9), *(finPtr_ + 10), *(finPtr_ + 11));
            const V3d tip_z_axis(*(finPtr_ + 12), *(finPtr_ + 13), *(finPtr_ + 14));
            tip_origin += dist(rd) * tip_x_axis + dist(rd) * tip_z_axis;
        } else if (face_type_ == 1) { // circle
            V3d rand_tip_x = V3d::Zero();
            const double r = *(finPtr_ +  9) * dist(rd);
            rand_tip_x << dist(rd), dist(rd), dist(rd);
            rand_tip_x.normalize();
            V3d rand_tip_y = rand_tip_x.cross(finger_normal);
            while (rand_tip_y.norm() < 0.01) {
                rand_tip_x << dist(rd), dist(rd), dist(rd);
                rand_tip_x.normalize();
                rand_tip_y = rand_tip_x.cross(finger_normal);
            }
            tip_origin += r * rand_tip_y;
        }
        transform = CoincideMatrix(tip_origin, finger_normal, obj_normal);
    }

// Add found grasp candidate to the list
    template <typename PointT>
    void FSG<PointT>::PushToList(
            const GraspCandidatePtr& candidate,
            const CloudPnPtr& contacts,
            const std::vector<CloudPnPtr>& tip_faces,
            const A3d& trans,
            const std::vector<int>& tip_point_indices) {
        const float worstScore = candidates_list_.size() < max_num_output_ ? 0 : candidates_list_.back()->gws_score;
        if (candidate->gws_score > gws_threshold_ && candidate->gws_score > worstScore) {
            // check if similar solution exists
            for (const auto& cand : candidates_list_) {
                Eigen::Matrix4d diff = cand->obj_transform.matrix() - trans.matrix();
                if (!(diff.cwiseAbs().array() > 0.005).any()) {
                    return;
                }
            }
            // add tip contact points
            candidate->contacts = contacts;
            const CloudPnPtr combined_tips (new CloudPn);
            for (const auto& tip : tip_faces) {
                *combined_tips += *tip;
            }
            candidate->tips = combined_tips;
            candidate->obj_transform = trans;
            // add finger joint angles
            for (int i = 0; i < finger_num_; i++) {
                if (tip_point_indices[i] == -1) {
                    candidate->angles.insert(candidate->angles.end(), angle_num_, 0);
                } else {
                    for (int j = 0; j < angle_num_; j++) {
                        candidate->angles.push_back(ws_.GetTipPointAngle(i, tip_point_indices[i], j));
                    }
                }
            }
            candidates_list_.push_back(candidate);
            sort(candidates_list_.begin(), candidates_list_.end(), CompareGws);
            if (candidates_list_.size() > max_num_output_) {
                candidates_list_.resize(max_num_output_);
            }
        }
    }

    template <typename PointT>
    double FSG<PointT>::GwsScore(const CloudPnPtr& contacts, const A3d& obj_transform) {
        A3d obj_centroid = A3d::Identity();
        obj_centroid.translation() = obj_.GetObjCentroid();

        A3d transformed_obj_centroid = obj_transform * obj_centroid;

        // current centroid
        // x,y,z axes of the transformed frame at centroid
        const V3d centroid_pos = transformed_obj_centroid.translation();
        const V3d x_base = transformed_obj_centroid.linear().col(0);
        const V3d y_base = transformed_obj_centroid.linear().col(1);
        const V3d z_base = transformed_obj_centroid.linear().col(2);

        const std::size_t num = contacts->size();
        Eigen::MatrixXd wrench(6, num * kCircleDiv);
        Eigen::MatrixXd wr(6, kCircleDiv);
        Eigen::MatrixXd result(6, kCircleDiv);
        V3d transformed_normal, translate_local;

        int i = 0;
        const float obj_max_r = obj_.GetObjMaxR();
        for (const auto& point : contacts->points) {
            // create unit wrench towards +z under contact frame
            for (int k = 0; k < kCircleDiv; k++) {
                const double angle = 2 * M_PI * k / kCircleDiv;
                wr(0, k) = cos(angle) * kMu;
                wr(1, k) = sin(angle) * kMu;
                wr(2, k) = 1;
                wr(3, k) = 0;
                wr(4, k) = 0;
                wr(5, k) = 0;
            }
            // normal of contact point under original frame
            V3d n_original = point.getNormalVector3fMap().cast<double>();
            // normal of contact point under transformed frame
            transformed_normal << n_original.dot(x_base), n_original.dot(y_base), n_original.dot(z_base);
            // translation from current centroid to contact point under original frame
            V3d translate_world = point.getVector3fMap().cast<double>() - centroid_pos;
            // translation from current centroid to contact point under transformed object frame
            translate_local << translate_world.dot(x_base), translate_world.dot(y_base), translate_world.dot(z_base);

            // ContactMap to transform unit wrench from contact frame to object frame
            Eigen::MatrixXd map = ContactMap(transformed_normal, translate_local);
            result = map * wr;

            // store results of one unit wrench
            for (int k = 0; k < kCircleDiv; k++) {
                const int idx = i * kCircleDiv + k;
                wrench(0, idx) = result(0, k);
                wrench(1, idx) = result(1, k);
                wrench(2, idx) = result(2, k);
                wrench(3, idx) = result(3, k) / obj_max_r;  // max distance from surface to centroid
                wrench(4, idx) = result(4, k) / obj_max_r;
                wrench(5, idx) = result(5, k) / obj_max_r;
            }
            i++;
        }
        L1GWS l1gws;
        return l1gws.build(wrench, ALL_DIMENSIONS);
    }

#ifdef BUILD_WITH_VISUALIZER
    template <typename PointT>
    int FSG<PointT>::VisOneFinger(
            Visualizer& vis, const std::vector<int>& tip_point_indices, const std::vector<CloudPnPtr>& tip_faces,
            const std::vector<CloudPnPtr>& contacts, const CloudPnPtr& obj_transformed,
            int id, int fin_id_0, int timer, int time) {
        int tip_size = 10;
        CloudXyzPtr tmp(new CloudXyz);
        // workspace cloud
        if (timer == time) {
            if (id == ws_.GetMinorTipId()) {
                vis.getViewer()->updateText("First finger with fewest points", 100, 980, "caption");
            } else {
                vis.getViewer()->updateText("Next finger", 100, 980, "caption");
            }
            vis.ChangeSize("ws" + std::to_string(id), 5);
            vis.ChangeOpacity("ws" + std::to_string(id), 1);
        }
        time += 100;

        // found point
        if (timer == time) {
            vis.AddSphere(pcl::PointXYZ(ws_.GetTips()[id]->points[tip_point_indices[id]].x,
                                        ws_.GetTips()[id]->points[tip_point_indices[id]].y,
                                        ws_.GetTips()[id]->points[tip_point_indices[id]].z),
                          0.004, "tip_point", 0.0, 0.0, 0.8);
            if (id == ws_.GetMinorTipId()) {
                vis.getViewer()->updateText("Contact point pair sampling", 100, 980, "caption");
                vis.RemoveShape("obj");
                CloudPnPtr part_minor = obj_.GetPartMinor();
                copyPointCloud(*part_minor, *tmp);
                vis.AddCloud(tmp, "obj", 5, 0.51, 0.27, 0.46);
                vis.AddSphere(pcl::PointXYZ(part_minor->points[obj_point_id_].x,
                                            part_minor->points[obj_point_id_].y,
                                            part_minor->points[obj_point_id_].z),
                              0.004, "obj_point", 0.8, 0.0, 0.4);
            }
            else {
                vis.getViewer()->updateText("Contact point pair searching", 100, 980, "caption");
            }
        }
        time += 100;

        if (id == ws_.GetMinorTipId()) {
            if (timer == time) {
                vis.getViewer()->updateText("Contact point pair matching", 100, 980, "caption");
                vis.RemoveShape("obj");
                vis.RemoveShape("obj_point");
                copyPointCloud(*obj_transformed, *tmp);
                vis.AddCloud(tmp, "obj", 5, 0.51, 0.27, 0.46);
            }
            time += 100;
        }

        // reform contact face
        if (timer == time) {
            vis.ChangeSize("ws" + std::to_string(id), 1);
            vis.ChangeOpacity("ws" + std::to_string(id), 0.5);
            vis.getViewer()->updateText("Get contact face and contact points", 100, 980, "caption");

            copyPointCloud(*(tip_faces[id]), *tmp);
            vis.AddCloud(tmp, "tip_face_" + std::to_string(id), tip_size, 0.99, 0.5, 0);
            vis.RemoveShape("tip_point");

            // show contacts
            copyPointCloud(*contacts[fin_id_0], *tmp);
            vis.AddCloud(tmp, "contacts" + std::to_string(fin_id_0), 15, 0.99, 0.7, 0.9);
        }
        time += 100;

        std::vector<CloudXyzPtr> object_passed;
        object_passed = ws_.GetPassedObj();
        std::sort(object_passed.begin(), object_passed.end(), [](const CloudXyzPtr& a, const CloudXyzPtr& b) {
            return a->size() > b->size(); // Comparison function: larger values come first
        });

        if (fin_id_0 >= object_passed.size()) {
            return time;
        }
        // passthrough filter
        if (timer == time) {
            vis.getViewer()->updateText("Neighbor points filtering", 100, 980, "caption");
            vis.RemoveShape("obj");
            copyPointCloud(*(object_passed[fin_id_0]), *tmp);
            vis.AddCloud(tmp, "obj", 5, 0.51, 0.27, 0.46);
        }
        time += 100;
        return time;
    }

    template <typename PointT>
    void FSG<PointT>::VisProcedure(
            std::vector<int> tip_point_indices, std::vector<CloudPnPtr> tip_faces,
            CloudPnPtr obj_transformed, const A3d& trans, const std::vector<CloudPnPtr>& contacts, const std::vector<CloudPnPtr>& link_col_clouds) {
        CloudXyzPtr tmp(new CloudXyz);
        Visualizer vis(0.0, 0.015, 0.6, 0.1);
        vis.getViewer()->setFullScreen(true);
        vis.getViewer()->addText("Initial setting", 100, 100, 50, 0.0, 0.0, 0.0, "caption");
        double r[5] = {0.99,    0,    0, 0.99, 0.99};
        double g[5] = {   0, 0.99,    0, 0.49, 0.99};
        double b[5] = {   0,    0, 0.99,    0,    0};
        for (int fin = 0; fin < finger_num_; fin++) {
            if (ws_.GetRequiredFingerList()[fin] == 0) continue;
            if (ws_.GetValidFingerList()[fin] == 0) continue;
            copyPointCloud(*(ws_.GetTips()[fin]), *tmp);
            vis.AddCloud(tmp, "ws" + std::to_string(fin), 5, r[fin], g[fin], b[fin]);
        }
        copyPointCloud(*(obj_.GetPart()), *tmp);
        vis.AddCloud(tmp, "obj", 5, 0.51, 0.27, 0.46);

        std::vector<int> fin_id_convert;
        fin_id_convert.clear();
        fin_id_convert.push_back(ws_.GetMinorTipId());
        for (int fin = 0; fin < finger_num_; fin++) {
            if (fin != ws_.GetMinorTipId())
                fin_id_convert.push_back(fin);
        }

        // Update the view in the while loop
        int timer = 0;
        while (!vis.WasStopped()) {
            vis.UpdateCamera();
            if (timer == 100) {
                for (int fin = 0; fin < finger_num_; fin++) {
                    if (ws_.GetRequiredFingerList()[fin] == 0) continue;
                    if (ws_.GetValidFingerList()[fin] == 0) continue;
                    vis.ChangeSize("ws" + std::to_string(fin), 1);
                    vis.ChangeOpacity("ws" + std::to_string(fin), 0.5);
                }
            }
            int time_tmp = 200;
            for (int fin = 0; fin < finger_num_; fin++) {
                if (ws_.GetRequiredFingerList()[fin] == 0) continue;
                if (ws_.GetValidFingerList()[fin] == 0) continue;
                time_tmp = VisOneFinger(vis, tip_point_indices, tip_faces, contacts, obj_transformed, fin_id_convert[fin], fin, timer, time_tmp);
            }

            // whole_ obj
            if (timer == time_tmp) {
                vis.RemoveShape("obj");
                vis.getViewer()->updateText("Final contact points", 100, 980, "caption");
                pcl::transformPointCloudWithNormals(*(obj_.GetWhole()), *obj_transformed, trans);
                copyPointCloud(*obj_transformed, *tmp);
                vis.AddCloud(tmp, "obj", 5, 0.51, 0.27, 0.46);
            }
            time_tmp += 100;

            if (timer == time_tmp) {
                for (int fin = 0; fin < finger_num_; fin++) {
                    if (ws_.GetRequiredFingerList()[fin] == 0) continue;
                    if (ws_.GetValidFingerList()[fin] == 0) continue;
                    copyPointCloud(*(link_col_clouds[fin]), *tmp);
                    vis.getViewer()->updateText("Link penetration checking", 100, 980, "caption");
                    vis.AddCloud(tmp, "fin_link_" + std::to_string(fin), 5, r[fin], g[fin], b[fin]);
                }
            }
            // time_tmp += 100;

            vis.UpdateViewer();
            timer++;
        }
        exit(0);
    }
#endif

}  // namespace fsg