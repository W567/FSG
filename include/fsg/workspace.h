#ifndef FSG_WORKSPACE_H_
#define FSG_WORKSPACE_H_

#include <vector>
#include <string>
#include <Eigen/Core>

#include "fsg/cloud_type.h"  // #define PCL_NO_PRECOMPILE
#include "fsg/pcl.h"

namespace fsg {

#define RED_TEXT "\033[1;31m"
#define RESET_COLOR "\033[0m"
#define CERR_RED(msg) std::cerr << RED_TEXT << msg << RESET_COLOR << std::endl;

    typedef Eigen::Vector3d V3d;
    typedef Eigen::Affine3d A3d;
    typedef Eigen::Vector3f V3f;

    template <typename PointT = PnV2A5>
    class Workspace
    {
        typedef pcl::PointCloud<PointT> CloudT;
        typedef typename CloudT::Ptr CloudPtr;
        typedef pcl::KdTreeFLANN<PointT> TreeT;
        typedef typename TreeT::Ptr TreePtr;

    public:
        static float Inner(const float* v1, const float* v2) {
            return static_cast<float>(v1[0]) * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
        }

        Workspace() = default;

        int InitWorkspace(const std::vector<std::string>& ws_paths,
                          int finger_num, double tip_interval, double fin_collision_thre,
                          const std::vector<float>& palm_min_max);

        int GetWorkspaceClouds(const std::vector<std::string>& ws_paths);

        void GetLinkCol(const std::vector<std::string>& link_col_paths);

        bool CheckLinkCollision(const CloudPnPtr& obj_transformed, std::vector<float>& joint_angles);

        int GetFaceType();

        void UpdateTree();

        void ReduceFinger();

        void GetMinorTip();

        void SetMaxFinList(float obj_max_len);

        bool CheckHandCollision(const V3d& obj_origin, const V3d& obj_y_axis, const CloudPnPtr& obj_transformed,
                                float height_threshold=0.05, int num_threshold=3);

        bool CheckFinCollision(const V3d& obj_origin, const V3d& obj_y_axis,
                               const std::vector<CloudPnPtr>& tip_faces) const;

        static void GetRectangleFace(const PointT& center, std::vector<CloudPnPtr>& tip_faces);

        static void GetCircleFace(const PointT& center, std::vector<CloudPnPtr>& tip_faces);

#ifdef BUILD_WITH_VISUALIZER
        static void GetCircleFace(const PointT& center, std::vector<CloudPnPtr>& tip_faces, std::vector<CloudPnPtr>& tip_real_faces);
#endif

        int GetIdx(TreePtr tip_tree, CloudPtr tip, const CloudPnPtr& obj, std::vector<CloudPnPtr>& obj_points,
                   float dist_threshold=0.000064, float norm_threshold=0.98);

        void IntervalFilter(int finger_id, int point_id,
                            const V3d& finger_normal, const CloudPnPtr& obj);

        bool GetTipFaces(int minor_point_id, const V3d& finger_normal, const CloudPnPtr& obj,
                         V3d* finger_normals, std::vector<int>& point_indices,
                         std::vector<CloudPnPtr>& tip_faces, std::vector<CloudPnPtr>& obj_points);

        static void GetOverlap(const CloudPnPtr& tip_face, const CloudPnPtr& tar, const CloudPnPtr& contacts,
                               float dist_threshold = 0.005);

        bool GetContacts(const std::vector<CloudPnPtr>& tip_faces, const std::vector<CloudPnPtr>& tar2tip,
                         const CloudPnPtr& contacts);

        bool CheckPrimaryFingerPos(const std::vector<int>& tip_point_indices);

        void WsXOrder();

        V3d GetMinorNormal() { return averaged_minor_normal_; }

        int GetRequiredFingerNum() const { return required_finger_num_; }

        std::vector<int> GetRequiredFingerList() { return required_finger_list_; }

        std::vector<bool> GetValidFingerList() { return valid_finger_list_; }

        std::vector<CloudPtr> GetTips() { return tips_original_; }

        int GetMinorTipId() const { return minor_tip_id_; }

        int GetMinorTipSize() { return tips_original_[minor_tip_id_]->size(); }

        float* GetMinorTipPointXPtr(int idx) { return &(tips_original_[minor_tip_id_]->points[idx].x); }

        float GetTipPointAngle(int tip_id, int point_id, int angle_id) {
            return tips_original_[tip_id]->points[point_id].angle[angle_id];
        }

        void SetMode(const int m) { mode_ = m; }

#ifdef BUILD_WITH_VISUALIZER
        std::vector<CloudXyzPtr> GetPassedObj() { return passed_obj_; }
        std::vector<CloudPnPtr> GetRealTipFaces() { return real_tip_faces_; }
        CloudPnPtr GetFirstContacts() { return first_contacts_; }
        std::vector<CloudPnPtr> GetOtherContacts() { return other_contacts_; }
        std::vector<CloudPnPtr> GetLinkSkeletons() { return link_skeletons_; }
#endif

    private:
        int mode_ = 1;                        // 1 - grasp, -1 - dataset
        int face_type_{};                     // 0 - point, 1 - circle, 2 - rectangle
        int finger_num_{};                    // number of fingers
        int valid_finger_num_{};              // number of valid fingers (with cloud)
        int minor_tip_id_{};                  // id of the finger with the least number of points
        V3d averaged_minor_normal_;           // averaged normal of tiny finger workspace cloud

        double fin_collision_threshold_{};         // distance threshold to avoid collision between tip and desk
        double tip_interval_{};               // interval between two adjacent fingertips

        std::vector<CloudPtr> tips_original_;       // original tip clouds
        std::vector<TreePtr>  tips_trees_;          // kd-tree for each finger
        std::vector<bool>     valid_finger_list_;   // True - with cloud, False - empty

        std::vector<CloudPtr> link_original_;       // original link collision clouds

        std::vector<PointT> middle_points_; // middle points between 'thumb' and other fingers
        std::vector<int> closest_thu_indices_, closest_pair_indices_;  // index of closest point pairs

        std::vector<float> centroid_dists_;       // distance between centroids of 'thumb' and other fingers
        std::vector<float> thu_depth_, fin_depth_; // depth of 'thumb' and other fingers along closest point pair normal.

        std::vector<float> palm_min_max_;         // x_min, x_max, y_min, y_max, z_min, z_max

        int required_finger_num_{};                 // number of fingers required
        std::vector<int> required_finger_list_;   // 1 - enable, 0 - disable

        std::vector<int> ws_centroid_x_descent_indices_primary_;  // index of fingertips in descending order of centroid x
        std::vector<int> ws_centroid_x_descent_indices_opposite_;

#ifdef BUILD_WITH_VISUALIZER
        std::vector<CloudXyzPtr> passed_obj_;
        std::vector<CloudPnPtr> real_tip_faces_;
        CloudPnPtr first_contacts_;
        std::vector<CloudPnPtr> other_contacts_;
        std::vector<CloudPnPtr> link_skeletons_;
#endif

        pcl::PCDWriter writer_;
    };

}  // namespace fsg

#include "impl/workspace_impl.hpp"

#endif  // FSG_WORKSPACE_H_
