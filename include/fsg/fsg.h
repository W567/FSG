#ifndef FSG_FSG_H_
#define FSG_FSG_H_

#include <omp.h>
#include <thread>
#include <random>
#include <sys/time.h>
#include <ros/ros.h>
#include <tf_conversions/tf_eigen.h>

#include "fsg/cloud_type.h"  // #define PCL_NO_PRECOMPILE
#include "fsg/pcl.h"
#include "fsg/ros_pcl.h"
#include "fsg/object.h"
#include "fsg/workspace.h"
#include "fsg/eigen.h"
#include "fsg/gws.h"
#include "fsg/GraspPose.h"
#include "pcl_interface/curvFilter.h"

#ifdef BUILD_WITH_VISUALIZER
#include "fsg/visualizer.h"
#endif

namespace fsg {
    constexpr float kMu = 0.5;     // friction coefficient
    constexpr int kCircleDiv = 8;  // number of divisions for circle

    struct GraspCandidate {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        float gws_score;           // grasp wrench space score
        A3d obj_transform;         // transformation matrix of object
        CloudPnPtr contacts;       // contact points from object cloud
        CloudPnPtr tips;           // contact points from fingertip cloud
        std::vector<float> angles; // joint angles of fingers
    };
    typedef std::shared_ptr<GraspCandidate> GraspCandidatePtr;

    template <typename PointT = PnV2A5>
    class FSG {
    private:
        static bool CompareGws(const GraspCandidatePtr& e1, const GraspCandidatePtr& e2) {
            return (e1->gws_score > e2->gws_score);
        }

        void GetWorkspace();

        void ObjPcCallback(const sensor_msgs::PointCloud2ConstPtr& input);

        void ObjPartPcCallback(const sensor_msgs::PointCloud2ConstPtr& input);

        void ObjPcPrepro(const sensor_msgs::PointCloud2ConstPtr& input);

        bool GetObjBottomPose();

        int CandidateFilter(int size, int valid_pose_num);

        void Estimate();

        double GwsScore(const CloudPnPtr& contacts, const A3d& obj_transform);

        void MatchPointPair(int& idx_tip, V3d& finger_normal, V3d& obj_normal,
                            A3d& transform, A3d& translate);

        void PushToList(const GraspCandidatePtr& candidate, const CloudPnPtr& contacts,
                        const std::vector<CloudPnPtr>& tip_faces,
                        const A3d& trans, const std::vector<int>& tip_point_indices);

#ifdef BUILD_WITH_VISUALIZER
        int VisOneFinger(Visualizer& vis, const std::vector<int>& tip_point_indices,
                         const std::vector<CloudPnPtr>& tip_faces,
                         const std::vector<CloudPnPtr>& contacts, const CloudPnPtr& obj_transformed,
                         int id, int fin_id_0, int timer, int time);

        void VisProcedure(std::vector<int> tip_point_indices, std::vector<CloudPnPtr> tip_faces,
                          CloudPnPtr obj_transformed, const A3d& trans, const std::vector<CloudPnPtr>& contacts,
                          const std::vector<CloudPnPtr>& link_col_clouds);
#endif

    public:
        FSG(int angle_num, int finger_num);

        bool FSGCallback(const GraspPose::Request &req, GraspPose::Response &res);

    private:
        ros::NodeHandle nh_;

        Object obj_;
        Workspace<PointT> ws_;

        int finger_num_, angle_num_, least_finger_num_{};
        int face_type_{};

        int outer_iteration_{}, inner_iteration_{}, thread_num_, max_num_output_{};
        bool with_obj_prepro_{}, with_palm_filter_{}, with_fin_iteration_{}, with_link_col_filter_{};
        double gws_threshold_{}, palm_h_threshold_{};

        std::string tip_prefix_, contact_prefix_, obj_pc_topic_, obj_part_pc_topic_;
        std::string robot_base_frame_, obj_bottom_frame_;
        std::string aff_ext_;

        A3d obj_pose_;
        bool standalone_{};
        bool with_aff_{};
        bool obj_pc_obtained_{}, obj_part_pc_obtained_{};
        V3d base_forward_, base_upward_;
        tf::TransformListener tf_listener_;
        ros::Subscriber obj_pc_subscriber_;
        ros::Subscriber obj_part_pc_subscriber_;

        ros::ServiceServer finlist_srv_, workspace_path_srv_;
        ros::ServiceClient curv_filter_srv_;

        std::vector<float> gws_scores_;
        std::vector<geometry_msgs::Pose> palm_poses_;
        std::vector<std::vector<float>> joint_angle_lists_;
        std::vector<GraspCandidatePtr, Eigen::aligned_allocator<A3d>> candidates_list_;

        pcl::PCDWriter writer_;
        double period_;
        timeval t_start_, t_end_;

#ifdef BUILD_WITH_VISUALIZER
        int obj_point_id_{};
#endif

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

}  // namespace fsg

#include "impl/fsg_impl.hpp"

#endif  // FSG_FSG_H_