#ifndef FSG_EIGEN_H_
#define FSG_EIGEN_H_

#include <geometry_msgs/Pose.h>
#include <eigen_conversions/eigen_msg.h>

namespace fsg {

    typedef Eigen::Vector3d V3d;
    typedef Eigen::Affine3d A3d;
    typedef Eigen::Matrix3d M3d;

    inline double Clamp(const double val, const double min, const double max) {
        return std::max(min, std::min(max, val));
    }

    inline void PoseMsg2Eigen(const geometry_msgs::Pose& m, A3d& e) {
        tf::poseMsgToEigen(m, e);
    }

    inline void PoseEigen2Msg(const A3d& e, geometry_msgs::Pose& m) {
        tf::poseEigenToMsg(e, m);
    }

// Get the transformation matrix to transform points in target object, in order to:
//   * have the positions of the point pair from obj and fin coincided
//   * have the normal direction of the point pair coincided
    A3d CoincideMatrix(const V3d& translate, const V3d& target_normal,  const V3d& source_normal);

// rotation matrix from obj frame to contact frame.
    M3d RotMat(const V3d& target_normal, const V3d& source_normal = V3d(0, 0, 1));

// transform matrix for wrench from contact frame to object frame
    Eigen::MatrixXd ContactMap(const V3d& target_z, const V3d& translate);

}  // namespace fsg

#endif  // FSG_EIGEN_H_