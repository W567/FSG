#include "fsg/eigen.h"

#include <cmath>

namespace fsg {

    A3d CoincideMatrix(const V3d& translate, const V3d& target_normal, const V3d& source_normal) {
        A3d transform;
        transform.linear() = RotMat(target_normal, source_normal);
        transform.translation() = translate;
        return transform;
    }

    M3d RotMat(const V3d& target_normal, const V3d& source_normal) {

        if (target_normal.isApprox(source_normal)) {
            return M3d::Identity();
        }
        if (target_normal.isApprox(-source_normal)) {
            const V3d axis = (target_normal[0] == 0 && target_normal[1] == 0) ? V3d(1, 0, 0) :
                             V3d(-target_normal[1], target_normal[0], 0).normalized();
            return Eigen::AngleAxisd(M_PI, axis).toRotationMatrix();
        }

        const V3d axis = source_normal.cross(target_normal).normalized();
        const double c2 = (target_normal - source_normal).squaredNorm();
        const double angle = acos(Clamp((2.0 - c2) / 2.0, -1.0, 1.0));
        return Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    }

    Eigen::MatrixXd ContactMap(const V3d& target_z, const V3d& translate) {
        M3d rot_mat = RotMat(target_z);
        M3d poc;
        poc << 0, -translate[2], translate[1],
                translate[2], 0, -translate[0],
                -translate[1], translate[0], 0;
        M3d p_r = poc * rot_mat;

        Eigen::MatrixXd map(6,6);
        map << rot_mat(0,0), rot_mat(0,1), rot_mat(0,2),           0,             0,            0,
                rot_mat(1,0), rot_mat(1,1), rot_mat(1,2),           0,             0,            0,
                rot_mat(2,0), rot_mat(2,1), rot_mat(2,2),           0,             0,            0,
                p_r(0,0),     p_r(0,1),     p_r(0,2), rot_mat(0,0), rot_mat(0,1), rot_mat(0,2),
                p_r(1,0),     p_r(1,1),     p_r(1,2), rot_mat(1,0), rot_mat(1,1), rot_mat(1,2),
                p_r(2,0),     p_r(2,1),     p_r(2,2), rot_mat(2,0), rot_mat(2,1), rot_mat(2,2);
        return map;
    }

}  // namespace fsg