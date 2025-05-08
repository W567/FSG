#include "fsg/object.h"

#include <vector>
#include <random>
#include "fsg/cgal.h"
#include "fsg/eigen.h"
#include "fsg/pcl.h"

namespace fsg {

    void Object::ShufflePart() const {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(part_->points.begin(), part_->points.end(), gen);
    }

// Tiny finger contact area
// Both the normal of minor finger and the normal of object are under the robot base frame
    void Object::GetCloud4Minor(const V3d& minor_normal) {
        const CloudPnPtr part_minor_tmp (new CloudPn);
        const Eigen::Vector3f minor_normal_f = minor_normal.cast<float>();
        for (const auto& point : part_->points) {
            if (minor_normal_f.dot(point.getNormalVector3fMap()) > -0.001) {
                part_minor_tmp->points.push_back(point);
            }
        }
        part_minor_ = part_minor_tmp;
    }

    void Object::GetWholeInfo() {
        GetCentroid<pcl::PointNormal>(whole_, obj_centroid_);
        GetMaxR<pcl::PointNormal>(whole_, obj_centroid_, obj_max_r_);
    }

// Find points from target object which locate inside the area of fingertip
    void Object::GetRectangleContactCandidates(
            const CloudPnPtr& tip, const V3d&  tip_normal,
            const CloudPnPtr& obj, const CloudPnPtr& contact) {
        const V3d origin(0, 0, 0);
        const V3d z_axis(0, 0, 1);
        // rotate tip_normal to be same with z_axis
        const A3d transform_matrix = CoincideMatrix(origin, z_axis, tip_normal);

        const CloudPnPtr obj_trans (new CloudPn);
        const CloudPnPtr tip_trans (new CloudPn);
        transformPointCloud(*obj, *obj_trans, transform_matrix);
        transformPointCloud(*tip, *tip_trans, transform_matrix);

        // Check on x-o-y plane
        std::vector<Point2> tip_face, obj_face;
        for (const auto& point : tip_trans->points) {
            tip_face.emplace_back(point.x, point.y);
        }
        for (const auto& point : obj_trans->points) {
            obj_face.emplace_back(point.x, point.y);
        }
        std::vector<std::size_t> indices = GetInside(tip_face, obj_face);
        for (const std::size_t i : indices) {
            contact->points.push_back(obj->points[i]);
        }
    }

    void Object::GetCircleContactCandidates(const CloudPnPtr& tip, const double radius,
                                            const CloudPnPtr& obj, const CloudPnPtr& contact) {
        const TreePnPtr obj_tree(new pcl::KdTreeFLANN<pcl::PointNormal>);
        obj_tree->setInputCloud(obj);

        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        obj_tree->radiusSearch(tip->points[0], radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
        contact->points.reserve(pointIdxRadiusSearch.size());
        for (const int i : pointIdxRadiusSearch) {
            contact->points.push_back(obj->points[i]);
        }
    }

}  // namespace fsg