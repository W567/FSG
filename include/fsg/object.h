#ifndef FSG_OBJECT_H_
#define FSG_OBJECT_H_

#include "fsg/cloud_type.h"  // #define PCL_NO_PRECOMPILE

namespace fsg {

    typedef Eigen::Vector3d V3d;
    typedef Eigen::Affine3d A3d;

    class Object
    {
    public:
        Object() = default;

        void SetPart(const CloudPnPtr& part) { part_ = part; }
        void SetWhole(const CloudPnPtr& whole) { whole_ = whole; }
        void SetPartMinor(const CloudPnPtr& part_minor) { part_minor_ = part_minor; }

        CloudPnPtr GetPart() { return part_; }
        CloudPnPtr GetWhole() { return whole_; }
        CloudPnPtr GetPartMinor() { return part_minor_; }

        void ShufflePart() const;

        void GetCloud4Minor(const V3d& minor_normal);

        void GetWholeInfo();

        static void GetRectangleContactCandidates(const CloudPnPtr& tip, const V3d& tip_normal,
                                                  const CloudPnPtr& obj, const CloudPnPtr& contact);
        static void GetCircleContactCandidates(const CloudPnPtr& tip, double radius,
                                               const CloudPnPtr& obj, const CloudPnPtr& contact);

        std::size_t GetPartMinorSize() const { return part_minor_->size(); }

        float* GetPartMinorPointXPtr(const int id) const { return &(part_minor_->points[id].x); }

        float GetObjMaxR() const { return obj_max_r_; }
        V3d GetObjCentroid() { return obj_centroid_; }

    private:
        CloudPnPtr whole_;      // point cloud of whole object
        CloudPnPtr part_;       // point cloud of target grasp part
        CloudPnPtr part_minor_; // point cloud of target grasp part for minor finger

        float obj_max_r_{};               // maximum distance from surface point of cloud to its centroid
        V3d obj_centroid_;              // centroid of whole object cloud
    };

}  // namespace fsg

#endif  // FSG_OBJECT_H_
