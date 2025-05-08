#ifndef FSG_VISUALIZER_H_
#define FSG_VISUALIZER_H_

#include <string>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace fsg {

    class Visualizer
    {
    public:
        Visualizer(double camera_angle, double camera_angle_step,
                   double camera_distance, double ax_size,
                   double bg_r=1.0, double bg_g=1.0, double bg_b=1.0,
                   const std::string& window_name="PCL Viewer", int spin_rate=10);

        void AddCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::string& id,
                      double size, double r, double g, double b) const;

        void AddSphere(pcl::PointXYZ point, double radius, const std::string& id,
                       double r=0.2, double g=0.2, double b=0.2) const;

        void RemoveShape(const std::string& id) const;

        void ChangeSize(const std::string& id, double size) const;

        void ChangeColor(const std::string& id, double r, double g, double b) const;

        void ChangeOpacity(const std::string& id, double opacity) const;

        void UpdateViewer() const;

        pcl::visualization::PCLVisualizer::Ptr getViewer() const { return viewer_; }

        void UpdateCamera();

        bool WasStopped() const { return viewer_->wasStopped(); }

    private:
        double angle_;
        double angle_step_;
        double distance_;
        int spin_rate_;
        pcl::visualization::PCLVisualizer::Ptr viewer_;
    };

}  // namespace fsg

#endif  // FSG_VISUALIZER_H_