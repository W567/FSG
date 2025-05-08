#include "fsg/visualizer.h"

namespace fsg {

    Visualizer::Visualizer(const double camera_angle, const double camera_angle_step,
                           const double camera_distance, const double ax_size,
                           const double bg_r, const double bg_g, const double bg_b,
                           const std::string& window_name, const int spin_rate) {
        viewer_ = boost::make_shared<pcl::visualization::PCLVisualizer>(window_name);
        viewer_->setBackgroundColor(bg_r, bg_g, bg_b);
        viewer_->addCoordinateSystem(ax_size);

        angle_ = camera_angle;
        angle_step_ = camera_angle_step;
        distance_ = camera_distance;
        spin_rate_ = spin_rate;
    }

    void Visualizer::AddCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                              const std::string& id, const double size,
                              const double r, const double g, const double b) const
    {
        viewer_->addPointCloud<pcl::PointXYZ>(cloud, id);
        ChangeColor(id, r, g, b);
        ChangeSize(id, size);
    }

    void Visualizer::AddSphere(const pcl::PointXYZ point,
                               const double radius, const std::string& id,
                               const double r, const double g, const double b) const
    {
        viewer_->addSphere(point, radius, r, g, b, id);
    }

    void Visualizer::RemoveShape(const std::string& id) const
    {
        viewer_->removeShape(id);
    }

    void Visualizer::ChangeSize(const std::string& id, const double size) const
    {
        viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, size, id);
    }

    void Visualizer::ChangeColor(const std::string& id, const double r, const double g, const double b) const
    {
        viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r, g, b, id);
    }

    void Visualizer::ChangeOpacity(const std::string& id, const double opacity) const
    {
        viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, opacity, id);
    }

    void Visualizer::UpdateViewer() const
    {
        viewer_->spinOnce(spin_rate_);
    }

    void Visualizer::UpdateCamera() {
        // Increment angle to rotate the view
        angle_ += angle_step_;

        // Update camera position to rotate around the y-axis
        const double camera_x = distance_ * sin(angle_);
        const double camera_z = distance_ * cos(angle_);
        viewer_->setCameraPosition(camera_x, 0, camera_z, 0, 1, 0);
    }

}  // namespace fsg