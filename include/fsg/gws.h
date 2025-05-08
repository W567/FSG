// Based on https://github.com/graspit-simulator/graspit/blob/4.0/src/gws.cpp
#ifndef FSG_GWS_H_
#define FSG_GWS_H_

#include <Eigen/Eigen>

namespace fsg {

// https://stackoverflow.com/questions/31549398/c-eigen-initialize-static-matrix
    static const Eigen::Matrix<int, 6, 1> ALL_DIMENSIONS = (Eigen::Matrix<int, 6, 1>() << 1, 1, 1, 1, 1, 1).finished();

    class GWS
    {
    public:
        virtual ~GWS() = default;
        GWS() = default;

        static int
        buildHyperplanesFromWrenches(
                const Eigen::MatrixXd& wr,
                const Eigen::Matrix<int, 6, 1>& use_dimensions,
                double &hull_area_,
                double &hull_volume_,
                int &num_hyperplanes_,
                Eigen::MatrixXd &hyperplanes_);

        static int
        computeQualityMetrics(
                const Eigen::MatrixXd &hyperplanes_,
                const int &num_hyperplanes_,
                double &epsilon_,
                double &signed_volume_,
                const double &hull_volume_,
                Eigen::MatrixXd &min_wrench_);

        virtual double
        build(
                const Eigen::MatrixXd& wrenches,
                const Eigen::Matrix<int, 6, 1>& use_dimensions) = 0;
    };

    class L1GWS final : public GWS
    {
    public:
        double
        build(
                const Eigen::MatrixXd& wrenches,
                const Eigen::Matrix<int, 6, 1>& use_dimensions) override;
    };

}  // namespace fsg

#endif  // FSG_GWS_H_