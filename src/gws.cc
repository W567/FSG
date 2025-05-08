// Based on https://github.com/graspit-simulator/graspit/blob/4.0/src/gws.cpp
#include "fsg/gws.h"

extern "C"
{
#include <libqhull_r/qhull_ra.h>    // libqhull_r/qhull_r.h  // multi-threads
}
#include <iostream>

namespace fsg {

    int GWS::buildHyperplanesFromWrenches(
            const Eigen::MatrixXd& wr,
            const Eigen::Matrix<int, 6, 1>& use_dimensions,
            double &hull_area_,
            double &hull_volume_,
            int &num_hyperplanes_,
            Eigen::MatrixXd &hyperplanes_) {
        // Set qhull variables
        const int num_wrenches = static_cast<int>(wr.cols());
        const int dimensions = static_cast<int>((use_dimensions.array() == 1).count());
        if (wr.rows() != 6) {
            std::cerr << "Row number of wrenches is " << wr.rows() << ", must be 6" << std::endl;
            return -1;
        }

        coordT* wrenches = new coordT[num_wrenches * dimensions];  // realT (i.e. double)

        for (int w = 0; w < num_wrenches; w++)
        {
            int i = 0;
            for (int d = 0; d < 6; d++) {
                if (use_dimensions(d)) {
                    wrenches[w * dimensions + i] = wr(d, w);
                    i++;
                }
            }
        }

        const auto struct_qhT (new qhT);
        /* initialize memory and stdio files */
        //  qhT    stdin  stdout  stderr argc argv
        qh_init_A(struct_qhT, nullptr, stdout, stderr, 0, nullptr);

        // http://www.qhull.org/src/libqhull_r/global_r.c
        struct_qhT->NOerrexit = False;

        char options[200];
        sprintf(options, "qhull Pp n Qx QJ C-0 Q12");
        qh_initflags(struct_qhT, options);

        constexpr boolT is_malloc = False;
        qh_init_B(struct_qhT, &wrenches[0], num_wrenches, dimensions, is_malloc);

        qh_qhull(struct_qhT);
        qh_check_output(struct_qhT);  // check at the end of qhull algorithm

        qh_getarea(struct_qhT, struct_qhT->facet_list);

        delete[] wrenches;
        hull_area_ = struct_qhT->totarea;
        hull_volume_ = struct_qhT->totvol;
        num_hyperplanes_ = struct_qhT->num_facets;

        hyperplanes_ = Eigen::MatrixXd(7, num_hyperplanes_);
        int i = 0;
        for (const facetT* facet=struct_qhT->facet_list; facet && facet->next; facet=facet->next) {
            int hd = 0;
            for (int d = 0; d < 6; d++) {
                if (use_dimensions(d)) {
                    hyperplanes_(d, i) = facet->normal[hd];
                    hd++;
                } else {
                    hyperplanes_(d, i) = 0;
                }
            }
            hyperplanes_(6, i) = facet->offset;
            i++;
        }

        int curlong, totlong;
        struct_qhT->NOerrexit = True;
        qh_freeqhull(struct_qhT, !qh_ALL);
        qh_memfreeshort(struct_qhT, &curlong, &totlong);
        if (curlong || totlong) {
            fprintf(stderr, "qhull internal warning (main): did not free %d bytes of long memory (%d pieces)\n", totlong,
                    curlong);
        }

        delete struct_qhT;
        return 0;
    }

    int GWS::computeQualityMetrics(
            const Eigen::MatrixXd &hyperplanes_,
            const int &num_hyperplanes_,
            double &epsilon_,
            double &signed_volume_,
            const double &hull_volume_,
            Eigen::MatrixXd &min_wrench_) {
        if (hyperplanes_.cols() == 0) {
            std::cerr << "hyperplanes is not set" << std::endl;
            return -1;
        }

        double min_offset_abs = std::numeric_limits<double>::max();
        int min_idx = -1;
        for (int i = 0; i < num_hyperplanes_; i++) {
            double offset = hyperplanes_(6, i);
            double offset_abs = std::abs(offset);
            if (offset_abs < min_offset_abs) {
                min_offset_abs = offset_abs;
                min_idx = i;
                epsilon_ = -offset;  // plus -> force-closure grasp
                signed_volume_ = (offset >= 0) ? -hull_volume_ : hull_volume_;
            }
        }

        if (min_idx < 0) {
            std::cerr << "No valid hyperplane found." << std::endl;
            return -1;
        }
        min_wrench_ = hyperplanes_.block<6, 1>(0, min_idx);
        return 0;
    }

    double
    L1GWS::build(
            const Eigen::MatrixXd& wrenches,
            const Eigen::Matrix<int, 6, 1>& use_dimensions)
    {
        double hull_volume_ = 0;
        int num_hyperplanes_ = 0;
        Eigen::MatrixXd hyperplanes_;
        hyperplanes_.resize(0, 0);
        double epsilon_ = -1;
        Eigen::MatrixXd min_wrench_;
        min_wrench_.resize(0, 0);

        const long dimensions = (use_dimensions.array() == 1).count();
        if (dimensions < 2) {
            std::cerr << "At least 2 used dimensions needed" << std::endl;
            return -1;
        }

        int result;
        try
        {
            double hull_area_ = 0;
            result = buildHyperplanesFromWrenches(wrenches, use_dimensions, hull_area_,
                                                  hull_volume_, num_hyperplanes_, hyperplanes_);
            if (result < 0) {
                std::cerr << "Build hyperplanes failed!!!" << std::endl;
                return epsilon_;
            }
        } catch (...) {
            std::cerr << "Build QHull failed!!!" << std::endl;
            return epsilon_;
        }


        double signed_volume_ = 0;
        result = computeQualityMetrics(hyperplanes_, num_hyperplanes_, epsilon_,
                                       signed_volume_, hull_volume_, min_wrench_);

        if (result < 0) {
            std::cerr << "Compute Quality Metrics failed!!!" << std::endl;
        }
        return epsilon_;
    }

}  // namespace fsg