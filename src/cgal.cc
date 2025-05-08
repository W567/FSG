#include "fsg/cgal.h"

#include <CGAL/convex_hull_2.h>
#include <CGAL/Polygon_2_algorithms.h>

namespace fsg {

    bool CheckInside(const Point2 pt, Point2 *pgn_begin, Point2 *pgn_end, const K traits) {
        switch(bounded_side_2(pgn_begin, pgn_end, pt, traits)) {
            case CGAL::ON_BOUNDED_SIDE :     // inside
            case CGAL::ON_BOUNDARY:          // on the boundary
                return true;
                // case CGAL::ON_UNBOUNDED_SIDE: // outside
                //   return false;
            default:
                return false;
        }
    }

    std::vector<std::size_t> GetInside(std::vector<Point2> points, const std::vector<Point2>& query) {
        std::vector<std::size_t> indices(points.size()), boundary, result;
        std::iota(indices.begin(), indices.end(), 0);

        convex_hull_2(indices.begin(), indices.end(), std::back_inserter(boundary),
                      CgalConvexHullTraits2(make_property_map(points.data())));

        std::vector<Point2> hull;
        for (const std::size_t i : boundary) hull.push_back(points[i]);
        hull.push_back(hull[0]);  // to get a closed loop

        for (std::size_t i = 0; i < query.size(); i++) {
            if (CheckInside(query[i], hull.data(), hull.data() + hull.size(), K())) {
                result.push_back(i);
            }
        }
        return result;
    }

}  // namespace fsg