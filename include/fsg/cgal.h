#ifndef FSG_CGAL_H_
#define FSG_CGAL_H_

#include <CGAL/property_map.h>
#include <CGAL/Convex_hull_traits_adapter_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

namespace fsg {

    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
    typedef K::Point_2 Point2;
    typedef CGAL::Convex_hull_traits_adapter_2<K, CGAL::Pointer_property_map<Point2>::type> CgalConvexHullTraits2;

    bool CheckInside(Point2 pt, Point2 *pgn_begin, Point2 *pgn_end, K traits);

    std::vector<std::size_t> GetInside(std::vector<Point2> points, const std::vector<Point2>& query);

}  // namespace fsg

#endif  // FSG_CGAL_H_