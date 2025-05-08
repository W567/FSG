#ifndef FSG_CLOUD_TYPE_H_
#define FSG_CLOUD_TYPE_H_

#define PCL_NO_PRECOMPILE
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/pcl_macros.h>
#include <pcl/kdtree/kdtree_flann.h>

#define COMMON_SLOT \
  PCL_ADD_POINT4D; \
  PCL_ADD_NORMAL4D; \
  float curvature; \
  float xx; \
  float xy; \
  float xz; \
  float zx; \
  float zy; \
  float zz;


#define COMMON_DATA \
  (float, x, x) \
  (float, y, y) \
  (float, z, z) \
  (float, normal_x, normal_x) \
  (float, normal_y, normal_y) \
  (float, normal_z, normal_z) \
  (float, curvature, curvature) \
  (float, xx, xx) \
  (float, xy, xy) \
  (float, xz, xz) \
  (float, zx, zx) \
  (float, zy, zy) \
  (float, zz, zz)


#define PCL_ADD_A1 \
  union EIGEN_ALIGN16 { \
    float angle[1]; \
    struct { \
      float a0; \
    }; \
  };


#define PCL_ADD_A2 \
  union EIGEN_ALIGN16 { \
    float angle[2]; \
    struct { \
      float a0; \
      float a1; \
    }; \
  };


#define PCL_ADD_A3 \
  union EIGEN_ALIGN16 { \
    float angle[3]; \
    struct { \
      float a0; \
      float a1; \
      float a2; \
    }; \
  };


#define PCL_ADD_A4 \
  union EIGEN_ALIGN16 { \
    float angle[4]; \
    struct { \
      float a0; \
      float a1; \
      float a2; \
      float a3; \
    }; \
  };


#define PCL_ADD_A5 \
  union EIGEN_ALIGN16 { \
    float angle[5]; \
    struct { \
      float a0; \
      float a1; \
      float a2; \
      float a3; \
      float a4; \
    }; \
  };


#define CLOUD_STRUCT(NAME, ADD) \
  struct EIGEN_ALIGN16 NAME \
  { \
    COMMON_SLOT \
    ADD \
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW \
  };


CLOUD_STRUCT (PnV2A1, PCL_ADD_A1)
POINT_CLOUD_REGISTER_POINT_STRUCT (PnV2A1,
                                   COMMON_DATA
                                           (float, a0, a0)
)


CLOUD_STRUCT (PnV2A2, PCL_ADD_A2)
POINT_CLOUD_REGISTER_POINT_STRUCT (PnV2A2,
                                   COMMON_DATA
                                           (float, a0, a0)
                                                   (float, a1, a1)
)


CLOUD_STRUCT (PnV2A3, PCL_ADD_A3)
POINT_CLOUD_REGISTER_POINT_STRUCT (PnV2A3,
                                   COMMON_DATA
                                           (float, a0, a0)
                                                   (float, a1, a1)
                                                   (float, a2, a2)
)


CLOUD_STRUCT (PnV2A4, PCL_ADD_A4)
POINT_CLOUD_REGISTER_POINT_STRUCT (PnV2A4,
                                   COMMON_DATA
                                           (float, a0, a0)
                                                   (float, a1, a1)
                                                   (float, a2, a2)
                                                   (float, a3, a3)
)


CLOUD_STRUCT (PnV2A5, PCL_ADD_A5)
POINT_CLOUD_REGISTER_POINT_STRUCT (PnV2A5,
                                   COMMON_DATA
                                           (float, a0, a0)
                                                   (float, a1, a1)
                                                   (float, a2, a2)
                                                   (float, a3, a3)
                                                   (float, a4, a4)
)


typedef pcl::PointCloud<pcl::PointXYZ>  CloudXyz;
typedef CloudXyz::Ptr                   CloudXyzPtr;
typedef pcl::KdTreeFLANN<pcl::PointXYZ> TreeXyz;
typedef TreeXyz::Ptr                    TreeXyzPtr;

typedef pcl::PointCloud<pcl::PointXYZRGB>  CloudXyzRgb;
typedef CloudXyzRgb::Ptr                   CloudXyzRgbPtr;
typedef pcl::KdTreeFLANN<pcl::PointXYZRGB> TreeXyzRgb;
typedef TreeXyzRgb::Ptr                    TreeXyzRgbPtr;

typedef pcl::PointCloud<pcl::PointNormal>  CloudPn;
typedef CloudPn::Ptr                       CloudPnPtr;
typedef pcl::KdTreeFLANN<pcl::PointNormal> TreePn;
typedef TreePn::Ptr                        TreePnPtr;

typedef pcl::PointCloud<PnV2A1>  CloudPnV2A1;
typedef CloudPnV2A1::Ptr         CloudPnV2A1Ptr;
typedef pcl::KdTreeFLANN<PnV2A1> TreePnV2A1;
typedef TreePnV2A1::Ptr          TreePnV2A1Ptr;

typedef pcl::PointCloud<PnV2A2>  CloudPnV2A2;
typedef CloudPnV2A2::Ptr         CloudPnV2A2Ptr;
typedef pcl::KdTreeFLANN<PnV2A2> TreePnV2A2;
typedef TreePnV2A2::Ptr          TreePnV2A2Ptr;

typedef pcl::PointCloud<PnV2A3>  CloudPnV2A3;
typedef CloudPnV2A3::Ptr         CloudPnV2A3Ptr;
typedef pcl::KdTreeFLANN<PnV2A3> TreePnV2A3;
typedef TreePnV2A3::Ptr          TreePnV2A3Ptr;

typedef pcl::PointCloud<PnV2A4>  CloudPnV2A4;
typedef CloudPnV2A4::Ptr         CloudPnV2A4Ptr;
typedef pcl::KdTreeFLANN<PnV2A4> TreePnV2A4;
typedef TreePnV2A4::Ptr          TreePnV2A4Ptr;

typedef pcl::PointCloud<PnV2A5>  CloudPnV2A5;
typedef CloudPnV2A5::Ptr         CloudPnV2A5Ptr;
typedef pcl::KdTreeFLANN<PnV2A5> TreePnV2A5;
typedef TreePnV2A5::Ptr          TreePnV2A5Ptr;

#endif  // FSG_CLOUD_TYPE_H_
