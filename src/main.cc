#include "fsg/fsg.h"
#include "fsg/GraspPose.h"

bool callback(fsg::GraspPose::Request  &req, fsg::GraspPose::Response &res) {
    if (req.angle_num == 1 && req.finger_num == 2) {
        fsg::FSG<PnV2A1> planner(1, 2);
        planner.FSGCallback(req, res);
    } else if (req.angle_num == 3 && req.finger_num == 3) {
        fsg::FSG<PnV2A3> planner(3, 3);
        planner.FSGCallback(req, res);
    } else if (req.angle_num == 4 && req.finger_num == 3) {
        fsg::FSG<PnV2A4> planner(4, 3);
        planner.FSGCallback(req, res);
    } else if (req.angle_num == 4 && req.finger_num == 4) {
        fsg::FSG<PnV2A4> planner(4, 4);
        planner.FSGCallback(req, res);
    } else if (req.angle_num == 5 && req.finger_num == 5) {
        fsg::FSG<PnV2A5> planner(5, 5);
        planner.FSGCallback(req, res);
    } else {
        ROS_ERROR_STREAM("[fsgServer] Invalid input on angle_num: " << req.angle_num <<
                                                                         " or finger_num: " << req.finger_num);
        return false;
    }
    return true;
}

int main (int argc, char **argv) {
    ros::init(argc, argv, "fsgServer");
    ros::NodeHandle nh;
    ros::ServiceServer rgp_srv_ = nh.advertiseService("grasp_pose", callback);
    ros::spin();
    return 0;
}
