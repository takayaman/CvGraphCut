/*=============================================================================
 * Project : CvGraphCut
 * Code : util.h
 * Written : N.Takayama, UEC
 * Date : 2014/12/ 5
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Definition of cvgraphcut_base::Util
 * Utility
 *===========================================================================*/

#ifndef _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_UTIL_H_
#define _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_UTIL_H_

/*=== Include ===============================================================*/

#include <stdint.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>


/*=== Define ================================================================*/

/*=== Class Definition  =====================================================*/

namespace cvgraphcut_base {


class Util {
public:
    /*!
    * Defoult constructor
    */
    Util(void);

    /*!
    * Default destructor
    */
    ~Util(void);

    static double_t calcDistance(cv::Point2f point0, cv::Point2f point1);
    static double_t calcL2NormOfVectors(const cv::Mat &fvector0, const cv::Mat &fvector1, bool aside_cols);


};


}  // namespace cvgraphcut_base


#endif  // _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_UTIL_H_
