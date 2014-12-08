/*=============================================================================
 * Project : CvGraphCut
 * Code : siftmatchpare.h
 * Written : N.Takayama, UEC
 * Date : 2014/12/ 5
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Definition of cvgraphcut_base::Siftmatchpare
 * Information about matched two sift keypoints
 *===========================================================================*/

#ifndef _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_SIFTMATCHPARE_H_
#define _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_SIFTMATCHPARE_H_

/*=== Include ===============================================================*/

#include <stdint.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

/*=== Define ================================================================*/

/*=== Class Definition  =====================================================*/

namespace cvgraphcut_base {


class SiftMatchPare {
public:
    /*!
    * Defoult constructor
    */
    SiftMatchPare(void);

    /*!
    * Default destructor
    */
    ~SiftMatchPare(void);

    /*!
    * Copy constructor
    */
    SiftMatchPare(const SiftMatchPare& rhs);

    /*!
    * Assignment operator
    * @param rhs Right hand side
    * @return pointer of this object
    */
    SiftMatchPare& operator=(const SiftMatchPare& rhs);

    static bool lessDistance(const SiftMatchPare &lhs, const SiftMatchPare &rhs)
    {
        return lhs.m_distance < rhs.m_distance;
    }

public:
    cv::KeyPoint m_source_key;
    cv::KeyPoint m_destination_key;

    cv::Mat m_source_descriptor;
    cv::Mat m_destination_descriptor;

    double_t m_distance;
    double_t m_diff_size;
    double_t m_diff_angle;
    double_t m_diff_response;
    double_t m_diff_octave;

    bool is_matched;
};

/*!
 * Log output operator
 * @param lhs Left hand side
 * @param rhs Right hand side
 * @return Pointer of google::LogSink object
 */
google::LogMessage& operator<<(google::LogMessage& lhs, const SiftMatchPare& rhs);

}  // namespace cvgraphcut_base


#endif  // _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_SIFTMATCHPARE_H_
