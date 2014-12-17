/*=============================================================================
 * Project : CvGraphCut
 * Code : siftmatchpare.h
 * Written : N.Takayama, UEC
 * Date : 2014/12/ 5
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Definition of cvgraphcut_base::Siftmatchpare
 * Information about matched two sift keypoints
 *===========================================================================*/

#ifndef CVGRAPHCUT_CVGRAPHCUT_SIFTMATCHPARE_H_
#define CVGRAPHCUT_CVGRAPHCUT_SIFTMATCHPARE_H_

/*=== Include ===============================================================*/

#include <stdint.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

/*=== Define ================================================================*/

/*=== Class Definition  =====================================================*/

namespace cvgraphcut_base {

/** Class to store SIFT matching information
 */
class SiftMatchPare {
 public:
  /**  Defoult constructor
   */
  SiftMatchPare(void);

  /** Default destructor
   */
  ~SiftMatchPare(void);

  /**  Copy constructor
   */
  SiftMatchPare(const SiftMatchPare& rhs);

  /**  Assignment operator
   * @param rhs Right hand side
   * @return pointer of this object
   */
  SiftMatchPare& operator=(const SiftMatchPare& rhs);

  /** Sort algorithm based on SIFT feature distance.
   * @param lhs Left hand side
   * @param rhs Right hand side
   * @return true : lhs is lower, false : lhs is bigger.
   */
  static bool lessDistance(const SiftMatchPare &lhs, const SiftMatchPare &rhs) {
    return lhs.m_distance < rhs.m_distance;
  }

 public:
  cv::KeyPoint m_source_key;            /**< Keypoint of Source */
  cv::KeyPoint m_destination_key;       /**< Keypoint of Destination */

  cv::Mat m_source_descriptor;          /**< SIFT features of Source */
  cv::Mat m_destination_descriptor;     /**< SIFT features of Destination */
  cv::Mat m_difference_descriptor;      /**< Difference of SIFT features */

  double_t m_distance;                  /**< L2-Norm of SIFT features */
  double_t m_diff_size;                 /**< Difference of SIFT size */
  double_t m_diff_angle;                /**< Difference of SIFT angle */
  double_t m_diff_response;             /**< Difference of SIFT responce */
  double_t m_diff_octave;               /**< Difference of SIFT octave */

  bool is_matched;                      /**< Flag to indicate mathing pare is found. */
};

/*!
 * Log output operator
 * @param lhs Left hand side
 * @param rhs Right hand side
 * @return Pointer of google::LogSink object
 */
google::LogMessage& operator<<(google::LogMessage& lhs, const SiftMatchPare& rhs);

}  // namespace cvgraphcut_base


#endif  // CVGRAPHCUT_CVGRAPHCUT_SIFTMATCHPARE_H_
