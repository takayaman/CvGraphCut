/*=============================================================================
 * Project : CvGraphCut
 * Code : siftmatcher.h
 * Written : N.Takayama, UEC
 * Date : 2014/12/ 5
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Definition of cvgraphcut_base::Siftmatcher
 * Build sift matching datas
 *===========================================================================*/

#ifndef CVGRAPHCUT_CVGRAPHCUT_SIFTMATCHER_H_
#define CVGRAPHCUT_CVGRAPHCUT_SIFTMATCHER_H_

/*=== Include ===============================================================*/

#include <stdint.h>
#include <glog/logging.h>

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "./siftdata.h"
#include "./siftmatchpare.h"

/*=== Define ================================================================*/

/*=== Class Definition  =====================================================*/

namespace cvgraphcut_base {

class SiftMatcher {
 public:
  /** Constructor
   * @param source_data Source SiftData for matching.
   * @param destination_data Destination SiftData for matching.
   */
  SiftMatcher(SiftData &source_data, SiftData &destination_data);

  /**  Default destructor
  */
  ~SiftMatcher(void);

  /**
  * Assignment operator
  * @param rhs Right hand side
  * @return pointer of this object
  */
  SiftMatcher& operator=(const SiftMatcher& rhs);

  /** Run SIFT matching algorithms.
   */
  void matching(void);

  /** Get result of matching.
   * @return result of matching.
   */
  std::vector<std::vector<SiftMatchPare> >& getMatchGroups(void);

  std::vector<SiftMatchPare>& getUnMatchGroup(void);

  /** Whether matching result is outputted.
   * @return true : build, false : not build.
   */
  bool isBuildMatchGroups(void);

  void drawMatching(const cv::Mat &image0, const cv::Mat &image1, cv::Mat &output);
  void drawMatchinOfGroup(const cv::Mat &image0, const cv::Mat &image1, cv::Mat &output, int32_t index_group, bool appendimage = false);

private:
  void appendImages(const cv::Mat &image0, const cv::Mat &image1, cv::Mat &output);


 private:

  SiftData &m_source_data;              /**< Source SiftData for matching */
  SiftData &m_destination_data;         /**< Destination SiftData for matching */

  std::vector<std::vector<SiftMatchPare> > m_match_groups; /**< Matching result */
  std::vector<SiftMatchPare> m_unmatch_group;
  std::vector<bool> m_match_flags0;     /**< UNUSED */
  std::vector<bool> m_match_flags1;     /**< UNUSED */

  bool is_buildmatchgroups;             /**< Flag to indicate matching is runned. */

};

/*!
 * Log output operator
 * @param lhs Left hand side
 * @param rhs Right hand side
 * @return Pointer of google::LogSink object
 */
google::LogMessage& operator<<(google::LogMessage& lhs, const SiftMatcher& rhs);

}  // namespace cvgraphcut_base


#endif  // CVGRAPHCUT_CVGRAPHCUT_SIFTMATCHER_H_
