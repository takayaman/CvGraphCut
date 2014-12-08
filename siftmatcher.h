/*=============================================================================
 * Project : CvGraphCut
 * Code : siftmatcher.h
 * Written : N.Takayama, UEC
 * Date : 2014/12/ 5
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Definition of cvgraphcut_base::Siftmatcher
 * Build sift matching datas
 *===========================================================================*/

#ifndef _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_SIFTMATCHER_H_
#define _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_SIFTMATCHER_H_

/*=== Include ===============================================================*/

#include <stdint.h>
#include <glog/logging.h>

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "siftdata.h"
#include "siftmatchpare.h"

/*=== Define ================================================================*/

/*=== Class Definition  =====================================================*/

namespace cvgraphcut_base {

class SiftMatcher {
public:
    /*!
    * Defoult constructor
    */
    SiftMatcher(SiftData &source_data, SiftData &destination_data);

    /*!
    * Default destructor
    */
    ~SiftMatcher(void);

    /*!
    * Assignment operator
    * @param rhs Right hand side
    * @return pointer of this object
    */
    SiftMatcher& operator=(const SiftMatcher& rhs);

    void matching(void);

    std::vector<std::vector<SiftMatchPare> >& getMatchGroups(void);
    bool isBuildMatchGroups(void);

private:
    SiftData &m_source_data;
    SiftData &m_destination_data;

    std::vector<std::vector<SiftMatchPare> > m_match_groups;
    std::vector<bool> m_match_flags0;
    std::vector<bool> m_match_flags1;

    bool is_buildmatchgroups;

};

/*!
 * Log output operator
 * @param lhs Left hand side
 * @param rhs Right hand side
 * @return Pointer of google::LogSink object
 */
google::LogMessage& operator<<(google::LogMessage& lhs, const SiftMatcher& rhs);

}  // namespace cvgraphcut_base


#endif  // _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_SIFTMATCHER_H_
