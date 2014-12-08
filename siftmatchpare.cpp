/*=============================================================================
 * Project : CvGraphCut
 * Code : siftmatchpare.cpp
 * Written : N.Takayama, UEC
 * Date : 2014/12/ 5
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Implementation of cvgraphcut_base::Siftmatchpare
 * Information about matched 2 sift keypoints
 *===========================================================================*/

/*=== Include ===============================================================*/

#include <siftmatchpare.h>
#include <stdint.h>
#include <glog/logging.h>

/*=== Local Define / Local Const ============================================*/

/*=== Class Implementation ==================================================*/

namespace cvgraphcut_base {


/*--- Constructor / Destructor / Initialize ---------------------------------*/

/* Defoult constructor */
SiftMatchPare::SiftMatchPare(void)
    : m_source_key(cv::KeyPoint()),
      m_destination_key(cv::KeyPoint()),
      m_source_descriptor(cv::Mat()),
      m_destination_descriptor(cv::Mat()),
      m_distance(0.0),
      m_diff_size(0.0),
      m_diff_angle(0.0),
      m_diff_response(0.0),
      m_diff_octave(0.0) {
}

/* Default destructor */
SiftMatchPare::~SiftMatchPare(void) {
}

/*  Copy constructor */
SiftMatchPare::SiftMatchPare(const SiftMatchPare& rhs) {
    if(this != &rhs){
        m_source_key = rhs.m_source_key;
        m_destination_key = rhs.m_destination_key;
        m_source_descriptor = rhs.m_source_descriptor;
        m_destination_descriptor = rhs.m_destination_descriptor;
        m_distance = rhs.m_distance;
        m_diff_size = rhs.m_diff_size;
        m_diff_angle = rhs.m_diff_angle;
        m_diff_response = rhs.m_diff_response;
        m_diff_octave = rhs.m_diff_octave;
    }
}

/* Assignment operator */
SiftMatchPare& SiftMatchPare::operator=(const SiftMatchPare& rhs) {
    if(this != &rhs){
        m_source_key = rhs.m_source_key;
        m_destination_key = rhs.m_destination_key;
        m_distance = rhs.m_distance;
        m_diff_size = rhs.m_diff_size;
        m_diff_angle = rhs.m_diff_angle;
        m_diff_response = rhs.m_diff_response;
        m_diff_octave = rhs.m_diff_octave;
    }
    return *this;
}

/*--- Operation -------------------------------------------------------------*/

/*  Log output operator */
google::LogMessage& operator<<(google::LogMessage& lhs, const SiftMatchPare& rhs) {
  lhs.stream() << "cvgraphcut_base::Siftmatchpare{" << std::endl
               << "m_source_key : (" << rhs.m_source_key.pt.x << "," << rhs.m_source_key.pt.y << ")" << std::endl
               << "m_destination_key : (" << rhs.m_destination_key.pt.x << "," << rhs.m_destination_key.pt.y << ")" << std::endl
               << "m_distance : " << rhs.m_distance << std::endl
               << "m_diff_size : " << rhs.m_diff_size << std::endl
               << "m_diff_angle : " << rhs.m_diff_angle << std::endl
               << "m_diff_response : " << rhs.m_diff_response << std::endl
               << "m_diff_octave : " << rhs.m_diff_octave << std::endl
               << "is_mathed : " << (rhs.is_matched ? "YES" : "NO") << std::endl
               << "}" << std::endl;
  return lhs;
}

/*--- Accessor --------------------------------------------------------------*/

/*--- Event -----------------------------------------------------------------*/

}  // namespace cvgraphcut_base


