/*=============================================================================
 * Project : CvGraphCut
 * Code : siftmatcher.cpp
 * Written : N.Takayama, UEC
 * Date : 2014/12/ 5
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Implementation of cvgraphcut_base::Siftmatcher
 * Build sift matching data
 *===========================================================================*/

/*=== Include ===============================================================*/

#include <siftmatcher.h>
#include <stdint.h>
#include <glog/logging.h>

#include "util.h"

/*=== Local Define / Local Const ============================================*/
static const double_t DefDistanceThreshold2Match = 50.0;

/*=== Class Implementation ==================================================*/

namespace cvgraphcut_base {

/*--- Constructor / Destructor / Initialize ---------------------------------*/

/* Defoult constructor */
SiftMatcher::SiftMatcher(SiftData &source_data, SiftData &destination_data)
    : m_source_data(source_data),
      m_destination_data(destination_data),
      is_buildmatchgroups(false) {
    m_match_groups.clear();
    m_match_flags0.clear();
    m_match_flags1.clear();
}

/* Default destructor */
SiftMatcher::~SiftMatcher(void) {
}

/* Assignment operator */
SiftMatcher& SiftMatcher::operator=(const SiftMatcher& rhs) {
  if (this != &rhs) {
    // TODO(N.Takayama): implement copy
  }
  return *this;
}

/*--- Operation -------------------------------------------------------------*/
void SiftMatcher::matching(void) {
    SiftMatchPare pare;
    const cv::Mat src_fvector = m_source_data.getDescriptor();
    const cv::Mat dst_fvector = m_destination_data.getDescriptor();
    m_match_flags0.resize(src_fvector.rows);
    m_match_flags1.resize(dst_fvector.rows);

    std::vector<cv::KeyPoint> src_keys = m_source_data.getKeyPoints();
    std::vector<cv::KeyPoint> dst_keys = m_destination_data.getKeyPoints();

    for(size_t index_src = 0; index_src < src_keys.size(); index_src++) {
        cv::KeyPoint src_key = src_keys.at(index_src);
        double_t mindistance = DBL_MAX;
        int32_t index_at_mindistance = -1;
        cv::Mat src_feature = src_fvector.row(index_src);
        for(size_t index_dst = 0; index_dst < dst_keys.size(); index_dst++) {
            cv::KeyPoint dst_key = dst_keys.at(index_dst);

            /* 離れたキー同士はマッチング対象外 */
            double_t distance = Util::calcDistance(src_key.pt, dst_key.pt);
            if(DefDistanceThreshold2Match < distance)
                continue;

            /* SIFT特徴量の距離を計算 */
            cv::Mat dst_feature = dst_fvector.row(index_dst);
            double_t l2_norm = Util::calcL2NormOfVectors(src_feature, dst_feature, true);
            if(l2_norm < mindistance) {
                pare.m_source_key = src_keys.at(index_src);
                pare.m_destination_key = dst_keys.at(index_dst);
                pare.m_source_descriptor = src_feature.clone();
                pare.m_destination_descriptor = dst_feature.clone();
                pare.m_distance = l2_norm;
                pare.m_diff_angle = pare.m_source_key.angle - pare.m_destination_key.angle;
                pare.m_diff_octave = pare.m_source_key.octave - pare.m_destination_key.octave;
                pare.m_diff_response = pare.m_source_key.response - pare.m_destination_key.response;
                pare.m_diff_size = pare.m_source_key.size - pare.m_destination_key.size;
                mindistance = l2_norm;
                index_at_mindistance = index_dst;
                m_match_flags0[index_src] = true;
                m_match_flags1[index_dst] = true;
                pare.is_matched = true;
            }
        }
        if(-1 == index_at_mindistance) {
            pare.m_source_key = src_keys.at(index_src);
            pare.m_destination_key = cv::KeyPoint(cv::Point2f(-1, -1), -1, -1, -1, -1, -1);
            pare.m_source_descriptor = src_feature.clone();
            pare.m_destination_descriptor = cv::Mat();
            pare.m_distance = -1;
            pare.m_diff_angle = -1;
            pare.m_diff_octave = -1;
            pare.m_diff_response = -1;
            pare.m_diff_size = -1;
            pare.is_matched = false;
        }
        /* マッチンググループ作成 */
        /* 最初のグループ */
        if(0 == m_match_groups.size()) {
            std::vector<SiftMatchPare> group;
            group.push_back(pare);
            m_match_groups.push_back(group);
        } else {
            bool is_groupfind = false;
            /* 同じマッチング点に対応する点群をグループ化 */
            for(size_t i = 0; i < m_match_groups.size(); i++) {
                std::vector<SiftMatchPare> group = m_match_groups.at(i);
                SiftMatchPare frontpare = group.front();
                if(frontpare.m_destination_key.pt == pare.m_destination_key.pt) {
                    m_match_groups[i].push_back(pare);
                    is_groupfind = true;
                    break;
                }
            }
            /* 見つからなかったら新規グループ作成 */
            if(!is_groupfind) {
                std::vector<SiftMatchPare> group;
                group.push_back(pare);
                m_match_groups.push_back(group);
            }
        }
    }
    /* マッチンググループの中で特徴量距離が少い順にソート */
    for(size_t i = 0; i < m_match_groups.size(); i++) {
        std::vector<SiftMatchPare> group = m_match_groups.at(i);
        if(group.size() < 2)
            continue;
        std::sort(group.begin(), group.end(), SiftMatchPare::lessDistance);
    }
}

/*  Log output operator */
google::LogMessage& operator<<(google::LogMessage& lhs, const SiftMatcher& rhs) {
  lhs.stream() << "cvgraphcut_base::Siftmatcher{" <<
      // TODO(N.Takayama): implement out stream of memder data
      "}" << std::endl;
  return lhs;
}

/*--- Accessor --------------------------------------------------------------*/
std::vector<std::vector<SiftMatchPare> >& SiftMatcher::getMatchGroups(void) {
    return m_match_groups;
}


bool SiftMatcher::isBuildMatchGroups(void) {
    return is_buildmatchgroups;
}


/*--- Event -----------------------------------------------------------------*/

}  // namespace cvgraphcut_base


