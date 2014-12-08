/*=============================================================================
 * Project : CvGraphCut
 * Code : siftgraphbuilder.h
 * Written : N.Takayama, UEC
 * Date : 2014/12/ 5
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Definition of cvgraphcut_base::Siftgraphbuilder
 * Build GCGraph based on Sift Keypoints
 *===========================================================================*/

#ifndef _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_SIFTGRAPHBUILDER_H_
#define _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_SIFTGRAPHBUILDER_H_

/*=== Include ===============================================================*/

#include <stdint.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "gcgraph.h"
#include "siftdata.h"
#include "siftmatchpare.h"

/*=== Define ================================================================*/
// 前景グループ 0, 1, 2,
// 背景グループ 3, 4, 12,

/*=== Class Definition  =====================================================*/

namespace cvgraphcut_base {


class SiftGraphBuilder {
public:
    typedef enum DrawType_TAG{
        DRAW_DEFAULT = 0,
        DRAW_SOURCE = DRAW_DEFAULT,
        DRAW_SINK,
        DRAW_BOTH,
        DRAW_NUM
    }DrawType;

    /*!
    * Defoult constructor
    */
    SiftGraphBuilder(std::vector<std::vector<SiftMatchPare> >& match_groups, cv::Rect image_area);

    /*!
    * Default destructor
    */
    ~SiftGraphBuilder(void);

    /*!
    * Assignment operator
    * @param rhs Right hand side
    * @return pointer of this object
    */
    SiftGraphBuilder& operator=(const SiftGraphBuilder& rhs);

    void build(void);

    void cutGraph(void);

    void drawPoints(cv::Mat &image, DrawType type);

private:
    void buildSubdivGraph(void);
    void calcTermWeights(void);
    void calcNeighborWeights(void);

private:
    std::vector<std::vector<SiftMatchPare> > &m_match_groups;
    GCGraph m_graph;
    cv::Rect m_image_area;

    /* ドロネー分割用 */
    cv::Subdiv2D m_subdiv;
    std::vector<cv::Point2f> m_delaunay_points;
    std::vector<cv::Vec4f> m_delaunay_edges;
    std::vector<int32_t> m_facet_index;
    std::vector<std::vector<cv::Point2f> > m_facet_lists;
    std::vector<cv::Point2f> m_facet_centers;

    std::vector<cv::Point2f> m_source_points;
    std::vector<cv::Point2f> m_sink_points;

};

/*!
 * Log output operator
 * @param lhs Left hand side
 * @param rhs Right hand side
 * @return Pointer of google::LogSink object
 */
google::LogMessage& operator<<(google::LogMessage& lhs, const SiftGraphBuilder& rhs);

}  // namespace cvgraphcut_base


#endif  // _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_SIFTGRAPHBUILDER_H_
