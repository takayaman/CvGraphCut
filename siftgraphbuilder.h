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
    static const int32_t kDefClusters = 5;

    typedef enum DrawType_TAG{
        DRAW_DEFAULT = 0,
        DRAW_SOURCE = DRAW_DEFAULT,
        DRAW_SINK,
        DRAW_BOTH,
        DRAW_NUM
    }DrawType;

    typedef enum DrawDivType_TAG{
        DRAW_DIV_DEFAULT = 0,
        DRAW_DIV_DELAUNAY = DRAW_DIV_DEFAULT,
        DRAW_DIV_VORONOY,
        DRAW_DIV_NUM
    }DrawDivType;

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

    void drawCuttedPoints(cv::Mat &image, DrawType type);
    void drawGraphs(cv::Mat &image, DrawDivType type);
    void drawSiftGroups(cv::Mat &image, int32_t index, cv::Scalar color);

private:
    void buildSubdivGraph(void);
    void learnVisualWord(void);
    void calcTermWeights(void);
    void calcNeighborWeights(void);

    void debugPrintMembers(void);

private:
    std::vector<std::vector<SiftMatchPare> > &m_match_groups;
    GCGraph m_graph;
    cv::Rect m_image_area;

    cv::Mat m_fore_visualword;
    cv::Mat m_back_visualword;
    double_t m_fore_coefficients[kDefClusters];
    double_t m_back_coefficients[kDefClusters];
    cv::Mat m_fore_centroid;
    cv::Mat m_back_centroid;
    cv::Mat m_fore_labels;
    cv::Mat m_back_labels;

    /* ドロネー分割用 */
    cv::Subdiv2D m_subdiv;
    /* keypointから抽出したドロネー分割の頂点 */
    std::vector<cv::Point2f> m_delaunay_points;
    /* ドロネー分割のエッジ */
    std::vector<cv::Vec4f> m_delaunay_edges;
    /* ボロノイ領域のIDリスト */
    std::vector<int32_t> m_facet_index;
    /* ボロノイ分割の頂点リスト */
    std::vector<std::vector<cv::Point2f> > m_facet_lists;
    /* ボロノイ分割の母点リスト */
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
