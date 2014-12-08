/*=============================================================================
 * Project : CvGraphCut
 * Code : siftgraphbuilder.cpp
 * Written : N.Takayama, UEC
 * Date : 2014/12/ 5
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Implementation of cvgraphcut_base::Siftgraphbuilder
 * Build GCGraph based on Sift keypoints
 *===========================================================================*/

/*=== Include ===============================================================*/

#include <siftgraphbuilder.h>
#include <stdint.h>
#include <glog/logging.h>

#include "util.h"

/*=== Local Define / Local Const ============================================*/
// 前景グループ 0, 1, 2,
// 背景グループ 3, 4, 12,

/*=== Class Implementation ==================================================*/

namespace cvgraphcut_base {


/*--- Constructor / Destructor / Initialize ---------------------------------*/

/* Defoult constructor */
SiftGraphBuilder::SiftGraphBuilder(std::vector<std::vector<SiftMatchPare> >& match_groups, cv::Rect image_area)
    : m_match_groups(match_groups),
      m_graph(GCGraph()),
      m_image_area(image_area) {
    m_delaunay_points.clear();
    m_delaunay_edges.clear();
    m_facet_index.clear();;
    m_facet_lists.clear();
    m_facet_centers.clear();
    m_source_points.clear();
    m_sink_points.clear();
}

/* Default destructor */
SiftGraphBuilder::~SiftGraphBuilder(void) {
}



/* Assignment operator */
SiftGraphBuilder& SiftGraphBuilder::operator=(const SiftGraphBuilder& rhs) {
  if (this != &rhs) {
    // TODO(N.Takayama): implement copy
  }
  return *this;
}

/*--- Operation -------------------------------------------------------------*/
void SiftGraphBuilder::build(void) {
    buildSubdivGraph();

    /* グラフ構築 */
    m_graph.create(m_facet_centers.size(), m_delaunay_edges.size());

    /* 頂点追加 + T-link追加 */
    calcTermWeights();

    /* N-link追加 */
    calcNeighborWeights();

}

void SiftGraphBuilder::buildSubdivGraph(void) {
    m_subdiv.initDelaunay(m_image_area);

    /* マッチング率が一番高いもののみ
     * TODO モードかフラグで他の対応点を使用するか切り替える
     */
    for(size_t i = 0; i < m_match_groups.size(); i++) {
        std::vector<SiftMatchPare> group = m_match_groups.at(i);
        SiftMatchPare frontpare = group.front();
        if(frontpare.is_matched)
            m_delaunay_points.push_back(frontpare.m_source_key.pt);
    }
    m_subdiv.insert(m_delaunay_points);
    m_subdiv.getVoronoiFacetList(m_facet_index, m_facet_lists, m_facet_centers);
    m_subdiv.getEdgeList(m_delaunay_edges);
}

void SiftGraphBuilder::calcTermWeights(void) {
    /* 正規化用基準値 */
    double_t value_for_normalize = Util::calcDistance(m_image_area.tl(), m_image_area.br());
    for(size_t i = 0; i < m_facet_centers.size(); i++) {
        m_graph.addVertex();
        double_t fromsource_weight, tosink_weight;
        fromsource_weight = Util::calcDistance(m_facet_centers.at(i), m_image_area.tl()) / value_for_normalize;
        tosink_weight = Util::calcDistance(m_facet_centers.at(i), m_image_area.br()) / value_for_normalize;
        m_graph.addTerminalWeights(i, fromsource_weight, tosink_weight);
    }
}

void SiftGraphBuilder::calcNeighborWeights(void) {
    /* 正規化用基準値 */
    double_t value_for_normalize = Util::calcDistance(m_image_area.tl(), m_image_area.br());
    int32_t edgeindex = 0;
    int32_t vertexindex = 0;
    for(size_t i = 0; i < m_facet_centers.size(); i++) {
        /* 始点 */
        vertexindex = m_subdiv.findNearest(m_facet_centers[i]);
        cv::Point2f origin_vertex = m_subdiv.getVertex(vertexindex, &edgeindex);
        /* ドロネー分割のために追加された外部頂点はグラフに含めない */
        if(!origin_vertex.inside(m_image_area))
            continue;

        int32_t firstedge = edgeindex;
        /* 始点を時計回りで終点を探して重みを追加 */
        do {
            cv::Point2f destination_vertex;
            int32_t destination_vertex_index = m_subdiv.edgeDst(edgeindex, &destination_vertex);
            /* ドロネー分割のために追加された外部頂点はグラフに含めない */
            if(!destination_vertex.inside(m_image_area)) {
                edgeindex = m_subdiv.getEdge(edgeindex, cv::Subdiv2D::NEXT_AROUND_ORG);
                continue;
            }
            if(0 <= destination_vertex_index) {
                uint8_t destination_facetcenter;
                /* facetcentersのインデクスがsubdivのインデクスと一致しないので
                 * 最も近いfacetCenterかを確認する
                 */
                double_t mindistance = DBL_MAX;
                for(size_t j = 0; j < m_facet_centers.size(); j++) {
                    if(0.0 == mindistance)
                        break;
                    double_t temp = Util::calcDistance(m_facet_centers[j], destination_vertex);
                    if(mindistance > temp) {
                        mindistance = temp;
                        destination_facetcenter = j;
                    }
                }
                /* graph.addEdges()の中で双方向グラフが作られるので
                 * 終点のインデクス > 始点のインデクスの場合のみ追加する
                 */
                if(destination_facetcenter > i) {
                    double_t weight = Util::calcDistance(origin_vertex, destination_vertex) / value_for_normalize;
                    m_graph.addEdges(i, destination_facetcenter, weight, weight);
                }
            } else {
                LOG(ERROR) << "Vertex index is invalid!!" << std::endl;
            }
            edgeindex = m_subdiv.getEdge(edgeindex, cv::Subdiv2D::NEXT_AROUND_ORG);
        } while(firstedge != edgeindex); /* 最初に戻ってきたら終了 */
    }
}

void SiftGraphBuilder::cutGraph(void) {
    m_graph.maxFlow();
    for(size_t i = 0; i < m_facet_centers.size(); i++) {
        if(m_graph.inSourceSegment(i))
            m_source_points.push_back(m_facet_centers[i]);
        else
            m_sink_points.push_back(m_facet_centers[i]);
    }
}

void SiftGraphBuilder::drawPoints(cv::Mat &image, DrawType type) {
    for(size_t i = 0; i < m_source_points.size(); i++)
        cv::circle(image, m_source_points[i], 4, cv::Scalar(0, 0, 255), -1);
    for(size_t i = 0; i < m_sink_points.size(); i++)
        cv::circle(image, m_sink_points[i], 4, cv::Scalar(255, 0, 0), -1);
}

/*  Log output operator */
google::LogMessage& operator<<(google::LogMessage& lhs, const SiftGraphBuilder& rhs) {
  lhs.stream() << "cvgraphcut_base::Siftgraphbuilder{" <<
      // TODO(N.Takayama): implement out stream of memder data
      "}" << std::endl;
  return lhs;
}

/*--- Accessor --------------------------------------------------------------*/

/*--- Event -----------------------------------------------------------------*/

}  // namespace cvgraphcut_base


