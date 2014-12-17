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

#include <fstream>

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
  for(int32_t i = 0; i < kDefClusters; i++) {
    m_fore_coefficients[i] = 0.0;
    m_back_coefficients[i] = 0.0;
  }
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

  learnVisualWord();

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

  debugPrintMembers();

}

void SiftGraphBuilder::learnVisualWord(void) {
  /* 前景,背景のVisualWordの学習 */
  std::vector<float_t> fore_data;
  std::vector<float_t> back_data;
  for(size_t i = 0; i < m_match_groups.size(); i++) {
    std::vector<SiftMatchPare> group = m_match_groups.at(i);
    for(size_t j = 0; j < group.size(); j++) {
      SiftMatchPare pare = group.at(j);
      switch (i) {
      case 0:
      case 1:
      case 2: {
        for(int32_t k = 0; k < pare.m_source_descriptor.cols; k++) {
          float_t val = pare.m_source_descriptor.at<float_t>(0, k);
          fore_data.push_back(val);
        }
        break;
      }
      case 3:
      case 4:
      case 12: {
        for(int32_t k = 0; k < pare.m_destination_descriptor.cols; k++) {
          float_t val = pare.m_destination_descriptor.at<float_t>(0, k);
          back_data.push_back(val);
        }
        break;
      }
      default:
        break;
      }
    }
  }
  /* ディープコピー */
  int32_t fore_rows = fore_data.size() / 128;
  int32_t back_rows = back_data.size() / 128;
  m_fore_visualword = cv::Mat(fore_rows, 128, CV_32FC1);
  m_back_visualword = cv::Mat(back_rows, 128, CV_32FC1);
  for(int32_t i = 0; i < fore_rows; i++)
    for(int32_t j = 0; j < 128; j++) {
      m_fore_visualword.at<float_t>(i, j) = fore_data[i * 128 + j];
    }
  for(int32_t i = 0; i < back_rows; i++)
    for(int32_t j = 0; j < 128; j++) {
      m_back_visualword.at<float_t>(i, j) = back_data[i * 128 + j];
    }
  m_fore_centroid = cv::Mat(kDefClusters, 128, CV_32FC1);
  m_back_centroid = cv::Mat(kDefClusters, 128, CV_32FC1);
  cv::TermCriteria termcriteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10, 1.0);
  cv::kmeans(m_fore_visualword, kDefClusters, m_fore_labels, termcriteria, 1, cv::KMEANS_PP_CENTERS, m_fore_centroid);
  cv::kmeans(m_back_visualword, kDefClusters, m_back_labels, termcriteria, 1, cv::KMEANS_PP_CENTERS, m_back_centroid);

  /* 混合ガウス分布の係数算出 */
  for(int32_t i = 0; i < m_fore_labels.rows; i++) {
    m_fore_coefficients[m_fore_labels.at<int32_t>(i, 0)]++;
  }
  for(int32_t i = 0; i < kDefClusters; i++) {
    m_fore_coefficients[i] /= m_fore_labels.rows;
  }
  for(int32_t i = 0; i < m_back_labels.rows; i++) {
    m_back_coefficients[m_back_labels.at<int32_t>(i, 0)]++;
  }
  for(int32_t i = 0; i < kDefClusters; i++) {
    m_back_coefficients[i] /= m_back_labels.rows;
  }


}

void SiftGraphBuilder::calcTermWeights(void) {
  double_t value_for_normalize = sqrt(pow(255, 2.0) * 128);

  for(size_t i = 0; i < m_facet_centers.size(); i++) {
    m_graph.addVertex();
    double_t fromsource_weight = 0.0;
    double_t tosink_weight = 0.0;
    std::vector<SiftMatchPare> group = m_match_groups.at(i);
    SiftMatchPare frontpare = group.front();
    switch(i) {
    case 0:
    case 1:
    case 2: {
      fromsource_weight = 10.0;
      tosink_weight = 0.0;
      m_graph.addTerminalWeights(i, fromsource_weight, tosink_weight);
      break;
    }
    case 3:
    case 4:
    case 12: {
      fromsource_weight = 0.0;
      tosink_weight = 10.0;
      m_graph.addTerminalWeights(i, fromsource_weight, tosink_weight);
      break;
    }
    default:
      break;
    }
#if 1   /* 重み付き平均を使用 */
    double_t weight = 0.0;
    double_t tempweight = 0.0;
    for(int32_t k = 0; k < m_fore_centroid.rows; k++) {
      tempweight = 0.0;
      for(int32_t l = 0; l < m_fore_centroid.cols; l++) {
        float_t val_word = m_fore_centroid.at<float_t>(k, l);
        float_t val_front = frontpare.m_source_descriptor.at<float_t>(0, l);
        tempweight += pow(abs(val_word - val_front), 2.0);
      }
      tempweight = sqrt(tempweight) * m_fore_coefficients[m_fore_labels.at<int32_t>(k, 0)];
      weight += tempweight;
    }
    fromsource_weight = weight / value_for_normalize;
    //fromsource_weight = -log(weight / value_for_normalize);
    //fromsource_weight = (value_for_normalize - weight) / value_for_normalize;

    weight = 0.0;
    tempweight = 0.0;
    for(int32_t k = 0; k < m_back_centroid.rows; k++) {
      tempweight = 0.0;
      for(int32_t l = 0; l < m_back_centroid.cols; l++) {
        float_t val_word = m_back_centroid.at<float_t>(k, l);
        float_t val_front = frontpare.m_source_descriptor.at<float_t>(0, l);
        tempweight += pow(abs(val_word - val_front), 2.0);
      }
      tempweight = sqrt(tempweight) * m_back_coefficients[m_back_labels.at<int32_t>(k, 0)];
      weight += tempweight;
    }
    tosink_weight = weight / value_for_normalize;
    //tosink_weight = -log(weight / value_for_normalize);
    //tosink_weight = (value_for_normalize - weight) / value_for_normalize;

#else   // 最小距離を使用
    double_t mindistance = DBL_MAX;
    double_t minlabel = -1;
    for(int32_t k = 0; k < m_fore_centroid.rows; k++) {
      double_t dist = 0.0;
      for(int32_t l = 0; l < m_fore_centroid.cols; l++) {
        float_t val_word = m_fore_centroid.at<float_t>(k, l);
        float_t val_front = frontpare.m_source_descriptor.at<float_t>(0, l);
        dist += pow(abs(val_word - val_front), 2.0);
      }
      dist = sqrt(dist);
      if(mindistance > dist) {
        mindistance = dist;
        minlabel = k;
      }
    }

    if(0 <= minlabel)
      fromsource_weight = mindistance / value_for_normalize;
    else
      fromsource_weight = 0.0;

    mindistance = DBL_MAX;
    minlabel = -1;
    for(int32_t k = 0; k < m_back_centroid.rows; k++) {
      double_t dist = 0.0;
      for(int32_t l = 0; l < m_back_centroid.cols; l++) {
        dist += pow(abs(m_back_centroid.at<float_t>(k, l) - frontpare.m_source_descriptor.at<float_t>(0, l)), 2.0);
      }
      dist = sqrt(dist);
      if(mindistance > dist) {
        mindistance = dist;
        minlabel = k;
      }
    }

    if(0 <= minlabel)
      tosink_weight = mindistance / value_for_normalize;
    else
      tosink_weight = 0.0;
#endif
    m_graph.addTerminalWeights(i, fromsource_weight, tosink_weight);
  }

#if 0 // グラフカットの動作確認用
  /* 正規化用基準値 */
  double_t value_for_normalize = Util::calcDistance(m_image_area.tl(), m_image_area.br());
  for(size_t i = 0; i < m_facet_centers.size(); i++) {
    m_graph.addVertex();
    double_t fromsource_weight, tosink_weight;
    fromsource_weight = Util::calcDistance(m_facet_centers.at(i), m_image_area.tl()) / value_for_normalize;
    tosink_weight = Util::calcDistance(m_facet_centers.at(i), m_image_area.br()) / value_for_normalize;
    m_graph.addTerminalWeights(i, fromsource_weight, tosink_weight);
  }
#endif
}

void SiftGraphBuilder::calcNeighborWeights(void) {

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
          double_t temp = Util::calcDistance<float_t>(m_facet_centers[j], destination_vertex);
          if(mindistance > temp) {
            mindistance = temp;
            destination_facetcenter = j;
          }
        }
        /* graph.addEdges()の中で双方向グラフが作られるので
         * 終点のインデクス > 始点のインデクスの場合のみ追加する
         */
        if(destination_facetcenter > i && m_match_groups[destination_facetcenter][0].is_matched) {
#if 0               /* 二点間のユークリッド距離を使用 */
          /* 正規化用基準値 */
          double_t value_for_normalize = Util::calcDistance(m_image_area.tl(), m_image_area.br());
          //double_t weight = (value_for_normalize - Util::calcDistance(origin_vertex, destination_vertex)) / value_for_normalize;
          double_t weight = Util::calcDistance(origin_vertex, destination_vertex) / value_for_normalize;
#endif
#if 1
          /* Sift特徴量を使用*/
          double_t weight = 0.0;
          double_t value_for_normalize = sqrt(pow(255, 2.0) * 128);
          cv::Mat origin_feature;
          cv::Mat destination_feature;
          /* 先頭ペアのみ取り出す */
#if 0                /* Sift特徴差を使用 */
          origin_feature = m_match_groups[i][0].m_source_descriptor;
          destination_feature = m_match_groups[destination_facetcenter][0].m_source_descriptor;
#else               /* マッチング画像からの変化量の差を使用 */
          origin_feature = m_match_groups[i][0].m_difference_descriptor;
          destination_feature = m_match_groups[destination_facetcenter][0].m_difference_descriptor;
#endif
          /* L2-norm */
          for(int32_t j = 0; j < origin_feature.cols; j++) {
            weight += pow(abs(origin_feature.at<float_t>(0, j) - destination_feature.at<float_t>(0, j)), 2.0);
          }

          weight = sqrt(weight) / value_for_normalize;
          //weight = (value_for_normalize - sqrt(weight)) / value_for_normalize;
          //weight = -log(sqrt(weight) / value_for_normalize);
          double_t debug = abs(m_match_groups[i][0].m_distance - m_match_groups[destination_facetcenter][0].m_distance);
          debug /= value_for_normalize;
#endif
          m_graph.addEdges(i, destination_facetcenter, weight, weight);
          //m_graph.addEdges(i, destination_facetcenter, debug, debug);

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

void SiftGraphBuilder::drawCuttedPoints(cv::Mat &image, DrawType type) {
  if(DRAW_BOTH == type || DRAW_SOURCE == type) {
    for(size_t i = 0; i < m_source_points.size(); i++)
      cv::circle(image, m_source_points[i], 4, cv::Scalar(0, 0, 255), -1);
  }
  if(DRAW_BOTH == type || DRAW_SINK == type) {
    for(size_t i = 0; i < m_sink_points.size(); i++)
      cv::circle(image, m_sink_points[i], 4, cv::Scalar(255, 0, 0), -1);
  }
}

void SiftGraphBuilder::drawGraphs(cv::Mat &image, DrawDivType type) {
  if(DRAW_DIV_DELAUNAY == type) {
    std::vector<cv::Vec4f>::iterator delaunay_edge;
    for(delaunay_edge = m_delaunay_edges.begin(); delaunay_edge != m_delaunay_edges.end(); delaunay_edge++) {
      cv::Point origin(delaunay_edge->val[0], delaunay_edge->val[1]);
      cv::Point destination(delaunay_edge->val[2], delaunay_edge->val[3]);
      /* 頂点が画像内の場合のみ描画 */
      if(origin.inside(m_image_area) && destination.inside(m_image_area))
        cv::line(image, origin, destination, cv::Scalar(0, 200, 0));
    }
    for(delaunay_edge = m_delaunay_edges.begin(); delaunay_edge != m_delaunay_edges.end(); delaunay_edge++) {
      cv::Point origin(delaunay_edge->val[0], delaunay_edge->val[1]);
      /* 頂点が画像内の場合のみ描画 */
      if(origin.inside(m_image_area))
        cv::circle(image, origin, 4, cv::Scalar(0, 255, 0), -1);
    }
  }
  if(DRAW_DIV_VORONOY == type) {
    for(size_t i = 0; i < m_facet_centers.size(); i++) {
      std::vector<cv::Point2f> facet_lists = m_facet_lists.at(i);
      cv::Point2f origin = m_facet_centers.at(i);
      for(size_t j = 0; j < facet_lists.size(); j++) {
        cv::Point2f destination = facet_lists.at(j);
        /* 頂点が画像内の場合のみ描画 */
        if(origin.inside(m_image_area) && destination.inside(m_image_area))
          cv::line(image, origin, destination, cv::Scalar(0, 200, 0));
      }
    }
    for(size_t i = 0; i < m_facet_centers.size(); i++) {
      cv::Point2f origin = m_facet_centers.at(i);
      /* 頂点が画像内の場合のみ描画 */
      if(origin.inside(m_image_area))
        cv::circle(image, origin, 4, cv::Scalar(0, 255, 0), -1);
    }
  }
}

void SiftGraphBuilder::drawSiftGroups(cv::Mat &image, int32_t index, cv::Scalar color) {
  if(m_match_groups.size() <= index || index < 0)
    return;
  std::vector<SiftMatchPare> group = m_match_groups.at(index);
  SiftMatchPare frontpare = group.at(0);
  cv::Point frontpoint = frontpare.m_source_key.pt;
  for(size_t i = 1; i < group.size(); i++) {
    SiftMatchPare pare = group.at(i);
    cv::Point point = pare.m_source_key.pt;
    cv::line(image, frontpoint, point, color);
    cv::circle(image, point, 5, color, -1);
  }
  /* 代表点は最後に描画 */
  cv::circle(image, frontpoint, 5, color, -1);
}


/* Log output for array member */
void SiftGraphBuilder::debugPrintMembers(void) {
  /* front pare of m_match_groups */
  if(!m_match_groups.empty()) {
    std::ofstream output;
    output.open("log_m_match_groups.txt");
    if(output.is_open()) {
      for(size_t i = 0; i < m_match_groups.size(); i++) {
        std::vector<SiftMatchPare> group = m_match_groups.at(i);
        SiftMatchPare frontpare = group.front();
        if(frontpare.is_matched) {
          cv::Point2f point = frontpare.m_source_key.pt;
          output << "Index_" << i << ":(" << point.x << "," << point.y << ")" << std::endl;
        }
      }
    }
    output.close();
  }

  /* m_delaunay_points */
  if(!m_delaunay_points.empty()) {
    std::ofstream output;
    output.open("log_m_delaunay_points.txt");
    if(output.is_open()) {
      for(size_t i = 0; i < m_delaunay_points.size(); i++) {
        cv::Point2f point = m_delaunay_points.at(i);
        output << "Index_" << i << ":(" << point.x << "," << point.y << ")" << std::endl;
      }
    }
    output.close();
  }
  /* m_delaunay_edges */
  if(!m_delaunay_edges.empty()) {
    std::ofstream output;
    output.open("log_m_delaunay_edges.txt");
    if(output.is_open()) {
      for(size_t i = 0; i < m_delaunay_edges.size(); i++) {
        cv::Vec4f edge = m_delaunay_edges.at(i);
        output << "Index_" << i << ":(" << edge.val[0] << "," << edge.val[1] << ")"
               << "-> (" << edge.val[2] << "," << edge.val[3] << ")" << std::endl;
      }
    }
    output.close();
  }
  /* m_facet_index */
  if(!m_facet_index.empty()) {
    std::ofstream output;
    output.open("log_m_facet_index.txt");
    if(output.is_open()) {
      for(size_t i = 0; i < m_facet_index.size(); i++) {
        int32_t facet_index = m_facet_index.at(i);
        output << "Index_" << i << ":" << facet_index << std::endl;
      }
    }
    output.close();
  }
  /* m_facet_lists */
  if(!m_facet_lists.empty()) {
    std::ofstream output;
    output.open("log_m_facet_lists.txt");
    if(output.is_open()) {
      for(size_t i = 0; i < m_facet_lists.size(); i++) {
        std::vector<cv::Point2f> voronoivertexes = m_facet_lists.at(i);
        for(size_t j = 0; j < voronoivertexes.size(); j++) {
          cv::Point2f point = voronoivertexes.at(j);
          output << "Index_(" << i << "," << j << "):"
                 << "(" << point.x << "," << point.y << ")" << std::endl;
        }
      }
    }
  }
  /* m_facet_centers */
  if(!m_facet_centers.empty()) {
    std::ofstream output;
    output.open("log_m_facet_centers.txt");
    if(output.is_open()) {
      for(size_t i = 0; i < m_facet_centers.size(); i++) {
        cv::Point2f point = m_facet_centers.at(i);
        output << "Index_" << i << ":(" << point.x << "," << point.y << ")" << std::endl;
      }
    }
    output.close();
  }
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


