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

static const int32_t kDefItterationCount = 1;

/* エッジの重みを何に基づいて求めるか */
typedef enum WhichDescriptorUse_TAG {
  DIFF_OF_SOURCE,   /* ターゲット画像のSIFT特徴量 */
  DIFF_OF_DST,      /* 比較画像のSIFT特徴量 */
  DIFF_OF_DIFF      /* ターゲットと比較画像のSIFT特徴量の差 */
} WhichDescriptorUse;

static const WhichDescriptorUse kDefWhichUseForNeighbor = DIFF_OF_DIFF;  /* 隣接エッジの重み */
static const WhichDescriptorUse kDefWhichUseForTerminal = DIFF_OF_DIFF;  /* SOURCE, SINKへの重み */

/* GMMの学習サンプルの与え方 */
typedef enum TrainingMode_TAG {
  MODE_SOFTDEFINE,        /* エリア指定されていない箇所は前景,背景双方に学習する */
  MODE_FORE_HARDDEFINE,   /* 前景 : エリア指定箇所, 背景予想 : その他, GrabCutと同様*/
  MODE_BOTH_HARDDEFINE    /* 前景,背景 : エリア指定箇所, その他 : 学習しない */
} TrainingMode;

static const TrainingMode kDefTrainingMode = MODE_BOTH_HARDDEFINE;

/* エッジの重みの求め方 */
typedef enum WeightCalculationMethod_TAG {
  METHODE_LOGLIKELIHOOD, /* 対数尤度 */
  METHODE_FEATUREDISTANCE, /* 特徴量の距離 */
  METHODE_LOGLIKELIHOOD_XYDISTANCE, /* 対数尤度 + 特徴点間の距離 */
  METHODE_FEATUREDISTANCE_XYDISTANCE /* 特徴量の距離 + 特徴点間の距離 */
} WeightCalculationMethod;

static const WeightCalculationMethod kDefWeightCalculationMethod = METHODE_FEATUREDISTANCE_XYDISTANCE;

/*--- Constructor / Destructor / Initialize ---------------------------------*/

/* Defoult constructor */
SiftGraphBuilder::SiftGraphBuilder(std::vector<std::vector<SiftMatchPare> >& match_groups, std::vector<SiftMatchPare> &unmatch_group, cv::Rect image_area, cv::Rect fore_area, cv::Rect back_area)
  : m_match_groups(match_groups),
    m_unmatch_group(unmatch_group),
    m_graph(GCGraph()),
    m_image_area(image_area),
    m_fore_area(fore_area),
    m_back_area(back_area) {
  m_delaunay_points.clear();
  m_delaunay_edges.clear();
  m_facet_lists.clear();
  m_facet_centers.clear();

  m_source_points.clear();
  m_sink_points.clear();
}

/* Default destructor */
SiftGraphBuilder::~SiftGraphBuilder(void) {
  m_fore_visualword.release();
  m_back_visualword.release();
}

/*--- Operation -------------------------------------------------------------*/
void SiftGraphBuilder::build(void) {
  /* DebugLog
  for(size_t i = 0; i < m_match_groups.size(); i++){
      cv::Mat vector = m_match_groups[i][0].m_source_descriptor;
      std::string logfile = "initvector_" + std::to_string(i) + ".cvs";
      Util::logCvMatAsCvs<double_t>(logfile, vector);
  }
  */

  buildSubdivGraph();

  /* DebugLog
  for(size_t i = 0; i < m_siftvertices.size(); i++){
      SiftVertex siftvertex = m_siftvertices.at(i);
      cv::Mat vector = m_match_groups[siftvertex.m_vertexid][0].m_source_descriptor;
      std::string logfile = "aftervector_" + std::to_string(siftvertex.m_vertexid) + ".cvs";
      Util::logCvMatAsCvs<double_t>(logfile, vector);
  }
  */

  if(MODE_SOFTDEFINE == kDefTrainingMode)
    initVisualWord();
  else if(MODE_FORE_HARDDEFINE == kDefTrainingMode)
    initVisualWord2();
  else if(MODE_BOTH_HARDDEFINE == kDefTrainingMode) {
    //initVisualWord3();
    initForeVisualWord3();
    initBackVisualWord3();
  }

  /* DebugLog 初期のアサイン結果 */
  LOG(INFO) << "Initialize" << std::endl;
  for(size_t i = 0; i < m_siftvertices.size(); i++) {
    SiftVertex siftvertex = m_siftvertices.at(i);
    if(SiftVertex::STATE_BACKGROUND == siftvertex.m_state)
      LOG(INFO) << "Vertex " << i << " is " << "BACKGROUND." << std::endl;
    else if(SiftVertex::STATE_FOREGROUND == siftvertex.m_state)
      LOG(INFO) << "Vertex " << i << " is " << "FOREGROUND." << std::endl;
    else if(SiftVertex::STATE_PROB_BACKGROUND == siftvertex.m_state)
      LOG(INFO) << "Vertex " << i << " is " << "PROB_BACKGROUND." << std::endl;
    else if(SiftVertex::STATE_PROB_FOREGROUND == siftvertex.m_state)
      LOG(INFO) << "Vertex " << i << " is " << "PROB_FOREGROUND." << std::endl;
  }

  //m_gamma = 10.0;
  LOG(INFO) << "Gamma : " << m_gamma << std::endl;
  m_lambda = 1.0;
  calcBeta();
  LOG(INFO) << "Beta : " << m_beta << std::endl;
  calcNeighborWeights();

  for(int32_t i = 0; i < kDefItterationCount; i++) {
    GCGraph graph;
    if(MODE_SOFTDEFINE == kDefTrainingMode) {
      assignGmmComponents();
      learnVisualWord();
      constructGcGraph(graph);
    } else if(MODE_FORE_HARDDEFINE == kDefTrainingMode) {
      assignGmmComponents2();
      learnVisualWord2();
    } else if(MODE_BOTH_HARDDEFINE == kDefTrainingMode) {
      assignGmmComponents3();
      learnVisualWord3();
    }
    constructGcGraph(graph);

    /* グラフ情報表示 */
    LOG(INFO) << "Itteration : " << i << std::endl;
    for(size_t i = 0; i < m_siftvertices.size(); i++) {
      LOG(INFO) << "Vertex : " << i << std::endl;
      LOG(INFO) << "FromSource : " << m_siftvertices[i].m_fromsource_weight << std::endl;
      LOG(INFO) << "ToSink : " << m_siftvertices[i].m_tosink_weight << std::endl;
      std::string message;
      for(size_t j = 0; j < m_siftvertices[i].m_dstverticesid.size(); j++) {
        message += "To " + std::to_string(m_siftvertices[i].m_dstverticesid[j]) + ":" + std::to_string(m_siftvertices[i].m_neighbor_weights[j]) + ",";
      }
      message += "\n";
      LOG(INFO) << message;
    }

    estimateSegmentation(graph);
  }

  //calcTermWeights();
}


void SiftGraphBuilder::buildSubdivGraph(void) {
  //m_subdiv.initDelaunay(m_image_area);

  /* マッチング率が一番高いもののみ
   * TODO モードかフラグで他の対応点を使用するか切り替える
   */
  //for(size_t i = 0; i < m_match_groups.size(); i++) {
  //  std::vector<SiftMatchPare> group = m_match_groups.at(i);
  //  SiftMatchPare frontpare = group.front();
  //  if(frontpare.is_matched)
  //    m_delaunay_points.push_back(frontpare.m_source_key.pt);
  //}
  //m_subdiv.insert(m_delaunay_points);
  //m_subdiv.getVoronoiFacetList(m_facet_index, m_facet_lists, m_facet_centers);
  //m_subdiv.getEdgeList(m_delaunay_edges);

  /* マッチング率が一番高いもののみ
   * TODO モードかフラグで他の対応点を使用するか切り替える
   */
  std::vector<cv::Point2f> delaunay_points;
  std::vector<cv::Vec4f> delaunay_edges;
  for(size_t i = 0; i < m_match_groups.size(); i++) {
    std::vector<SiftMatchPare> group = m_match_groups.at(i);
    SiftMatchPare frontpare = group.front();
    if(frontpare.is_matched)
      delaunay_points.push_back(frontpare.m_source_key.pt);
  }

  cv::Subdiv2D subdiv;
  std::vector<int32_t> facet_index;
  std::vector<std::vector<cv::Point2f> > facet_lists;
  std::vector<cv::Point2f> facet_centers;
  subdiv.initDelaunay(m_image_area);
  subdiv.insert(delaunay_points);
  subdiv.getVoronoiFacetList(facet_index, facet_lists, facet_centers);
  subdiv.getEdgeList(delaunay_edges);


  /* 描画処理のためにコピー */
  for(size_t i = 0; i < facet_lists.size(); i++)
    m_facet_lists.push_back(facet_lists[i]);
  for(size_t i = 0; i < facet_centers.size(); i++)
    m_facet_centers.push_back(facet_centers[i]);
  for(size_t i = 0; i < delaunay_points.size(); i++)
    m_delaunay_points.push_back(delaunay_points[i]);
  for(size_t i = 0; i < delaunay_edges.size(); i++)
    m_delaunay_edges.push_back(delaunay_edges[i]);

  m_vertexcount = facet_centers.size();
  m_edgecount = delaunay_edges.size();

  /* SiftGraphの構築 */
  int32_t edgeindex = 0;
  int32_t firstedgeindex = 0;
  int32_t vertexindex = 0;
  for(size_t i = 0; i < facet_centers.size(); i++) {
    /* 始点 */
    vertexindex = subdiv.findNearest(facet_centers[i]);
    cv::Point2f origin_vertex = subdiv.getVertex(vertexindex, &edgeindex);
    /* ドロネー分割のために追加された外部頂点はグラフに含めない */
    if(!origin_vertex.inside(m_image_area))
      continue;
    int32_t firstedge = firstedgeindex = edgeindex;

    SiftVertex origin_siftvertex;
    origin_siftvertex.m_vertexid = i;
    origin_siftvertex.m_state = SiftVertex::STATE_DEFAULT;

    /* 既に登録済みかチェック */
    bool check = false;
    for(size_t j = 0; j < m_siftvertices.size(); j++) {
      if( m_siftvertices[j].m_vertexid == vertexindex ) {
        check = true;
        break;
      }
    }
    if(!check) {
      m_siftvertices.push_back(origin_siftvertex);
    }
    /* 接続先の頂点を登録しつつグラフを構築 */
    do {
      cv::Point2f destination_vertex;
      int32_t destination_vertex_index = subdiv.edgeDst(edgeindex, &destination_vertex);
      /* ドロネー分割のために追加された外部頂点はグラフに含めない */
      if(!destination_vertex.inside(m_image_area)) {
        edgeindex = subdiv.getEdge(edgeindex, cv::Subdiv2D::NEXT_AROUND_ORG);
        continue;
      }
      if(0 <= destination_vertex_index) {
        uint8_t destination_facetcenter;
        /* facetcentersのインデクスがsubdivのインデクスと一致しないので
         * 最も近いfacetCenterかを確認する
         */
        double_t mindistance = DBL_MAX;
        for(size_t j = 0; j < facet_centers.size(); j++) {
          if(0.0 == mindistance)
            break;
          double_t temp = Util::calcDistance<float_t>(facet_centers[j], destination_vertex);
          if(mindistance > temp) {
            mindistance = temp;
            destination_facetcenter = j;
          }
        }
        /* 始点に戻ってくるエッジは無視 */
        if(vertexindex == destination_facetcenter) {
          edgeindex = subdiv.getEdge(edgeindex, cv::Subdiv2D::NEXT_AROUND_ORG);
          continue;
        }

        /* graph.addEdges()の中で双方向グラフが作られるので
         * 終点のインデクス > 始点のインデクスの場合のみ追加する
         */
        if(destination_facetcenter > i && m_match_groups[destination_facetcenter][0].is_matched) {
          m_siftvertices[i].m_dstverticesid.push_back(destination_facetcenter);
        }
      } else {
        LOG(ERROR) << "Vertex index is invalid!!" << std::endl;
      }
      edgeindex = subdiv.getEdge(edgeindex, cv::Subdiv2D::NEXT_AROUND_ORG);
    } while(firstedge != edgeindex); /* 最初に戻ってきたら終了 */
  }
}

void SiftGraphBuilder::initVisualWord(void) {
  const int32_t kMeansItterationCount = 10;
  const int32_t kMeansType = cv::KMEANS_PP_CENTERS;

  /* 前景,背景のVisualWordの学習
   * cv::kmeansはfloat型のMatしか受け付けないのでコンバート
   */
  std::vector<float_t> fore_data;
  std::vector<float_t> back_data;
  for(size_t i = 0; i < m_siftvertices.size(); i++) {
    SiftVertex siftvertex = m_siftvertices.at(i);
    SiftMatchPare frontpare = m_match_groups[siftvertex.m_vertexid].front();
    if(frontpare.m_source_key.pt.inside(m_fore_area)) {
      m_siftvertices[i].m_state = SiftVertex::STATE_FOREGROUND;
      for(size_t j = 0; j < m_match_groups[siftvertex.m_vertexid].size(); j++) {
        SiftMatchPare pare = m_match_groups[siftvertex.m_vertexid].at(j);
        for(int32_t k = 0; k < pare.m_source_descriptor.cols; k++) {
          float_t value = 0.0;
          if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_source_descriptor.at<double_t>(0, k));
          else if (DIFF_OF_DST == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_destination_descriptor.at<double_t>(0, k));
          else if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_difference_descriptor.at<double_t>(0, k));
          fore_data.push_back(value);
        }
      }
    } else if(frontpare.m_source_key.pt.inside(m_back_area)) {
      m_siftvertices[i].m_state = SiftVertex::STATE_BACKGROUND;
      for(size_t j = 0; j < m_match_groups[siftvertex.m_vertexid].size(); j++) {
        SiftMatchPare pare = m_match_groups[siftvertex.m_vertexid].at(j);
        for(int32_t k = 0; k < pare.m_source_descriptor.cols; k++) {
          float_t value = 0.0;
          if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_source_descriptor.at<double_t>(0, k));
          if(DIFF_OF_DST == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_destination_descriptor.at<double_t>(0, k));
          if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_difference_descriptor.at<double_t>(0, k));
          back_data.push_back(value);
        }
      }
    } else {
      m_siftvertices[i].m_state = SiftVertex::STATE_PROB_BACKGROUND;
      for(size_t j = 0; j < m_match_groups[siftvertex.m_vertexid].size(); j++) {
        SiftMatchPare pare = m_match_groups[siftvertex.m_vertexid].at(j);
        for(int32_t k = 0; k < pare.m_source_descriptor.cols; k++) {
          float_t value = 0.0;
          if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_source_descriptor.at<double_t>(0, k));
          if(DIFF_OF_DST == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_destination_descriptor.at<double_t>(0, k));
          if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_difference_descriptor.at<double_t>(0, k));
          fore_data.push_back(value);
          back_data.push_back(value);
        }
      }
    }
  }
  /* ディープコピー */
  int32_t fore_rows = fore_data.size() / 128;
  int32_t back_rows = back_data.size() / 128;
  m_fore_samples = cv::Mat(fore_rows, 128, CV_32FC1);
  m_back_samples = cv::Mat(back_rows, 128, CV_32FC1);
  for(int32_t i = 0; i < fore_rows; i++)
    for(int32_t j = 0; j < 128; j++) {
      m_fore_samples.at<float_t>(i, j) = fore_data[i * 128 + j];
    }
  for(int32_t i = 0; i < back_rows; i++)
    for(int32_t j = 0; j < 128; j++) {
      m_back_samples.at<float_t>(i, j) = back_data[i * 128 + j];
    }
  cv::TermCriteria termcriteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, kMeansItterationCount, 1.0);
  cv::kmeans(m_fore_samples, kDefClusters, m_fore_labels, termcriteria, 1, kMeansType);
  cv::kmeans(m_back_samples, kDefClusters, m_back_labels, termcriteria, 1, kMeansType);

  m_fore_gmm.initLearning();

  for(int32_t i = 0; i < m_fore_samples.rows; i++) {
    cv::Mat samplef = m_fore_samples.row(i);
    cv::Mat sampled;
    samplef.convertTo(sampled, CV_64FC1);
    m_fore_gmm.addSample(m_fore_labels.at<int32_t>(i, 0), sampled);
  }

  m_fore_gmm.endLearning();

  m_back_gmm.initLearning();
  for(int32_t i = 0; i < m_back_samples.rows; i++) {
    cv::Mat samplef = m_back_samples.row(i);
    cv::Mat sampled;
    samplef.convertTo(sampled, CV_64FC1);
    m_back_gmm.addSample(m_back_labels.at<int32_t>(i, 0), sampled);
  }
  m_back_gmm.endLearning();
}

void SiftGraphBuilder::initVisualWord2(void) {
  const int32_t kMeansItterationCount = 10;
  const int32_t kMeansType = cv::KMEANS_PP_CENTERS;

  /* 前景,背景のVisualWordの学習
   * cv::kmeansはfloat型のMatしか受け付けないのでコンバート
   */
  std::vector<float_t> fore_data;
  std::vector<float_t> back_data;
  for(size_t i = 0; i < m_siftvertices.size(); i++) {
    SiftVertex siftvertex = m_siftvertices.at(i);
    SiftMatchPare frontpare = m_match_groups[siftvertex.m_vertexid].front();
    if(frontpare.m_source_key.pt.inside(m_fore_area)) {
      m_siftvertices[i].m_state = SiftVertex::STATE_FOREGROUND;
      for(size_t j = 0; j < m_match_groups[siftvertex.m_vertexid].size(); j++) {
        SiftMatchPare pare = m_match_groups[siftvertex.m_vertexid].at(j);
        for(int32_t k = 0; k < pare.m_source_descriptor.cols; k++) {
          float_t value = 0.0;
          if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_source_descriptor.at<double_t>(0, k));
          else if (DIFF_OF_DST == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_destination_descriptor.at<double_t>(0, k));
          else if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_difference_descriptor.at<double_t>(0, k));
          fore_data.push_back(value);
        }
      }
    } else {
      m_siftvertices[i].m_state = SiftVertex::STATE_PROB_BACKGROUND;
      for(size_t j = 0; j < m_match_groups[siftvertex.m_vertexid].size(); j++) {
        SiftMatchPare pare = m_match_groups[siftvertex.m_vertexid].at(j);
        for(int32_t k = 0; k < pare.m_source_descriptor.cols; k++) {
          float_t value = 0.0;
          if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_source_descriptor.at<double_t>(0, k));
          if(DIFF_OF_DST == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_destination_descriptor.at<double_t>(0, k));
          if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_difference_descriptor.at<double_t>(0, k));
          back_data.push_back(value);
        }
      }
    }
  }
  /* ディープコピー */
  int32_t fore_rows = fore_data.size() / 128;
  int32_t back_rows = back_data.size() / 128;
  m_fore_samples = cv::Mat(fore_rows, 128, CV_32FC1);
  m_back_samples = cv::Mat(back_rows, 128, CV_32FC1);
  for(int32_t i = 0; i < fore_rows; i++)
    for(int32_t j = 0; j < 128; j++) {
      m_fore_samples.at<float_t>(i, j) = fore_data[i * 128 + j];
    }
  for(int32_t i = 0; i < back_rows; i++)
    for(int32_t j = 0; j < 128; j++) {
      m_back_samples.at<float_t>(i, j) = back_data[i * 128 + j];
    }
  cv::TermCriteria termcriteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, kMeansItterationCount, 1.0);
  cv::kmeans(m_fore_samples, kDefClusters, m_fore_labels, termcriteria, 1, kMeansType);
  cv::kmeans(m_back_samples, kDefClusters, m_back_labels, termcriteria, 1, kMeansType);

  m_fore_gmm.initLearning();

  for(int32_t i = 0; i < m_fore_samples.rows; i++) {
    cv::Mat samplef = m_fore_samples.row(i);
    cv::Mat sampled;
    samplef.convertTo(sampled, CV_64FC1);
    m_fore_gmm.addSample(m_fore_labels.at<int32_t>(i, 0), sampled);
  }

  m_fore_gmm.endLearning();

  m_back_gmm.initLearning();
  for(int32_t i = 0; i < m_back_samples.rows; i++) {
    cv::Mat samplef = m_back_samples.row(i);
    cv::Mat sampled;
    samplef.convertTo(sampled, CV_64FC1);
    m_back_gmm.addSample(m_back_labels.at<int32_t>(i, 0), sampled);
  }
  m_back_gmm.endLearning();
}

void SiftGraphBuilder::initVisualWord3(void) {
  const int32_t kMeansItterationCount = 10;
  const int32_t kMeansType = cv::KMEANS_PP_CENTERS;

  /* 前景,背景のVisualWordの学習
   * cv::kmeansはfloat型のMatしか受け付けないのでコンバート
   */
  std::vector<float_t> fore_data;
  std::vector<float_t> back_data;
  for(size_t i = 0; i < m_siftvertices.size(); i++) {
    SiftVertex siftvertex = m_siftvertices.at(i);
    SiftMatchPare frontpare = m_match_groups[siftvertex.m_vertexid].front();
    if(frontpare.m_source_key.pt.inside(m_fore_area)) {
      m_siftvertices[i].m_state = SiftVertex::STATE_FOREGROUND;
      for(size_t j = 0; j < m_match_groups[siftvertex.m_vertexid].size(); j++) {
        SiftMatchPare pare = m_match_groups[siftvertex.m_vertexid].at(j);
        for(int32_t k = 0; k < pare.m_source_descriptor.cols; k++) {
          float_t value = 0.0;
          if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_source_descriptor.at<double_t>(0, k));
          else if (DIFF_OF_DST == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_destination_descriptor.at<double_t>(0, k));
          else if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_difference_descriptor.at<double_t>(0, k));
          fore_data.push_back(value);
        }
      }
    } else if(frontpare.m_source_key.pt.inside(m_back_area)) {
      m_siftvertices[i].m_state = SiftVertex::STATE_BACKGROUND;
      for(size_t j = 0; j < m_match_groups[siftvertex.m_vertexid].size(); j++) {
        SiftMatchPare pare = m_match_groups[siftvertex.m_vertexid].at(j);
        for(int32_t k = 0; k < pare.m_source_descriptor.cols; k++) {
          float_t value = 0.0;
          if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_source_descriptor.at<double_t>(0, k));
          if(DIFF_OF_DST == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_destination_descriptor.at<double_t>(0, k));
          if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_difference_descriptor.at<double_t>(0, k));
          back_data.push_back(value);
        }
      }
    }
  }
  /* ディープコピー */
  int32_t fore_rows = fore_data.size() / 128;
  int32_t back_rows = back_data.size() / 128;
  m_fore_samples = cv::Mat(fore_rows, 128, CV_32FC1);
  m_back_samples = cv::Mat(back_rows, 128, CV_32FC1);
  for(int32_t i = 0; i < fore_rows; i++)
    for(int32_t j = 0; j < 128; j++) {
      m_fore_samples.at<float_t>(i, j) = fore_data[i * 128 + j];
    }
  for(int32_t i = 0; i < back_rows; i++)
    for(int32_t j = 0; j < 128; j++) {
      m_back_samples.at<float_t>(i, j) = back_data[i * 128 + j];
    }
  cv::TermCriteria termcriteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, kMeansItterationCount, 1.0);
  cv::kmeans(m_fore_samples, kDefClusters, m_fore_labels, termcriteria, 1, kMeansType);
  cv::kmeans(m_back_samples, kDefClusters, m_back_labels, termcriteria, 1, kMeansType);

  m_fore_gmm.initLearning();

  for(int32_t i = 0; i < m_fore_samples.rows; i++) {
    cv::Mat samplef = m_fore_samples.row(i);
    cv::Mat sampled;
    samplef.convertTo(sampled, CV_64FC1);
    m_fore_gmm.addSample(m_fore_labels.at<int32_t>(i, 0), sampled);
  }

  m_fore_gmm.endLearning();

  m_back_gmm.initLearning();
  for(int32_t i = 0; i < m_back_samples.rows; i++) {
    cv::Mat samplef = m_back_samples.row(i);
    cv::Mat sampled;
    samplef.convertTo(sampled, CV_64FC1);
    m_back_gmm.addSample(m_back_labels.at<int32_t>(i, 0), sampled);
  }
  m_back_gmm.endLearning();
}

void SiftGraphBuilder::initForeVisualWord3(void) {
  const int32_t kMeansItterationCount = 10;
  const int32_t kMeansType = cv::KMEANS_PP_CENTERS;

  /* 前景のVisualWordの学習
   * cv::kmeansはfloat型のMatしか受け付けないのでコンバート
   */
  std::vector<float_t> fore_data;
  for(size_t i = 0; i < m_siftvertices.size(); i++) {
    SiftVertex siftvertex = m_siftvertices.at(i);
    SiftMatchPare frontpare = m_match_groups[siftvertex.m_vertexid].front();
    if(frontpare.m_source_key.pt.inside(m_fore_area)) {
      m_siftvertices[i].m_state = SiftVertex::STATE_FOREGROUND;
      for(size_t j = 0; j < m_match_groups[siftvertex.m_vertexid].size(); j++) {
        SiftMatchPare pare = m_match_groups[siftvertex.m_vertexid].at(j);
        for(int32_t k = 0; k < pare.m_source_descriptor.cols; k++) {
          float_t value = 0.0;
          if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_source_descriptor.at<double_t>(0, k));
          else if (DIFF_OF_DST == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_destination_descriptor.at<double_t>(0, k));
          else if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_difference_descriptor.at<double_t>(0, k));
          fore_data.push_back(value);
        }
      }
    }
  }
  /* ディープコピー */
  int32_t fore_rows = fore_data.size() / 128;
  m_fore_samples = cv::Mat(fore_rows, 128, CV_32FC1);
  for(int32_t i = 0; i < fore_rows; i++)
    for(int32_t j = 0; j < 128; j++) {
      m_fore_samples.at<float_t>(i, j) = fore_data[i * 128 + j];
    }
  cv::TermCriteria termcriteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, kMeansItterationCount, 1.0);
  cv::kmeans(m_fore_samples, kDefClusters, m_fore_labels, termcriteria, 1, kMeansType, m_fore_visualword);

  m_fore_gmm.initLearning();
  for(int32_t i = 0; i < m_fore_samples.rows; i++) {
    cv::Mat samplef = m_fore_samples.row(i);
    cv::Mat sampled;
    samplef.convertTo(sampled, CV_64FC1);
    m_fore_gmm.addSample(m_fore_labels.at<int32_t>(i, 0), sampled);
  }
  m_fore_gmm.endLearning();

  /* 前景サンプルとビジュアルワードの最大距離 */
  m_fore_maxfeaturedifference = 0.0;
  for(int32_t i = 0; i < m_fore_samples.rows; i++) {
    cv::Mat samplef = m_fore_samples.row(i);
    cv::Mat visualword = m_fore_visualword.row(m_fore_labels.at<int32_t>(i, 0));
    double_t diff = Util::calcL2NormOfVectors<float_t>(samplef, visualword, true);
    if(m_fore_maxfeaturedifference < diff)
      m_fore_maxfeaturedifference = diff;
  }
}

void SiftGraphBuilder::initBackVisualWord3(void) {
  const int32_t kMeansItterationCount = 10;
  const int32_t kMeansType = cv::KMEANS_PP_CENTERS;

  /* 前景のVisualWordの学習
   * cv::kmeansはfloat型のMatしか受け付けないのでコンバート
   */
  std::vector<float_t> back_data;
  for(size_t i = 0; i < m_siftvertices.size(); i++) {
    SiftVertex siftvertex = m_siftvertices.at(i);
    SiftMatchPare frontpare = m_match_groups[siftvertex.m_vertexid].front();
    if(SiftVertex::STATE_FOREGROUND == m_siftvertices[i].m_state)
      continue;
    /* 前景のビジュアルワードとの距離が一定以上のものを背景サンプルとする */
    cv::Mat sample;
    if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
      sample = frontpare.m_source_descriptor;
    else if(DIFF_OF_DST == kDefWhichUseForTerminal)
      sample = frontpare.m_destination_descriptor;
    else if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
      sample = frontpare.m_difference_descriptor;
    double_t mindifference = DBL_MAX;
    for(int32_t j = 0; j < m_fore_visualword.rows; j++) {
      cv::Mat fore_visualwordf = m_fore_visualword.row(j);
      cv::Mat fore_visualwordd;
      fore_visualwordf.convertTo(fore_visualwordd, CV_64FC1);
      double_t diff = Util::calcL2NormOfVectors<double_t>(sample, fore_visualwordd, true);
      if(mindifference > diff)
        mindifference = diff;
    }
    double_t areadistance = Util::calcDistance<int32_t>(m_fore_area.tl(), m_fore_area.br());
    cv::Point areacenter = (m_fore_area.tl() + m_fore_area.br()) / 2;
    cv::Point samplepoint = m_match_groups[siftvertex.m_vertexid].front().m_source_key.pt;
    double_t sampledistance = Util::calcDistance<int32_t>(areacenter, samplepoint);

    if(mindifference > m_fore_maxfeaturedifference && sampledistance > areadistance) {
      m_siftvertices[i].m_state = SiftVertex::STATE_BACKGROUND;
      for(size_t j = 0; j < m_match_groups[siftvertex.m_vertexid].size(); j++) {
        SiftMatchPare pare = m_match_groups[siftvertex.m_vertexid].at(j);
        for(int32_t k = 0; k < pare.m_source_descriptor.cols; k++) {
          float_t value = 0.0;
          if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_source_descriptor.at<double_t>(0, k));
          else if (DIFF_OF_DST == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_destination_descriptor.at<double_t>(0, k));
          else if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
            value = static_cast<float_t>(pare.m_difference_descriptor.at<double_t>(0, k));
          back_data.push_back(value);
        }
      }
    }
  }
  /* ディープコピー */
  int32_t back_rows = back_data.size() / 128;
  m_back_samples = cv::Mat(back_rows, 128, CV_32FC1);
  for(int32_t i = 0; i < back_rows; i++)
    for(int32_t j = 0; j < 128; j++) {
      m_back_samples.at<float_t>(i, j) = back_data[i * 128 + j];
    }
  cv::TermCriteria termcriteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, kMeansItterationCount, 1.0);
  cv::kmeans(m_back_samples, kDefClusters, m_back_labels, termcriteria, 1, kMeansType, m_back_visualword);

  m_back_gmm.initLearning();

  for(int32_t i = 0; i < m_back_samples.rows; i++) {
    cv::Mat samplef = m_back_samples.row(i);
    cv::Mat sampled;
    samplef.convertTo(sampled, CV_64FC1);
    m_back_gmm.addSample(m_back_labels.at<int32_t>(i, 0), sampled);
  }
  m_back_gmm.endLearning();
}

void SiftGraphBuilder::calcBeta(void) {
  double_t beta = 0.0;
  double_t maxdistance = 0.0;
  double_t mindistance = DBL_MAX;
  for(size_t i = 0; i < m_siftvertices.size(); i++) {
    SiftVertex origin_siftvertex = m_siftvertices.at(i);
    /* Sift特徴量を使用*/
    cv::Mat origin_feature;
    if(DIFF_OF_SOURCE == kDefWhichUseForNeighbor)
      origin_feature = m_match_groups[origin_siftvertex.m_vertexid][0].m_source_descriptor;
    if(DIFF_OF_DST == kDefWhichUseForNeighbor)
      origin_feature = m_match_groups[origin_siftvertex.m_vertexid][0].m_destination_descriptor;
    if(DIFF_OF_DIFF == kDefWhichUseForNeighbor)
      origin_feature = m_match_groups[origin_siftvertex.m_vertexid][0].m_difference_descriptor;

    cv::Point2f origin_point = m_match_groups[origin_siftvertex.m_vertexid][0].m_source_key.pt;

    for(size_t j = 0; j < origin_siftvertex.m_dstverticesid.size(); j++) {
      int32_t dst_vertexid = origin_siftvertex.m_dstverticesid[j];
      cv::Mat destination_feature;
      if(DIFF_OF_SOURCE == kDefWhichUseForNeighbor)
        destination_feature = m_match_groups[dst_vertexid][0].m_source_descriptor;
      if(DIFF_OF_DST == kDefWhichUseForNeighbor)
        destination_feature = m_match_groups[dst_vertexid][0].m_destination_descriptor;
      if(DIFF_OF_DIFF == kDefWhichUseForNeighbor)
        destination_feature = m_match_groups[dst_vertexid][0].m_difference_descriptor;

      cv::Point2f dst_point = m_match_groups[dst_vertexid][0].m_source_key.pt;
      double_t distance = Util::calcDistance<float_t>(origin_point, dst_point);
      if(maxdistance < distance)
        maxdistance = distance;
      if(mindistance > distance)
        mindistance = distance;

      cv::Mat diff = origin_feature - destination_feature;
      beta += diff.dot(diff);
    }
  }
  if(std::numeric_limits<double_t>::epsilon() >= beta)
    beta = 0;
  else
    beta = 1.f / (2 * beta / m_edgecount);
  m_beta = beta;
  m_maxdistance = maxdistance;
  m_mindistance = mindistance;
}

double_t SiftGraphBuilder::calcTerminalWeights(const cv::Mat &vector, const Gmm &gmm) {
  double_t weight = gmm.calcMinDistance(vector);
  //weight *= -m_beta;
  //weight = cv::exp(weight);
  return weight;
}

void SiftGraphBuilder::calcNeighborWeights(void) {
  double_t maxneighbordifference = 0.0;
  double_t sumofneighborcost = 0.0;

  for(size_t i = 0; i < m_siftvertices.size(); i++) {
    sumofneighborcost = 0.0;
    SiftVertex origin_siftvertex = m_siftvertices.at(i);
    /* どの特徴差を利用するか指定 */
    cv::Mat origin_feature;
    if(DIFF_OF_SOURCE == kDefWhichUseForNeighbor)
      origin_feature = m_match_groups[origin_siftvertex.m_vertexid][0].m_source_descriptor;
    if(DIFF_OF_DST == kDefWhichUseForNeighbor)
      origin_feature = m_match_groups[origin_siftvertex.m_vertexid][0].m_destination_descriptor;
    if(DIFF_OF_DIFF == kDefWhichUseForNeighbor)
      origin_feature = m_match_groups[origin_siftvertex.m_vertexid][0].m_difference_descriptor;

    for(size_t j = 0; j < origin_siftvertex.m_dstverticesid.size(); j++) {
      int32_t dst_vertexid = origin_siftvertex.m_dstverticesid[j];
      cv::Mat destination_feature;
      if(DIFF_OF_SOURCE == kDefWhichUseForNeighbor)
        destination_feature = m_match_groups[dst_vertexid][0].m_source_descriptor;
      if(DIFF_OF_DST == kDefWhichUseForNeighbor)
        destination_feature = m_match_groups[dst_vertexid][0].m_destination_descriptor;
      if(DIFF_OF_DIFF == kDefWhichUseForNeighbor)
        destination_feature = m_match_groups[dst_vertexid][0].m_difference_descriptor;

      cv::Mat diff = origin_feature - destination_feature;

      cv::Point2f source_point = m_match_groups[origin_siftvertex.m_vertexid][0].m_source_key.pt;
      cv::Point2f destination_point = m_match_groups[dst_vertexid][0].m_source_key.pt;
      double_t distance = Util::calcDistance<float_t>(source_point, destination_point);
      double_t weight = Util::calcL2NormOfVectors<double_t>(origin_feature, destination_feature, true);
      if(METHODE_LOGLIKELIHOOD == kDefWeightCalculationMethod) {
        weight *= -m_beta;
        weight = cv::exp(weight);
        weight = m_gamma * weight;
      } else if(METHODE_LOGLIKELIHOOD_XYDISTANCE == kDefWeightCalculationMethod) {
        weight *= -m_beta;
        weight = cv::exp(weight);
        // weight = m_gamma / (distance / m_maxdistance) * weight;
        weight = m_gamma * (distance / m_maxdistance) * weight;
        //weight = m_gamma * (distance / m_mindistance) * weight;
      } else if (METHODE_FEATUREDISTANCE == kDefWeightCalculationMethod) {
        weight = m_gamma * weight;
      } else if (METHODE_FEATUREDISTANCE_XYDISTANCE == kDefWeightCalculationMethod) {
        // weight = m_gamma / (distance / m_maxdistance) * weight;
        weight = m_gamma * (distance / m_maxdistance) * weight;
        //weight = m_gamma * (distance / m_mindistance) * weight;
      }
      m_siftvertices[i].m_neighbor_weights.push_back(weight);
      sumofneighborcost += weight;
    }
    if(maxneighbordifference < sumofneighborcost)
      maxneighbordifference = sumofneighborcost;
  }
  m_maxneighbordifference = maxneighbordifference;
}


void SiftGraphBuilder::assignGmmComponents(void) {
  for(size_t i = 0; i < m_siftvertices.size(); i++) {
    SiftVertex siftvertex = m_siftvertices.at(i);
    for(size_t j = 0; j < m_match_groups[siftvertex.m_vertexid].size(); j++) {
      SiftMatchPare pare = m_match_groups[siftvertex.m_vertexid].at(j);
      if(SiftVertex::STATE_BACKGROUND != siftvertex.m_state) {
        int32_t whichcomponent = -1;
        if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
          whichcomponent = m_fore_gmm.whichComponent(pare.m_source_descriptor);
        else if(DIFF_OF_DST == kDefWhichUseForTerminal)
          whichcomponent = m_fore_gmm.whichComponent(pare.m_destination_descriptor);
        else if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
          whichcomponent = m_fore_gmm.whichComponent(pare.m_difference_descriptor);
        m_siftvertices[i].m_fore_componentindices.push_back(whichcomponent);
      }
      if(SiftVertex::STATE_FOREGROUND != siftvertex.m_state) {
        int32_t whichcomponent = -1;
        if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
          whichcomponent = m_back_gmm.whichComponent(pare.m_source_descriptor);
        else if(DIFF_OF_DST == kDefWhichUseForTerminal)
          whichcomponent = m_back_gmm.whichComponent(pare.m_destination_descriptor);
        else if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
          whichcomponent = m_back_gmm.whichComponent(pare.m_difference_descriptor);
        m_siftvertices[i].m_back_componentindices.push_back(whichcomponent);
      }
    }
  }
}

void SiftGraphBuilder::assignGmmComponents2(void) {
  for(size_t i = 0; i < m_siftvertices.size(); i++) {
    SiftVertex siftvertex = m_siftvertices.at(i);
    for(size_t j = 0; j < m_match_groups[siftvertex.m_vertexid].size(); j++) {
      SiftMatchPare pare = m_match_groups[siftvertex.m_vertexid].at(j);
      if(SiftVertex::STATE_FOREGROUND == siftvertex.m_state || SiftVertex::STATE_PROB_FOREGROUND == siftvertex.m_state) {
        int32_t whichcomponent = -1;
        if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
          whichcomponent = m_fore_gmm.whichComponent(pare.m_source_descriptor);
        else if(DIFF_OF_DST == kDefWhichUseForTerminal)
          whichcomponent = m_fore_gmm.whichComponent(pare.m_destination_descriptor);
        else if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
          whichcomponent = m_fore_gmm.whichComponent(pare.m_difference_descriptor);
        m_siftvertices[i].m_fore_componentindices.push_back(whichcomponent);
      }
      if(SiftVertex::STATE_BACKGROUND == siftvertex.m_state || SiftVertex::STATE_PROB_BACKGROUND == siftvertex.m_state) {
        int32_t whichcomponent = -1;
        if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
          whichcomponent = m_back_gmm.whichComponent(pare.m_source_descriptor);
        else if(DIFF_OF_DST == kDefWhichUseForTerminal)
          whichcomponent = m_back_gmm.whichComponent(pare.m_destination_descriptor);
        else if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
          whichcomponent = m_back_gmm.whichComponent(pare.m_difference_descriptor);
        m_siftvertices[i].m_back_componentindices.push_back(whichcomponent);
      }
    }
  }
}

void SiftGraphBuilder::assignGmmComponents3(void) {
  for(size_t i = 0; i < m_siftvertices.size(); i++) {
    SiftVertex siftvertex = m_siftvertices.at(i);
    for(size_t j = 0; j < m_match_groups[siftvertex.m_vertexid].size(); j++) {
      SiftMatchPare pare = m_match_groups[siftvertex.m_vertexid].at(j);
      if(SiftVertex::STATE_FOREGROUND == siftvertex.m_state) {
        int32_t whichcomponent = -1;
        if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
          whichcomponent = m_fore_gmm.whichComponent(pare.m_source_descriptor);
        else if(DIFF_OF_DST == kDefWhichUseForTerminal)
          whichcomponent = m_fore_gmm.whichComponent(pare.m_destination_descriptor);
        else if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
          whichcomponent = m_fore_gmm.whichComponent(pare.m_difference_descriptor);
        m_siftvertices[i].m_fore_componentindices.push_back(whichcomponent);
      }
      if(SiftVertex::STATE_BACKGROUND == siftvertex.m_state) {
        int32_t whichcomponent = -1;
        if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
          whichcomponent = m_back_gmm.whichComponent(pare.m_source_descriptor);
        else if(DIFF_OF_DST == kDefWhichUseForTerminal)
          whichcomponent = m_back_gmm.whichComponent(pare.m_destination_descriptor);
        else if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
          whichcomponent = m_back_gmm.whichComponent(pare.m_difference_descriptor);
        m_siftvertices[i].m_back_componentindices.push_back(whichcomponent);
      }
    }
  }
}

void SiftGraphBuilder::learnVisualWord(void) {
  m_fore_gmm.initLearning();
  m_back_gmm.initLearning();
  for(int32_t index_component = 0; index_component < Gmm::kComponentsCount; index_component++) {
    for(size_t i = 0; i < m_siftvertices.size(); i++) {
      SiftVertex siftvertex = m_siftvertices.at(i);
      for(size_t j = 0; j < m_match_groups[siftvertex.m_vertexid].size(); j++) {
        SiftMatchPare pare = m_match_groups[siftvertex.m_vertexid].at(j);
        if(SiftVertex::STATE_FOREGROUND == siftvertex.m_state) {
          if(index_component == siftvertex.m_fore_componentindices[j]) {
            if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
              m_fore_gmm.addSample(index_component, pare.m_source_descriptor);
            if(DIFF_OF_DST == kDefWhichUseForTerminal)
              m_fore_gmm.addSample(index_component, pare.m_destination_descriptor);
            if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
              m_fore_gmm.addSample(index_component, pare.m_difference_descriptor);
          }
        } else if(SiftVertex::STATE_BACKGROUND == siftvertex.m_state) {
          if(index_component == siftvertex.m_back_componentindices[j]) {
            if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
              m_back_gmm.addSample(index_component, pare.m_source_descriptor);
            if(DIFF_OF_DST == kDefWhichUseForTerminal)
              m_back_gmm.addSample(index_component, pare.m_destination_descriptor);
            if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
              m_back_gmm.addSample(index_component, pare.m_difference_descriptor);
          }
        } else {
          if(DIFF_OF_SOURCE == kDefWhichUseForTerminal) {
            m_fore_gmm.addSample(siftvertex.m_fore_componentindices[j], pare.m_source_descriptor);
            m_back_gmm.addSample(siftvertex.m_back_componentindices[j], pare.m_source_descriptor);
          }
          if(DIFF_OF_DST == kDefWhichUseForTerminal) {
            m_fore_gmm.addSample(siftvertex.m_fore_componentindices[j], pare.m_destination_descriptor);
            m_back_gmm.addSample(siftvertex.m_back_componentindices[j], pare.m_destination_descriptor);
          }
          if(DIFF_OF_DIFF == kDefWhichUseForTerminal) {
            m_fore_gmm.addSample(siftvertex.m_fore_componentindices[j], pare.m_difference_descriptor);
            m_back_gmm.addSample(siftvertex.m_back_componentindices[j], pare.m_difference_descriptor);
          }
        }
      }
    }
  }
  m_fore_gmm.endLearning();
  m_back_gmm.endLearning();
}

void SiftGraphBuilder::learnVisualWord2(void) {
  m_fore_gmm.initLearning();
  m_back_gmm.initLearning();
  for(int32_t index_component = 0; index_component < Gmm::kComponentsCount; index_component++) {
    for(size_t i = 0; i < m_siftvertices.size(); i++) {
      SiftVertex siftvertex = m_siftvertices.at(i);
      for(size_t j = 0; j < m_match_groups[siftvertex.m_vertexid].size(); j++) {
        SiftMatchPare pare = m_match_groups[siftvertex.m_vertexid].at(j);
        if(SiftVertex::STATE_FOREGROUND == siftvertex.m_state || SiftVertex::STATE_PROB_FOREGROUND == siftvertex.m_state) {
          if(index_component == siftvertex.m_fore_componentindices[j]) {
            if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
              m_fore_gmm.addSample(index_component, pare.m_source_descriptor);
            if(DIFF_OF_DST == kDefWhichUseForTerminal)
              m_fore_gmm.addSample(index_component, pare.m_destination_descriptor);
            if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
              m_fore_gmm.addSample(index_component, pare.m_difference_descriptor);
          }
        } else if(SiftVertex::STATE_BACKGROUND == siftvertex.m_state || SiftVertex::STATE_PROB_BACKGROUND == siftvertex.m_state) {
          if(index_component == siftvertex.m_back_componentindices[j]) {
            if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
              m_back_gmm.addSample(index_component, pare.m_source_descriptor);
            if(DIFF_OF_DST == kDefWhichUseForTerminal)
              m_back_gmm.addSample(index_component, pare.m_destination_descriptor);
            if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
              m_back_gmm.addSample(index_component, pare.m_difference_descriptor);
          }
        }
      }
    }
  }
  m_fore_gmm.endLearning();
  m_back_gmm.endLearning();
}


void SiftGraphBuilder::learnVisualWord3(void) {
  m_fore_gmm.initLearning();
  m_back_gmm.initLearning();
  for(int32_t index_component = 0; index_component < Gmm::kComponentsCount; index_component++) {
    for(size_t i = 0; i < m_siftvertices.size(); i++) {
      SiftVertex siftvertex = m_siftvertices.at(i);
      for(size_t j = 0; j < m_match_groups[siftvertex.m_vertexid].size(); j++) {
        SiftMatchPare pare = m_match_groups[siftvertex.m_vertexid].at(j);
        if(SiftVertex::STATE_FOREGROUND == siftvertex.m_state) {
          if(index_component == siftvertex.m_fore_componentindices[j]) {
            if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
              m_fore_gmm.addSample(index_component, pare.m_source_descriptor);
            if(DIFF_OF_DST == kDefWhichUseForTerminal)
              m_fore_gmm.addSample(index_component, pare.m_destination_descriptor);
            if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
              m_fore_gmm.addSample(index_component, pare.m_difference_descriptor);
          }
        } else if(SiftVertex::STATE_BACKGROUND == siftvertex.m_state) {
          if(index_component == siftvertex.m_back_componentindices[j]) {
            if(DIFF_OF_SOURCE == kDefWhichUseForTerminal)
              m_back_gmm.addSample(index_component, pare.m_source_descriptor);
            if(DIFF_OF_DST == kDefWhichUseForTerminal)
              m_back_gmm.addSample(index_component, pare.m_destination_descriptor);
            if(DIFF_OF_DIFF == kDefWhichUseForTerminal)
              m_back_gmm.addSample(index_component, pare.m_difference_descriptor);
          }
        }
      }
    }
  }
  m_fore_gmm.endLearning();
  m_back_gmm.endLearning();
}

void SiftGraphBuilder::constructGcGraph(GCGraph &graph) {
  /* グラフ構築 */
  m_graph.create(m_vertexcount, m_edgecount);
  for(size_t i = 0; i < m_siftvertices.size(); i++)
    graph.addVertex();

  for(size_t i = 0; i < m_siftvertices.size(); i++) {
    /* 頂点の追加 */
    SiftVertex siftvertex = m_siftvertices.at(i);
    int32_t originvertex = i;
    SiftMatchPare originpare = m_match_groups[siftvertex.m_vertexid].front();
    //cv::Mat originfeature = originpare.m_source_descriptor; // SIFT特徴量
    cv::Mat originfeature = originpare.m_difference_descriptor;

    /* Source, Sinkへの重み追加 */
    double_t fromsource_weight, tosink_weight;
    if(SiftVertex::STATE_FOREGROUND == siftvertex.m_state) {
      fromsource_weight = m_maxneighbordifference + 1.0;
      tosink_weight = 0.0;
    } else if(SiftVertex::STATE_BACKGROUND == siftvertex.m_state) {
      fromsource_weight = 0.0;
      tosink_weight = m_maxneighbordifference + 1.0;
    } else {
      if(METHODE_LOGLIKELIHOOD == kDefWeightCalculationMethod || METHODE_LOGLIKELIHOOD_XYDISTANCE == kDefWeightCalculationMethod) {
        fromsource_weight = -log(m_back_gmm(originfeature));
        tosink_weight = -log(m_fore_gmm(originfeature));
      } else if(METHODE_FEATUREDISTANCE == kDefWeightCalculationMethod || METHODE_FEATUREDISTANCE_XYDISTANCE == kDefWeightCalculationMethod) {
        fromsource_weight = calcTerminalWeights(originfeature, m_back_gmm);
        tosink_weight = calcTerminalWeights(originfeature, m_fore_gmm);
      }
      // debug print
      LOG(INFO) << "Group : " << i << "," << "FromSource : " << fromsource_weight << "," << "ToSink : "<< tosink_weight << std::endl;
    }
    m_siftvertices[i].m_fromsource_weight = fromsource_weight;
    m_siftvertices[i].m_tosink_weight = tosink_weight;
    graph.addTerminalWeights(originvertex, fromsource_weight, tosink_weight);

    /* 隣接頂点への重み追加 */
    for(size_t j = 0; j < siftvertex.m_dstverticesid.size(); j++) {
      int32_t destinationvertex = siftvertex.m_dstverticesid.at(j);
      double_t weight = siftvertex.m_neighbor_weights.at(j);
      graph.addEdges(originvertex, destinationvertex, weight, weight);
    }
  }
}

void SiftGraphBuilder::estimateSegmentation(GCGraph &graph) {
  graph.maxFlow();
  for(size_t i = 0; i < m_siftvertices.size(); i++) {
    SiftVertex siftvertex = m_siftvertices.at(i);
    SiftMatchPare frontpare = m_match_groups[siftvertex.m_vertexid].front();
    if(graph.inSourceSegment(i)) {
      for(size_t j = 0; j < m_match_groups[siftvertex.m_vertexid].size(); j++) {
        cv::Point point = m_match_groups[siftvertex.m_vertexid][j].m_source_key.pt;
        m_source_points.push_back(point);
      }
    } else {
      for(size_t j = 0; j < m_match_groups[siftvertex.m_vertexid].size(); j++) {
        cv::Point point = m_match_groups[siftvertex.m_vertexid][j].m_source_key.pt;
        m_sink_points.push_back(point);
      }
    }
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
      if(origin.inside(m_image_area) && destination.inside(m_image_area)) {
        cv::line(image, origin, destination, cv::Scalar(0, 200, 0));
        //cv::circle(image, origin, 4, cv::Scalar(0, 255, 0), -1);
      }
    }
    for(size_t i = 0; i < m_facet_centers.size(); i++) {
      cv::Point2f origin = m_facet_centers.at(i);
      /* 頂点が画像内の場合のみ描画 */
      if(origin.inside(m_image_area))
        cv::circle(image, origin, 4, cv::Scalar(0, 255, 0), -1);
    }
    //for(delaunay_edge = m_delaunay_edges.begin(); delaunay_edge != m_delaunay_edges.end(); delaunay_edge++) {
    //  cv::Point origin(delaunay_edge->val[0], delaunay_edge->val[1]);
    //  /* 頂点が画像内の場合のみ描画 */
    //  if(origin.inside(m_image_area))
    //    cv::circle(image, origin, 4, cv::Scalar(0, 255, 0), -1);
    //}
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

void SiftGraphBuilder::drawGraphs2(cv::Mat &image, DrawDivType type) {
  if(DRAW_DIV_DELAUNAY == type) {
    for(size_t i = 0; i < m_siftvertices.size(); i++) {
      SiftVertex siftvertex = m_siftvertices.at(i);
      SiftMatchPare originpare = m_match_groups[siftvertex.m_vertexid].front();
      cv::Point originpoint = cv::Point(originpare.m_source_key.pt);
      for(size_t j = 0; j < siftvertex.m_dstverticesid.size(); j++) {
        SiftMatchPare dstpare = m_match_groups[siftvertex.m_dstverticesid.at(j)].front();
        cv::Point dstpoint = cv::Point(dstpare.m_source_key.pt);
        cv::line(image, originpoint, dstpoint, cv::Scalar(0, 200, 0));
        //cv::circle(image, dstpoint, 4, cv::Scalar(0, 200, 0));
      }
      cv::circle(image, originpoint, 4, cv::Scalar(0, 255, 0), -1);
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

void SiftGraphBuilder::drawSiftGroups(cv::Mat &image, DrawModelType type, cv::Scalar forecolor, cv::Scalar backcolor, cv::Scalar othercolor) {
  for(size_t i = 0; i < m_siftvertices.size(); i++) {
    if(DRAW_MODEL_FORE_ONLY == type) {
      SiftVertex siftvertex = m_siftvertices.at(i);
      if(SiftVertex::STATE_FOREGROUND == siftvertex.m_state) {
        std::vector<SiftMatchPare> group = m_match_groups.at(i);
        cv::Point frontpoint = group[0].m_source_key.pt;
        for(size_t j = 1; j < group.size(); j++) {
          cv::Point point = group[j].m_source_key.pt;
          cv::line(image, frontpoint, point, forecolor);
          cv::circle(image, point, 5, forecolor, -1);
        }
        /* 代表点は最後に描画 */
        cv::circle(image, frontpoint, 5, forecolor, -1);
      }
    } else if(DRAW_MODEL_BACK_ONLY == type) {
      SiftVertex siftvertex = m_siftvertices.at(i);
      if(SiftVertex::STATE_BACKGROUND == siftvertex.m_state) {
        std::vector<SiftMatchPare> group = m_match_groups.at(i);
        cv::Point frontpoint = group[0].m_source_key.pt;
        for(size_t j = 1; j < group.size(); j++) {
          cv::Point point = group[j].m_source_key.pt;
          cv::line(image, frontpoint, point, backcolor);
          cv::circle(image, point, 5, backcolor, -1);
        }
        /* 代表点は最後に描画 */
        cv::circle(image, frontpoint, 5, backcolor, -1);
      }
    } else if(DRAW_MODEL_BOTH == type) {
      SiftVertex siftvertex = m_siftvertices.at(i);
      if(SiftVertex::STATE_FOREGROUND == siftvertex.m_state) {
        std::vector<SiftMatchPare> group = m_match_groups.at(i);
        cv::Point frontpoint = group[0].m_source_key.pt;
        for(size_t j = 1; j < group.size(); j++) {
          cv::Point point = group[j].m_source_key.pt;
          cv::line(image, frontpoint, point, forecolor);
          cv::circle(image, point, 5, forecolor, -1);
        }
        /* 代表点は最後に描画 */
        cv::circle(image, frontpoint, 5, forecolor, -1);
      } else if(SiftVertex::STATE_BACKGROUND == siftvertex.m_state) {
        std::vector<SiftMatchPare> group = m_match_groups.at(i);
        cv::Point frontpoint = group[0].m_source_key.pt;
        for(size_t j = 1; j < group.size(); j++) {
          cv::Point point = group[j].m_source_key.pt;
          cv::line(image, frontpoint, point, backcolor);
          cv::circle(image, point, 5, backcolor, -1);
        }
        /* 代表点は最後に描画 */
        cv::circle(image, frontpoint, 5, backcolor, -1);
      }
    } else if(DRAW_MODEL_FORE_OTHER == type) {
      SiftVertex siftvertex = m_siftvertices.at(i);
      if(SiftVertex::STATE_FOREGROUND == siftvertex.m_state) {
        std::vector<SiftMatchPare> group = m_match_groups.at(i);
        cv::Point frontpoint = group[0].m_source_key.pt;
        for(size_t j = 1; j < group.size(); j++) {
          cv::Point point = group[j].m_source_key.pt;
          cv::line(image, frontpoint, point, forecolor);
          cv::circle(image, point, 5, forecolor, -1);
        }
        /* 代表点は最後に描画 */
        cv::circle(image, frontpoint, 5, forecolor, -1);
      } else if(SiftVertex::STATE_BACKGROUND != siftvertex.m_state) {
        std::vector<SiftMatchPare> group = m_match_groups.at(i);
        cv::Point frontpoint = group[0].m_source_key.pt;
        for(size_t j = 1; j < group.size(); j++) {
          cv::Point point = group[j].m_source_key.pt;
          cv::line(image, frontpoint, point, othercolor);
          cv::circle(image, point, 5, othercolor, -1);
        }
        /* 代表点は最後に描画 */
        cv::circle(image, frontpoint, 5, othercolor, -1);
      }
    } else if(DRAW_MODEL_BACK_OTHER == type) {
      SiftVertex siftvertex = m_siftvertices.at(i);
      if(SiftVertex::STATE_FOREGROUND == siftvertex.m_state) {
        std::vector<SiftMatchPare> group = m_match_groups.at(i);
        cv::Point frontpoint = group[0].m_source_key.pt;
        for(size_t j = 1; j < group.size(); j++) {
          cv::Point point = group[j].m_source_key.pt;
          cv::line(image, frontpoint, point, backcolor);
          cv::circle(image, point, 5, backcolor, -1);
        }
        /* 代表点は最後に描画 */
        cv::circle(image, frontpoint, 5, backcolor, -1);
      } else if(SiftVertex::STATE_FOREGROUND != siftvertex.m_state) {
        std::vector<SiftMatchPare> group = m_match_groups.at(i);
        cv::Point frontpoint = group[0].m_source_key.pt;
        for(size_t j = 1; j < group.size(); j++) {
          cv::Point point = group[j].m_source_key.pt;
          cv::line(image, frontpoint, point, othercolor);
          cv::circle(image, point, 5, othercolor, -1);
        }
        /* 代表点は最後に描画 */
        cv::circle(image, frontpoint, 5, othercolor, -1);
      }
    } else if(DRAW_MODEL_ALL == type) {
      SiftVertex siftvertex = m_siftvertices.at(i);
      if(SiftVertex::STATE_FOREGROUND == siftvertex.m_state) {
        std::vector<SiftMatchPare> group = m_match_groups.at(i);
        cv::Point frontpoint = group[0].m_source_key.pt;
        for(size_t j = 1; j < group.size(); j++) {
          cv::Point point = group[j].m_source_key.pt;
          cv::line(image, frontpoint, point, forecolor);
          cv::circle(image, point, 5, forecolor, -1);
        }
        /* 代表点は最後に描画 */
        cv::circle(image, frontpoint, 5, forecolor, -1);
      } else if(SiftVertex::STATE_BACKGROUND == siftvertex.m_state) {
        std::vector<SiftMatchPare> group = m_match_groups.at(i);
        cv::Point frontpoint = group[0].m_source_key.pt;
        for(size_t j = 1; j < group.size(); j++) {
          cv::Point point = group[j].m_source_key.pt;
          cv::line(image, frontpoint, point, backcolor);
          cv::circle(image, point, 5, backcolor, -1);
        }
        /* 代表点は最後に描画 */
        cv::circle(image, frontpoint, 5, backcolor, -1);
      } else {
        std::vector<SiftMatchPare> group = m_match_groups.at(i);
        cv::Point frontpoint = group[0].m_source_key.pt;
        for(size_t j = 1; j < group.size(); j++) {
          cv::Point point = group[j].m_source_key.pt;
          cv::line(image, frontpoint, point, othercolor);
          cv::circle(image, point, 5, othercolor, -1);
        }
        /* 代表点は最後に描画 */
        cv::circle(image, frontpoint, 5, othercolor, -1);
      }
    }
  }
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
void SiftGraphBuilder::setGamma(double_t gammma) {
  m_gamma = gammma;
}

/*--- Event -----------------------------------------------------------------*/

}  // namespace cvgraphcut_base


