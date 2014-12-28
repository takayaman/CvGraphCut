/*=============================================================================
 * Project : CvGraphCut
 * Code : siftgraphbuilder.h
 * Written : N.Takayama, UEC
 * Date : 2014/12/ 5
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Definition of cvgraphcut_base::Siftgraphbuilder
 * Build GCGraph based on Sift Keypoints
 *===========================================================================*/

#ifndef CVGRAPHCUT_CVGRAPHCUT_SIFTGRAPHBUILDER_H_
#define CVGRAPHCUT_CVGRAPHCUT_SIFTGRAPHBUILDER_H_

/*=== Include ===============================================================*/

#include <stdint.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <vector>

#include "./gcgraph.h"
#include "./siftdata.h"
#include "./siftmatchpare.h"
#include "./gmm.h"

/*=== Define ================================================================*/
// 前景グループ 0, 1, 2,
// 背景グループ 3, 4, 12,

/*=== Class Definition  =====================================================*/

namespace cvgraphcut_base {

/** Class for making and running graphcut.
 */
class SiftGraphBuilder {
 public:
  static const int32_t kDefClusters = 5; /**< The Number of BoF clusters */

  typedef enum DrawType_TAG {           /**< Draw types for drawing segment result */
    DRAW_DEFAULT = 0,
    DRAW_SOURCE = DRAW_DEFAULT,         /**< Draw points in source segment. */
    DRAW_SINK,                          /**< Draw points in sink segment. */
    DRAW_BOTH,                          /**< Draw points in both segment. */
    DRAW_NUM
  } DrawType;

  typedef enum DrawDivType_TAG {        /**< Draw types for drawing subdivided result. */
    DRAW_DIV_DEFAULT = 0,
    DRAW_DIV_DELAUNAY = DRAW_DIV_DEFAULT, /**< Draw result of delaunay subdivision. */
    DRAW_DIV_VORONOY,                   /**< Draw result of voronoy subdivision. */
    DRAW_DIV_NUM
  } DrawDivType;

  typedef enum DrawModelType_TAG {
    DRAW_MODEL_DEFAULT = 0,
    DRAW_MODEL_FORE_ONLY = DRAW_MODEL_DEFAULT,
    DRAW_MODEL_FORE_OTHER,
    DRAW_MODEL_BACK_ONLY,
    DRAW_MODEL_BACK_OTHER,
    DRAW_MODEL_BOTH,
    DRAW_MODEL_ALL,
    DRAW_MODEL_NUM
  } DrawModelType;

  /** Constructor
   * @param match_groups Matching sift pares.
   * @param image_area Rectangle information for running subdivision algorithms.
   */
  SiftGraphBuilder(std::vector<std::vector<SiftMatchPare> >& match_groups, std::vector<SiftMatchPare> &unmatch_group, cv::Rect image_area, cv::Rect fore_area, cv::Rect back_area);

  /** Default destructor
   */
  ~SiftGraphBuilder(void);

  /** Build graph to cut
   * This method must call before other public methods.
   */
  void build(void);

  /** Run graphcut algorithms.
   * This method must call after build() is called.
   */
  //void cutGraph(void);

  /** Draw result of cutted graph.
   * @param image Image for drawing cutted points.
   * @param type DrawType
   */
  void drawCuttedPoints(cv::Mat &image, DrawType type);

  /** Draw vertex and edges on image
   * @param image Image for drawing.
   * @param type DrawDivType
   */
  void drawGraphs(cv::Mat &image, DrawDivType type);
  void drawGraphs2(cv::Mat &image, DrawDivType type);

  /** Draw vertex group which match to same SIFT keypoint.
   * @param image Image for drawing SIFT keypoints.
   * @param index Index of group.
   * @param color Color for drawing.
   */
  void drawSiftGroups(cv::Mat &image, DrawModelType type, cv::Scalar forecolor, cv::Scalar backcolor, cv::Scalar othercolor);

  void setGamma(double_t gammma);

 private:

  /** Build graph using subdivided algorithms.
   */
  void buildSubdivGraph(void);

  /** Learning visual words from predefined foreground/background SIFT keypoints.
   */
  void initVisualWord(void);
  void initVisualWord2(void);
  void initVisualWord3(void);
  void initForeVisualWord3(void);
  void initBackVisualWord3(void);

  /** Calculate T-Weights(Source/Sink - vertex)
   */
  double_t calcTerminalWeights(const cv::Mat &vector, const Gmm &gmm);

  /** Calculate G-Weights(vertex - vertex)
   */
  void calcNeighborWeights(void);

  void calcBeta(void);

  /** Print std::vertex members
   */
  void debugPrintMembers(void);

  void assignGmmComponents(void);
  void assignGmmComponents2(void);
  void assignGmmComponents3(void);

  void learnVisualWord(void);
  void learnVisualWord2(void);
  void learnVisualWord3(void);


  void constructGcGraph(GCGraph &graph);

  void estimateSegmentation(GCGraph &graph);

 private:
  class SiftVertex {
   public:
    typedef enum State_TAG {
      STATE_DEFAULT = 0,
      STATE_PROB_BACKGROUND = STATE_DEFAULT,
      STATE_PROB_FOREGROUND,
      STATE_BACKGROUND,
      STATE_FOREGROUND,
      STATE_NUM
    } State;

    int32_t m_vertexid;
    double_t m_fromsource_weight;
    double_t m_tosink_weight;
    //std::vector<std::vector<SiftMatchPare> *> m_siftgroups;
    std::vector<int32_t> m_dstverticesid;
    std::vector<double_t> m_neighbor_weights;
    std::vector<int32_t> m_fore_componentindices;
    std::vector<int32_t> m_back_componentindices;
    State m_state;
  };

  std::vector<std::vector<SiftMatchPare> > &m_match_groups; /**< SIFT Matching Groups */
  std::vector<SiftMatchPare> &m_unmatch_group;
  std::vector<SiftVertex> m_siftvertices;
  int32_t m_vertexcount;
  int32_t m_edgecount;

  GCGraph m_graph;                      /**< Graph for cutting */
  cv::Rect m_image_area;                /**< Rectangle information for subdividing algorithms */
  cv::Rect m_fore_area;
  cv::Rect m_back_area;

  /* For Visual Word */
  cv::Mat m_fore_samples;            /**< Visual words for foreground. */
  cv::Mat m_back_samples;            /**< Visual words for background.  */
  cv::Mat m_fore_visualword;
  cv::Mat m_back_visualword;
  cv::Mat m_fore_labels;                /**< Indicate which feature vector is in which centroid */
  cv::Mat m_back_labels;                /**< Indicate which feature vector is in which centroid */
  double_t m_fore_maxfeaturedifference;

  Gmm m_fore_gmm;
  Gmm m_back_gmm;

  double_t m_gamma;
  double_t m_lambda;
  double_t m_beta;

  /* For graph creation using subdivision algorithms. */
  //cv::Subdiv2D m_subdiv;
  std::vector<cv::Point2f> m_delaunay_points; /**< Vertices of delaunay subdivision. */
  std::vector<cv::Vec4f> m_delaunay_edges; /**< Edges of delaunay subdivision. */
  //std::vector<int32_t> m_facet_index;   /**< ID lists of volonoy regions. */
  std::vector<std::vector<cv::Point2f> > m_facet_lists; /**< Vertices lists of volonoy. */
  std::vector<cv::Point2f> m_facet_centers; /**< Centroids lists of volonoy */

  std::vector<cv::Point2f> m_source_points; /**< Points in Source after cut. */
  std::vector<cv::Point2f> m_sink_points; /**< Points in Sink after cut. */

  double_t m_maxneighbordifference;
  double_t m_maxdistance;
  double_t m_mindistance;

};

/*!
 * Log output operator
 * @param lhs Left hand side
 * @param rhs Right hand side
 * @return Pointer of google::LogSink object
 */
google::LogMessage& operator<<(google::LogMessage& lhs, const SiftGraphBuilder& rhs);

}  // namespace cvgraphcut_base


#endif  // CVGRAPHCUT_CVGRAPHCUT_SIFTGRAPHBUILDER_H_
