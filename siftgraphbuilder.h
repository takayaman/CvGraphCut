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

  /** Constructor
   * @param match_groups Matching sift pares.
   * @param image_area Rectangle information for running subdivision algorithms.
   */
  SiftGraphBuilder(const std::vector<std::vector<SiftMatchPare> >& match_groups, cv::Rect image_area);

  /** Default destructor
   */
  ~SiftGraphBuilder(void);

  /*!
  * Assignment operator
  * @param rhs Right hand side
  * @return pointer of this object
  */
  SiftGraphBuilder& operator=(const SiftGraphBuilder& rhs);

  /** Build graph to cut
   * This method must call before other public methods.
   */
  void build(void);

  /** Run graphcut algorithms.
   * This method must call after build() is called.
   */
  void cutGraph(void);

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

  /** Draw vertex group which match to same SIFT keypoint.
   * @param image Image for drawing SIFT keypoints.
   * @param index Index of group.
   * @param color Color for drawing.
   */
  void drawSiftGroups(cv::Mat &image, int32_t index, cv::Scalar color);

 private:

  /** Build graph using subdivided algorithms.
   */
  void buildSubdivGraph(void);

  /** Learning visual words from predefined foreground/background SIFT keypoints.
   */
  void learnVisualWord(void);

  /** Calculate T-Weights(Source/Sink - vertex)
   */
  void calcTermWeights(void);

  /** Calculate G-Weights(vertex - vertex)
   */
  void calcNeighborWeights(void);

  /** Print std::vertex members
   */
  void debugPrintMembers(void);

 private:
  std::vector<std::vector<SiftMatchPare> > &m_match_groups; /**< SIFT Matching Groups */
  GCGraph m_graph;                      /**< Graph for cutting */
  cv::Rect m_image_area;                /**< Rectangle information for subdividing algorithms */

  /* For Visual Word */
  cv::Mat m_fore_visualword;            /**< Visual words for foreground. */
  cv::Mat m_back_visualword;            /**< Visual words for background.  */
  double_t m_fore_coefficients[kDefClusters]; /**< Relative coefficients between gaussian components. */
  double_t m_back_coefficients[kDefClusters]; /**< Relative coefficients between gaussian components. */
  cv::Mat m_fore_centroid;              /**< Centroids of foreground visual words. */
  cv::Mat m_back_centroid;              /**< Centroids of background visual words. */
  cv::Mat m_fore_labels;                /**< Indicate which feature vector is in which centroid */
  cv::Mat m_back_labels;                /**< Indicate which feature vector is in which centroid */

  /* For graph creation using subdivision algorithms. */
  cv::Subdiv2D m_subdiv;
  std::vector<cv::Point2f> m_delaunay_points; /**< Vertices of delaunay subdivision. */
  std::vector<cv::Vec4f> m_delaunay_edges; /**< Edges of delaunay subdivision. */
  std::vector<int32_t> m_facet_index;   /**< ID lists of volonoy regions. */
  std::vector<std::vector<cv::Point2f> > m_facet_lists; /**< Vertices lists of volonoy. */
  std::vector<cv::Point2f> m_facet_centers; /**< Centroids lists of volonoy */

  std::vector<cv::Point2f> m_source_points; /**< Points in Source after cut. */
  std::vector<cv::Point2f> m_sink_points; /**< Points in Sink after cut. */

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
