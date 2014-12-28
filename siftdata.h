/*=============================================================================
 * Project : CvGraphCut
 * Code : siftdata.h
 * Written : N.Takayama, UEC
 * Date : 2014/12/ 5
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Definition of cvgraphcut_base::Siftdata
 * Packing sift information
 *===========================================================================*/

#ifndef CVGRAPHCUT_CVGRAPHCUT_SIFTDATA_H_
#define CVGRAPHCUT_CVGRAPHCUT_SIFTDATA_H_

/*=== Include ===============================================================*/

#include <stdint.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <vector>

/*=== Define ================================================================*/


/*=== Class Definition  =====================================================*/

namespace cvgraphcut_base {


/** Class for basic SIFT operation and matching.
 */
class SiftData {
 public:
  /** Constructor
   * @param image Image for extract SIFT
   */
  explicit SiftData(cv::Mat &image);

  /** Default destructor
   */
  ~SiftData(void);

  /*!
  * Assignment operator
  * @param rhs Right hand side
  * @return pointer of this object
  */
  SiftData& operator=(const SiftData& rhs);

  /** Get reference of keypoints
   * @return Reference of keypoints.
   */
  std::vector<cv::KeyPoint>& getKeyPoints(void);

  /** Get SIFT feature vector.
   * @return Reference of feature vector.
   */
  cv::Mat& getDescriptor(void);

  /** Set SIFT Parameters for detection and extraction.
   * @param num_features Number of detecting feature.
   * @param octave_layers Octave layers
   * @param constrast_threshold Threshold for removing keypoints besed on contrast.
   * @param edge_threshold Threshold for removing keypoints whether a point is on edge.
   * @param sigma Gaussian step for generating pyramid images.
   */
  void setSiftParams(double_t num_features, double_t octave_layers,
                     double_t constrast_threshold, double_t edge_threshold,
                     double_t sigma);

  /** Get SIFT parameter, number of detecting features.
   * @return Number of detecting features.
   */
  double_t getNumFeatures(void) const;

  /** Get SIFT parameter, octave layers
   * @return Octave layers
   */
  double_t getOctaveLayers(void) const;

  /** Get SIFT paramter, contrast threshold 
   * @return Contrast threshold
   */
  double_t getContrastThreshold(void) const;

  /** Get SIFT parameter, edge threshold
   * @return Edge threshold
   */
  double_t getEdgeThreshold(void) const;

  /** Get SIFT parameter, Gaussian step for generating pyramid images.
   * @return Gaussian step.
   */
  double_t getSigma(void) const;

  /** Build SIFT and run detection and extration
   */
  void build(void);

  /** Whether SIFT is builed.
   * @return true : builed, false : not builded.
   */
  bool isBuilded(void) const;


 private:
  cv::Mat &m_image;                     /**< Image for detecting SIFT features */
  std::vector<cv::KeyPoint> m_keypoints; /**< SIFT keypoints */
  cv::Mat m_descriptor;                 /**< SIFT feature vectors */

  double_t m_num_features;              /**< SIFT param, number of detecting features. */
  double_t m_octave_layers;             /**< SIFT param, octave layers. */
  double_t m_contrast_threshold;        /**< SIFT param, contrast threshold for removing noisy keypoints. */
  double_t m_edge_threshold;            /**< SIFT param, edge threshold for removing keypoints which are on edges. */
  double_t m_sigma;                     /**< SIFT param, gaussian step */

  bool is_builded;                      /**< Flag for indicating builded */
};

/*!
 * Log output operator
 * @param lhs Left hand side
 * @param rhs Right hand side
 * @return Pointer of google::LogSink object
 */
google::LogMessage& operator<<(google::LogMessage& lhs, const SiftData& rhs);

}  // namespace cvgraphcut_base


#endif  // CVGRAPHCUT_CVGRAPHCUT_SIFTDATA_H_
