/*=============================================================================
 * Project : CvGraphCut
 * Code : gmm.h
 * Written : N.Takayama, UEC
 * Date : 2014/11/29
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Definition of cvgraphcut_base::Gmm
 * This class is modified version of GMM class in opencv3.0a
 *===========================================================================*/

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_GMM_H_
#define _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_GMM_H_

/*=== Include ===============================================================*/

#include <stdint.h>
#include <glog/logging.h>

#include <opencv2/opencv.hpp>

/*=== Define ================================================================*/

/*=== Class Definition  =====================================================*/

namespace cvgraphcut_base {

class Gmm {
 public:
  static const int32_t kComponentsCount = 5;
  static const int32_t kVectorSize = 128;

  typedef cv::Vec<double_t, kVectorSize> MeanData;

  /*!
  * Defoult constructor
  */
  Gmm(void);

  /*!
  * Default destructor
  */
  ~Gmm(void);

  double_t operator ()(const cv::Vec3d color) const;
  double_t operator ()(const cv::Mat &vector) const;
  double_t operator ()(int32_t index_component, const cv::Vec3d color) const;
  double_t operator ()(int32_t index_component, const cv::Mat &vector) const;

  double_t calcMinDistance(const cv::Mat& vector) const;

  int32_t whichComponent(const cv::Vec3d color) const;
  int32_t whichComponent(const cv::Mat& vector) const;


  /* ガウス変数計算用のメンバを初期化 */
  void initLearning(void);

  /* 色値を基にガウス変数を計算 */
  void addSample(int32_t index_component, const cv::Vec3d color);
  void addSample(int32_t index_component, const cv::Mat& vector);

  /* addSample()の計算値をガウス変数に反映 */
  void endLearning(void);

  void debugPrintMembers(void);


 private:
  void init(void);

  void calcInverseCovAndDeterm(int32_t index_component);
  double_t calcLikelihoodInComponent(int32_t index_component, const cv::Mat &vector) const;
  double_t calcSumOfLikelihood(const cv::Mat &vector) const;

 private:

  /* ガウス分布間の相対係数 */
  std::vector<double_t> m_coefficients;
  /* 平均値 */
  std::vector<cv::Mat> m_centroids;
  std::vector<MeanData> m_vectormeans;
  std::vector<MeanData> m_vectorsums;

  /* 共分散 */
  std::vector<cv::Mat> m_covariant_matrices;

  /* 逆行列の共分散 */
  std::vector<cv::Mat> m_inverse_covariant_matrices;

  /* 行列式 */
  std::vector<double_t> m_covariant_determinants;

  /* データの積 */
  std::vector<cv::Mat> m_product_matrices;

  /* ガウス分布毎のデータ数 */
  int32_t m_samplecounts[kComponentsCount];
  /* 全体のデータ数 */
  int32_t m_totalsamplecount;

};

/*!
 * Log output operator
 * @param lhs Left hand side
 * @param rhs Right hand side
 * @return Pointer of google::LogSink object
 */
google::LogMessage& operator<<(google::LogMessage& lhs, const Gmm& rhs);

}  // namespace cvgraphcut_base


#endif  // _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_GMM_H_
