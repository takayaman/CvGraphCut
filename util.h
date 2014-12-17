/*=============================================================================
 * Project : CvGraphCut
 * Code : util.h
 * Written : N.Takayama, UEC
 * Date : 2014/12/ 5
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Definition of cvgraphcut_base::Util
 * Utility
 *===========================================================================*/

#ifndef CVGRAPHCUT_CVGRAPHCUT_UTIL_H_
#define CVGRAPHCUT_CVGRAPHCUT_UTIL_H_

/*=== Include ===============================================================*/

#include <stdint.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include <fstream>

/*=== Define ================================================================*/

/*=== Class Definition  =====================================================*/

namespace cvgraphcut_base {

/** Utilyty class
 */
class Util {
 public:
  /** Defoult constructor
   * To avoid automatic generation of constructor.
   */
  Util(void);

  /** Default destructor
   * To avoid automatic generation of destructor.
   */
  ~Util(void);

  /** Return L2-Norm between points.
   * @param point0 point0
   * @param point1 point1
   * @return L2-Norm
   * \f$ distance = \sqrt{(point0.x - point1.x)^2 + (point0.y - point1.y)^2} \f$
   */
  template<typename _Tp>
  static double_t calcDistance(cv::Point_<_Tp> point0, cv::Point_<_Tp> point1) {
    double_t diff_X2 = pow(fabs(point0.x - point1.x), 2.0);
    double_t diff_Y2 = pow(fabs(point0.x - point1.x), 2.0);
    return sqrt(diff_X2 + diff_Y2);
  }

  /** Return L2-Norm between vectors.
   * @param vector0 vector0
   * @param vector1 vector1
   * @param aside_rows true : Calculate aside row, false : Calculate aside column.
   * @return L2-Norm
   * This method expects arguments of cv::Mat are vector.
   * Therefore, when aside_rows : true -> vector(0, 1).rows == 1, aside_rows : false -> vector(0, 1).cols == 1
   * \f$ distance = \sum^N_{i = 0}sqrt{(vector0_i - vector1_i)^2}; N\f$はベクトルの要素数
   */
  template<typename _Tp>
  static double_t calcL2NormOfVectors(const cv::Mat &vector0, const cv::Mat &vector1, bool aside_rows) {
    double_t l2_norm = 0.0;
    if(aside_rows) {
      for(int32_t i = 0; i < vector0.cols; i++)
        l2_norm += pow(abs(vector0.at<_Tp>(0, i) - abs(vector1.at<_Tp>(0, i))), 2.0);
    } else {
      for(int32_t i = 0; i < vector0.cols; i++)
        l2_norm += pow(abs(vector0.at<_Tp>(i, 0) - abs(vector1.at<_Tp>(i, 0))), 2.0);
    }
    return sqrt(l2_norm);
  }

  /** Save contents of cv::Mat as CVS file.
   * @param filename File name to save
   * @param mat Data to save
   * @return None
   * Data Format
   * row_0-col_0, row_0-col_1, ... , row_0-col_j, ... , row_0-col_m
   * row_1-col_0, row_1-col_1, ... , row_1-col_j, ... , row_1-col_m
   * ...
   * row_i-col_0, row_i-col_1, ..., row_i-col_j, ... , row_i-col_m
   * ...
   * row_n-col_0, row_n-col_1, ... , row_n-col_j, ... , row_n-col_m
   *
   * with row_i-col_j is an element at ith row and jth coumn,
   * n and m indicate the number of row and column repectively.
   */
  template<typename _Tp>
  static void logCvMatAsCvs(const std::string filename, const cv::Mat& mat) {
    std::ofstream logcvs;
    logcvs.open(filename);
    if(!logcvs.is_open()) {
      LOG(ERROR) << "Can not open log file " << filename << std::endl;
      return;
    }
    /* Save as CVS */
    int32_t rows = mat.rows;
    int32_t cols = mat.cols;
    for(int32_t row = 0; row < rows; row++) {
      for(int32_t col = 0; col < cols; col++) {
        if(cols - 1 == col)
          logcvs << mat.data[row * cols + col] << std::endl;
        else
          logcvs << mat.data[row * cols + col] << ",";
      }
    }
  }
};


}  // namespace cvgraphcut_base


#endif  // CVGRAPHCUT_CVGRAPHCUT_UTIL_H_
