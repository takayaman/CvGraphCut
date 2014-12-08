/*=============================================================================
 * Project : CvGraphCut
 * Code : siftdata.h
 * Written : N.Takayama, UEC
 * Date : 2014/12/ 5
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Definition of cvgraphcut_base::Siftdata
 * Packing sift information
 *===========================================================================*/

#ifndef _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_SIFTDATA_H_
#define _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_SIFTDATA_H_

/*=== Include ===============================================================*/

#include <stdint.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

/*=== Define ================================================================*/

/*=== Class Definition  =====================================================*/

namespace cvgraphcut_base {


class SiftData {
public:
    /*!
    * Defoult constructor
    */
    SiftData(cv::Mat &image);

    /*!
    * Default destructor
    */
    ~SiftData(void);

    /*!
    * Assignment operator
    * @param rhs Right hand side
    * @return pointer of this object
    */
    SiftData& operator=(const SiftData& rhs);

    std::vector<cv::KeyPoint>& getKeyPoints(void);
    cv::Mat& getDescriptor(void);

    void setSiftParams(double_t num_features, double_t octave_layers,
                           double_t constrast_threshold, double_t edge_threshold,
                           double_t sigma);
    double_t getNumFeatures(void) const;
    double_t getOctaveLayers(void) const;
    double_t getContrastThreshold(void) const;
    double_t getEdgeThreshold(void) const;
    double_t getSigma(void) const;

    void build(void);
    bool isBuilded(void) const;

private:
    cv::Mat &m_image;
    std::vector<cv::KeyPoint> m_keypoints;
    cv::Mat m_descriptor;

    double_t m_num_features;
    double_t m_octave_layers;
    double_t m_contrast_threshold;
    double_t m_edge_threshold;
    double_t m_sigma;

    bool is_builded;
};

/*!
 * Log output operator
 * @param lhs Left hand side
 * @param rhs Right hand side
 * @return Pointer of google::LogSink object
 */
google::LogMessage& operator<<(google::LogMessage& lhs, const SiftData& rhs);

}  // namespace cvgraphcut_base


#endif  // _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_SIFTDATA_H_
