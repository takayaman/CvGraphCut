/*=============================================================================
 * Project : CvGraphCut
 * Code : siftdata.cpp
 * Written : N.Takayama, UEC
 * Date : 2014/12/ 5
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Implementation of cvgraphcut_base::Siftdata
 * Packing sift information
 *===========================================================================*/

/*=== Include ===============================================================*/

#include <siftdata.h>
#include <stdint.h>
#include <glog/logging.h>

/*=== Local Define / Local Const ============================================*/
static const double_t DefNumFeatures = 0.0;
static const double_t DefOctaveLayers = 3;
static const double_t DefContrastThreshold = 0.04;
static const double_t DefEdgeThreshold = 10;
static const double_t DefSigma = 1.6;

/* エイリアス */
namespace cvx = cv::xfeatures2d;
typedef cvx::SiftFeatureDetector Detector;
typedef cvx::SiftDescriptorExtractor Extractor;

/*=== Class Implementation ==================================================*/

namespace cvgraphcut_base {

/*--- Constructor / Destructor / Initialize ---------------------------------*/

/* Defoult constructor */
SiftData::SiftData(cv::Mat &image)
    : m_image(image),
      m_descriptor(cv::Mat()),
      m_num_features(DefNumFeatures),
      m_octave_layers(DefOctaveLayers),
      m_contrast_threshold(DefContrastThreshold),
      m_edge_threshold(DefEdgeThreshold),
      m_sigma(DefSigma),
      is_builded(false) {
    m_keypoints.clear();
}

/* Default destructor */
SiftData::~SiftData() {
}

/* Assignment operator */
SiftData& SiftData::operator=(const SiftData& rhs) {
  if (this != &rhs) {
    // TODO(N.Takayama): implement copy
  }
  return *this;
}

/*--- Operation -------------------------------------------------------------*/
void SiftData::build(void) {
    if(m_image.empty()) {
        LOG(ERROR) << "m_image is invalid!!" << std::endl;
        return;
    }

    cv::Ptr<Detector> p_siftdetector;
    p_siftdetector = Detector::create(
                m_num_features,
                m_octave_layers,
                m_contrast_threshold,
                m_edge_threshold,
                m_sigma);
    p_siftdetector->detect(m_image, m_keypoints);
    cv::Ptr<Extractor> p_siftextractor;
    p_siftextractor = Extractor::create(
                m_num_features,
                m_octave_layers,
                m_contrast_threshold,
                m_edge_threshold,
                m_sigma);
    p_siftextractor->compute(m_image, m_keypoints, m_descriptor);

    is_builded = true;
}

/*  Log output operator */
google::LogMessage& operator<<(google::LogMessage& lhs, const SiftData& rhs) {
  lhs.stream() << "cvgraphcut_base::Siftdata{" <<
      // TODO(N.Takayama): implement out stream of memder data
      "}" << std::endl;
  return lhs;
}

/*--- Accessor --------------------------------------------------------------*/
std::vector<cv::KeyPoint>& SiftData::getKeyPoints(void) {
    return m_keypoints;
}

cv::Mat& SiftData::getDescriptor(void) {
    return m_descriptor;
}

void SiftData::setSiftParams(double_t num_features,
                             double_t octave_layers,
                             double_t constrast_threshold,
                             double_t edge_threshold,
                             double_t sigma) {
    m_num_features = num_features;
    m_octave_layers = octave_layers;
    m_contrast_threshold = constrast_threshold;
    m_edge_threshold = edge_threshold;
    m_sigma = sigma;
}

double_t SiftData::getNumFeatures(void) const {
    return m_num_features;
}

double_t SiftData::getOctaveLayers(void) const {
    return m_octave_layers;
}

double_t SiftData::getContrastThreshold(void) const {
    return m_contrast_threshold;
}

double_t SiftData::getEdgeThreshold(void) const {
    return m_edge_threshold;
}

double_t SiftData::getSigma(void) const {
    return m_sigma;
}

bool SiftData::isBuilded(void) const {
    return is_builded;
}

/*--- Event -----------------------------------------------------------------*/

}  // namespace cvgraphcut_base


