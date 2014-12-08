/*=============================================================================
 * Project : CvGraphCut
 * Code : gmm.cpp
 * Written : N.Takayama, UEC
 * Date : 2014/11/29
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Implementation of cvgraphcut_base::Gmm
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

/*=== Include ===============================================================*/

#include <gmm.h>
#include <stdint.h>
#include <glog/logging.h>

/*=== Local Define / Local Const ============================================*/

/*=== Class Implementation ==================================================*/

namespace cvgraphcut_base {

/*--- Constructor / Destructor / Initialize ---------------------------------*/

/* Defoult constructor */
Gmm::Gmm(void)
    : m_totalsamplecount(0)
{
    init();
}

/* Default destructor */
Gmm::~Gmm(void) {
}

/*  Copy constructor */
Gmm::Gmm(const Gmm& rhs) {
}

/* Assignment operator */
Gmm& Gmm::operator=(const Gmm& rhs) {
  if (this != &rhs) {
    // TODO(N.Takayama): implement copy
  }
  return *this;
}

double_t Gmm::operator ()(const cv::Vec3d color) const {
    return calcSumOfLikelihood(color);
}

double_t Gmm::operator ()(int32_t index_component, const cv::Vec3d color) const {
    return calcLikelihoodInComponent(index_component, color);
}

void Gmm::init(void) {
    m_coefficients.resize(kComponentsCount);
    m_means.resize(kComponentsCount);
    m_covariants.resize(kComponentsCount);
    for(int32_t i = 0; i < kComponentsCount; i++){
        m_coefficients[i] = 0.0;
        m_means[i] = cv::Vec<double_t, kMeanSize>(0.0);
        m_covariants[i] = cv::Vec<double_t, kCovarianceSize>(0.0);

        m_inverse_covariants[i][0][0] = m_inverse_covariants[i][0][1] = m_inverse_covariants[i][0][2] = 0.0;
        m_inverse_covariants[i][1][0] = m_inverse_covariants[i][1][1] = m_inverse_covariants[i][1][2] = 0.0;
        m_inverse_covariants[i][2][0] = m_inverse_covariants[i][2][1] = m_inverse_covariants[i][1][2] = 0.0;

        m_covariant_Determinants[i] = 0;
        m_sums[i][0] = m_sums[i][1] = m_sums[i][2] = 0.0;
        m_products[i][0][0] = m_products[i][0][1] = m_products[i][0][2] = 0.0;
        m_products[i][1][0] = m_products[i][1][1] = m_products[i][1][2] = 0.0;
        m_products[i][2][0] = m_products[i][2][1] = m_products[i][1][2] = 0.0;
        m_samplecounts[i] = 0;
    }
}

void Gmm::initLearning(void) {
    for(int32_t index_components = 0; index_components < kComponentsCount; index_components++ ) {
        m_sums[index_components][0] = m_sums[index_components][1] = m_sums[index_components][2] = 0;
        m_products[index_components][0][0] = m_products[index_components][0][1] = m_products[index_components][0][2] = 0;
        m_products[index_components][1][0] = m_products[index_components][1][1] = m_products[index_components][1][2] = 0;
        m_products[index_components][2][0] = m_products[index_components][2][1] = m_products[index_components][2][2] = 0;
        m_samplecounts[index_components] = 0;
    }
    m_totalsamplecount = 0;
}

void Gmm::endLearning(void) {
    const double_t variance = 0.01;
    for(int32_t index_component = 0; index_component < kComponentsCount; index_component++) {
        double_t samplecount = static_cast<double_t>(m_samplecounts[index_component]);

        if(0 == samplecount)
            m_coefficients[index_component] = 0.0;
        else {
            m_coefficients[index_component] = samplecount / static_cast<double_t>(m_totalsamplecount);

            MeanData &mean = m_means.at(index_component);
            mean[0] = m_sums[index_component][0] / samplecount;
            mean[1] = m_sums[index_component][1] / samplecount;
            mean[2] = m_sums[index_component][2] / samplecount;

            CoVarianceData &covariance = m_covariants.at(index_component);
            covariance[0] = m_products[index_component][0][0] / samplecount - mean[0] * mean[0];
            covariance[1] = m_products[index_component][0][1] / samplecount - mean[0] * mean[1];
            covariance[2] = m_products[index_component][0][2] / samplecount - mean[0] * mean[2];
            covariance[3] = m_products[index_component][1][0] / samplecount - mean[1] * mean[0];
            covariance[4] = m_products[index_component][1][1] / samplecount - mean[1] * mean[1];
            covariance[5] = m_products[index_component][1][2] / samplecount - mean[1] * mean[2];
            covariance[6] = m_products[index_component][2][0] / samplecount - mean[2] * mean[0];
            covariance[7] = m_products[index_component][2][1] / samplecount - mean[2] * mean[1];
            covariance[8] = m_products[index_component][2][2] / samplecount - mean[2] * mean[2];

            double_t determinant = covariance[0] * (covariance[4] * covariance[8] - covariance[5] * covariance[7]) \
                                   - covariance[1] * (covariance[3] * covariance[8] - covariance[5] * covariance[6]) \
                                   + covariance[2] * (covariance[3] * covariance[7] - covariance[4] * covariance[6]);
            if(std::numeric_limits<double_t>::epsilon() >=  determinant) {
                covariance[0] += variance;
                covariance[4] += variance;
                covariance[8] += variance;
            }
            calcInverseCovAndDeterm(index_component);
        }
    }
}

/*--- Operation -------------------------------------------------------------*/
void Gmm::addSample(int32_t index_component, const cv::Vec3d color) {
    m_sums[index_component][0] += color[0];
    m_sums[index_component][1] += color[1];
    m_sums[index_component][2] += color[2];
    m_products[index_component][0][0] += color[0] * color[0];
    m_products[index_component][0][1] += color[0] * color[1];
    m_products[index_component][0][2] += color[0] * color[2];
    m_products[index_component][1][0] += color[1] * color[0];
    m_products[index_component][1][1] += color[1] * color[1];
    m_products[index_component][1][2] += color[1] * color[2];
    m_products[index_component][2][0] += color[2] * color[0];
    m_products[index_component][2][1] += color[2] * color[1];
    m_products[index_component][2][2] += color[2] * color[2];
    m_samplecounts[index_component]++;
    m_totalsamplecount++;
}

void Gmm::calcInverseCovAndDeterm(int32_t index_component) {
    if (0 < m_coefficients[index_component]) {
        CoVarianceData covariance = m_covariants.at(index_component);
        double_t determinant = covariance[0] * (covariance[4] * covariance[8] - covariance[5] * covariance[7]) \
                                           - covariance[1] * (covariance[3] * covariance[8] - covariance[5] * covariance[6]) \
                                           + covariance[2] * (covariance[3] * covariance[7] - covariance[4] * covariance[6]);
        m_covariant_Determinants[index_component] = determinant;
        CV_Assert(std::numeric_limits<double_t>::epsilon() < determinant);
        m_inverse_covariants[index_component][0][0] = (covariance[4] * covariance[8] - covariance[5] * covariance[7]) / determinant;
        m_inverse_covariants[index_component][1][0] = (covariance[3] * covariance[8] - covariance[5] * covariance[6]) / determinant;
        m_inverse_covariants[index_component][2][0] = (covariance[3] * covariance[7] - covariance[4] * covariance[6]) / determinant;
        m_inverse_covariants[index_component][0][1] = (covariance[1] * covariance[8] - covariance[2] * covariance[7]) / determinant;
        m_inverse_covariants[index_component][1][1] = (covariance[0] * covariance[8] - covariance[2] * covariance[6]) / determinant;
        m_inverse_covariants[index_component][2][1] = (covariance[0] * covariance[7] - covariance[1] * covariance[6]) / determinant;
        m_inverse_covariants[index_component][0][2] = (covariance[1] * covariance[5] - covariance[2] * covariance[4]) / determinant;
        m_inverse_covariants[index_component][1][2] = (covariance[0] * covariance[5] - covariance[2] * covariance[3]) / determinant;
        m_inverse_covariants[index_component][2][2] = (covariance[0] * covariance[4] - covariance[1] * covariance[3]) / determinant;
    }
}

double_t Gmm::calcLikelihoodInComponent(int32_t index_component, const cv::Vec3d color) const {
    double_t likelihood = 0.0;
    if(0 < m_coefficients[index_component]){
        CV_Assert(std::numeric_limits<double_t>::epsilon() < m_covariant_Determinants[index_component]);
        cv::Vec3d diff = color;
        MeanData mean = m_means.at(index_component);
        diff[0] -= mean[0];
        diff[1] -= mean[1];
        diff[2] -= mean[2];
        double_t multipication =
                diff[0] * (diff[0] * m_inverse_covariants[index_component][0][0] + diff[1] * m_inverse_covariants[index_component][1][0] + diff[2] * m_inverse_covariants[index_component][2][0])
                + diff[1] * (diff[0] * m_inverse_covariants[index_component][0][1] + diff[1] * m_inverse_covariants[index_component][1][1] + diff[2] * m_inverse_covariants[index_component][2][1])
                + diff[2] * (diff[0] * m_inverse_covariants[index_component][0][2] + diff[1] * m_inverse_covariants[index_component][1][2] + diff[2] * m_inverse_covariants[index_component][2][2]);
        likelihood = 1.0f / sqrt(m_covariant_Determinants[index_component]) * exp(-0.5f * multipication);
    }
    return likelihood;
}

double_t Gmm::calcSumOfLikelihood(const cv::Vec3d color) const {
    double_t likelihood = 0.0;
    for(int32_t index_component = 0; index_component < kComponentsCount; index_component++)
        likelihood += m_coefficients[index_component] * calcLikelihoodInComponent(index_component, color);
    return likelihood;
}

/*  Log output operator */
google::LogMessage& operator<<(google::LogMessage& lhs, const Gmm& rhs) {
  lhs.stream() << "cvgraphcut_base::Gmm{" <<
      // TODO(N.Takayama): implement out stream of memder data
      "}" << std::endl;
  return lhs;
}

/*--- Accessor --------------------------------------------------------------*/
int32_t Gmm::whichComponent(const cv::Vec3d color) const {
    int32_t which = 0;
    double_t max = 0.0;
    for(int32_t index_component = 0; index_component < kComponentsCount; index_component++) {
        double_t likelihood = calcLikelihoodInComponent(index_component, color);
        if(likelihood > max) {
            which = index_component;
            max = likelihood;
        }
    }
    return which;
}


/*--- Event -----------------------------------------------------------------*/

}  // namespace cvgraphcut_base


