/*=============================================================================
 * Project : CvGraphCut
 * Code : util.cpp
 * Written : N.Takayama, UEC
 * Date : 2014/12/ 5
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Implementation of cvgraphcut_base::Util
 * Utility
 *===========================================================================*/

/*=== Include ===============================================================*/

#include <util.h>
#include <stdint.h>
#include <glog/logging.h>


/*=== Local Define / Local Const ============================================*/

/*=== Class Implementation ==================================================*/

namespace cvgraphcut_base {


/*--- Constructor / Destructor / Initialize ---------------------------------*/

/* Defoult constructor */
Util::Util(void) {
}

/* Default destructor */
Util::~Util(void) {
}


/*--- Operation -------------------------------------------------------------*/
/* テンプレート化してutil.hに移動
double_t Util::calcDistance(cv::Point2f point0, cv::Point2f point1) {
    double_t diff_X2 = pow(fabs(point0.x - point1.x), 2.0);
    double_t diff_Y2 = pow(fabs(point0.y - point1.y), 2.0);
    return sqrt(diff_X2 + diff_Y2);
}
*/

/* テンプレート化してutil.hに移動
double_t Util::calcL2NormOfVectors(const cv::Mat &fvector0, const cv::Mat &fvector1, bool aside_cols) {
    double_t l2_norm = 0.0;
    if(aside_cols) {
        for(int32_t i = 0; i < fvector0.cols; i++)
            l2_norm += pow(abs(fvector0.at<uint8_t>(0, i) - abs(fvector1.at<uint8_t>(0, i))), 2.0);
    } else {
        for(int32_t i = 0; i < fvector0.cols; i++)
            l2_norm += pow(abs(fvector0.at<uint8_t>(i, 0) - abs(fvector1.at<uint8_t>(i, 0))), 2.0);
    }
    return sqrt(l2_norm);
}
*/

/*--- Accessor --------------------------------------------------------------*/

/*--- Event -----------------------------------------------------------------*/

}  // namespace cvgraphcut_base


