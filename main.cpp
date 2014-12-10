/*=============================================================================
 * Project : CvGraphCut
 * Code : main.cpp
 * Written : N.Takayama, UEC
 * Date : 2014/11/29
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * GraphCut for image segmentation
 *===========================================================================*/

/*=== Include ===============================================================*/

#include <stdint.h>
#include <time.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "gcgraph.h"
#include "siftdata.h"
#include "siftmatcher.h"
#include "siftgraphbuilder.h"

/*=== Local Define / Local Const ============================================*/

static const int32_t DefPatchRadius = 10;

/* SiftDetectorのパラメータ */
static const double_t DefSiftNumFeatures = 0;
static const double_t DefSiftOctaveLayers = 3;
static const double_t DefSiftContrastThreshold = 0.04;
static const double_t DefSiftEdgeThreshold = 10;
static const double_t DefSiftSigma = 1.6;


/*=== Local Variable ========================================================*/
cv::Mat focused_image;
cv::Mat blurred_image;

/*=== Local Function Define =================================================*/


/*=== Local Function Implementation =========================================*/

/*=== Global Function Implementation ========================================*/
int main(int argc, char *argv[]) {
    /* Initialize */
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    /* 画像読み込み(2枚) */
    if(2 > argc) {
        LOG(ERROR) << "Usage : CvGraphCut [focused_image] [blurred_image]" << std::endl;
        return EXIT_FAILURE;
    }

    focused_image = cv::imread(argv[1]);
    if(focused_image.empty()) {
        LOG(ERROR) << "Can not read " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }
    blurred_image = cv::imread(argv[2]);
    if(blurred_image.empty()) {
        LOG(ERROR) << "Can not read " << argv[2] << std::endl;
        return EXIT_FAILURE;
    }
    /* 最終結果表示用にコピー */
    cv::Mat cutresultimage = focused_image.clone();
    cv::Mat delaunayresultimage = focused_image.clone();
    cv::Mat volonoyresultimage = focused_image.clone();
    cv::Mat modelresultimage = focused_image.clone();

    /* SIFT + マッチング */
    cvgraphcut_base::SiftData focused_data(focused_image);
    cvgraphcut_base::SiftData blurred_data(blurred_image);
    focused_data.setSiftParams(DefSiftNumFeatures,
                               DefSiftOctaveLayers,
                               DefSiftContrastThreshold,
                               DefSiftEdgeThreshold,
                               DefSiftSigma);
    blurred_data.setSiftParams(DefSiftNumFeatures,
                               DefSiftOctaveLayers,
                               DefSiftContrastThreshold,
                               DefSiftEdgeThreshold,
                               DefSiftSigma);
    focused_data.build();
    blurred_data.build();

    if(!focused_data.isBuilded() || !blurred_data.isBuilded()) {
        LOG(ERROR) << "Sift building is failed!!" << std::endl;
        return EXIT_FAILURE;
    }
    cvgraphcut_base::SiftMatcher matcher(focused_data, blurred_data);
    matcher.matching();

    if(!matcher.isBuildMatchGroups()) {
        LOG(ERROR) << "Sift matching is failed!!" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<std::vector<cvgraphcut_base::SiftMatchPare> > match_groups = matcher.getMatchGroups();

    /* ドロネー分割によるグラフ構築 */
    cvgraphcut_base::SiftGraphBuilder gcbuilder(match_groups, cv::Rect(cv::Point(0, 0), cv::Point(focused_image.cols, focused_image.rows)));
    gcbuilder.build();
    gcbuilder.cutGraph();

    /* 結果描画 */
    gcbuilder.drawCuttedPoints(cutresultimage, cvgraphcut_base::SiftGraphBuilder::DRAW_BOTH);
    gcbuilder.drawGraphs(delaunayresultimage, cvgraphcut_base::SiftGraphBuilder::DRAW_DIV_DELAUNAY);
    gcbuilder.drawGraphs(volonoyresultimage, cvgraphcut_base::SiftGraphBuilder::DRAW_DIV_VORONOY);

    /* 前景モデル */
    gcbuilder.drawSiftGroups(modelresultimage, 0, cv::Scalar(0, 0, 255));
    gcbuilder.drawSiftGroups(modelresultimage, 1, cv::Scalar(0, 0, 255));
    gcbuilder.drawSiftGroups(modelresultimage, 2, cv::Scalar(0, 0, 255));

    /* 背景モデル */
    gcbuilder.drawSiftGroups(modelresultimage, 3, cv::Scalar(255, 0, 0));
    gcbuilder.drawSiftGroups(modelresultimage, 4, cv::Scalar(255, 0, 0));
    gcbuilder.drawSiftGroups(modelresultimage, 12, cv::Scalar(255, 0, 0));

    //cv::imshow("Result", resultimage);
    //cv::waitKey(0);

    cv::imwrite("cutresult.png", cutresultimage);
    cv::imwrite("delaunay.png", delaunayresultimage);
    cv::imwrite("volonoy.png", volonoyresultimage);
    cv::imwrite("model.png", modelresultimage);

    /* Finalize */
    google::InstallFailureSignalHandler();


    return EXIT_SUCCESS;
}

