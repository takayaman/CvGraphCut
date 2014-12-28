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
#include <sys/stat.h>
#include <dirent.h>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "fstream"

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

int remove_directory(const char *path);

/*=== Local Function Implementation =========================================*/

int remove_directory(const char *path) {
  DIR *d = opendir(path);
  size_t path_len = strlen(path);
  int r = -1;
  if (d) {
    struct dirent *p;
    r = 0;
    while (!r && (p=readdir(d))) {
      int r2 = -1;
      char *buf;
      size_t len;
      /* Skip the names "." and ".." as we don't want to recurse on them. */
      if (!strcmp(p->d_name, ".") || !strcmp(p->d_name, "..")) {
        continue;
      }
      len = path_len + strlen(p->d_name) + 2;
      buf = static_cast<char *>(malloc(len));
      if (buf) {
        struct stat statbuf;
        snprintf(buf, len, "%s/%s", path, p->d_name);

        if (!stat(buf, &statbuf)) {
          if (S_ISDIR(statbuf.st_mode)) {
            r2 = remove_directory(buf);
          } else {
            r2 = unlink(buf);
          }
        }
        free(buf);
      }
      r = r2;
    }
    closedir(d);
  }
  if (!r) {
    r = rmdir(path);
  }
  return r;
}

/*=== Global Function Implementation ========================================*/
int main(int argc, char *argv[]) {
  /* Initialize */
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;

  /* 画像読み込み(2枚) */
  if(3 > argc) {
    LOG(ERROR) << "Usage : CvGraphCut [focused_image] [blurred_image] [gamma] " << std::endl;
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

  double_t gamma = std::atof(argv[3]);

  char stringbuff[PATH_MAX];
  getcwd(stringbuff, sizeof(stringbuff));
  chdir("./result");
  getcwd(stringbuff, sizeof(stringbuff));
  std::string dirname = "temp_gamma_" + std::to_string(gamma);
  int32_t check = mkdir(dirname.c_str(), 0775);
  dirname = "./temp_gamma_" + std::to_string(gamma);
  if(0 == check) {
    chdir(dirname.c_str());
    getcwd(stringbuff, sizeof(stringbuff));
  } else {
    int32_t status = remove_directory(dirname.c_str());
    if(0 != status) {
      LOG(ERROR) << "Failed to remove directory " << dirname << std::endl;
      return EXIT_FAILURE;
    }
    status = mkdir(dirname.c_str(), 0775);
    if(0 != status) {
      LOG(ERROR) << "Failed to make directory " << dirname << std::endl;
      return EXIT_FAILURE;
    }
    chdir(dirname.c_str());
    getcwd(stringbuff, sizeof(stringbuff));
  }

  FLAGS_logtostderr = false;
  FLAGS_log_dir = stringbuff;

  /* 最終結果表示用にコピー */
  cv::Mat cutresultimage = focused_image.clone();
  cv::Mat delaunayresultimage = focused_image.clone();
  cv::Mat siftgraphresultimage = focused_image.clone();
  cv::Mat volonoyresultimage = focused_image.clone();
  cv::Mat modelresultimage = focused_image.clone();
  cv::Mat matchingresultimage;

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
  std::vector<cvgraphcut_base::SiftMatchPare> unmatch_group = matcher.getUnMatchGroup();

  cv::Rect fore_rect = cv::Rect(cv::Point(380, 190), cv::Point(600, 500));
  cv::Rect back_rect = cv::Rect(cv::Point(0, 0), cv::Point(700, 160));
  cv::Rect image_area = cv::Rect(cv::Point(0, 0), cv::Point(focused_image.cols, focused_image.rows));
  /* ドロネー分割によるグラフ構築 */
  cvgraphcut_base::SiftGraphBuilder gcbuilder(match_groups, unmatch_group, image_area, fore_rect, back_rect);
  gcbuilder.setGamma(gamma);

  gcbuilder.build();


  gcbuilder.drawCuttedPoints(cutresultimage, cvgraphcut_base::SiftGraphBuilder::DRAW_BOTH);
  gcbuilder.drawGraphs(delaunayresultimage, cvgraphcut_base::SiftGraphBuilder::DRAW_DIV_DELAUNAY);
  gcbuilder.drawGraphs2(siftgraphresultimage, cvgraphcut_base::SiftGraphBuilder::DRAW_DIV_DELAUNAY);
  gcbuilder.drawGraphs(volonoyresultimage, cvgraphcut_base::SiftGraphBuilder::DRAW_DIV_VORONOY);

  /* 前景モデル, 背景モデル */
  gcbuilder.drawSiftGroups(modelresultimage, cvgraphcut_base::SiftGraphBuilder::DRAW_MODEL_ALL, cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0));

  /* マッチング結果 */
  matcher.drawMatching(focused_image, blurred_image, matchingresultimage);

  for(size_t i = 0; i < match_groups.size(); i++) {
    std::string filename = "matching_" + std::to_string(i) + ".png";
    cv::Mat matchinggroupimage;
    matcher.drawMatchinOfGroup(focused_image, blurred_image, matchinggroupimage, i, true);
    cv::imwrite(filename, matchinggroupimage);
  }

  cv::imwrite("cutresult.png", cutresultimage);
  cv::imwrite("delaunay.png", delaunayresultimage);
  cv::imwrite("siftgraph.png", siftgraphresultimage);
  cv::imwrite("volonoy.png", volonoyresultimage);
  cv::imwrite("model.png", modelresultimage);
  cv::imwrite("matching.png", matchingresultimage);

  /* 理想的な条件のための点群抽出 */
  /* 前景
   * 0, 1, 2, 6, 7, 8, 9, 11, 16, 17, 18, 21, 22, 24
   */
  /* 背景
   * 3, 4, 5, 10, 12, 13, 14, 15, 19, 20, 23
   */
  FLAGS_logtostderr = true;
  std::ofstream fore_output;
  std::ofstream back_output;
  std::ofstream fore_groupoutput;
  std::ofstream back_groupoutput;
  fore_output.open("foresampling.cvs");
  if(!fore_output.is_open()) {
    LOG(ERROR) << "Can not open foresampling.cvs" << std::endl;
    return EXIT_FAILURE;
  }
  back_output.open("backsampling.cvs");
  if(!back_output.is_open()) {
    LOG(ERROR) << "Can not open backsampling.cvs" << std::endl;
    return EXIT_FAILURE;
  }
  for(size_t i = 0; i < match_groups.size(); i++) {
    for(size_t j = 0; j < match_groups[i].size(); j++) {
      cv::Point point = match_groups[i][j].m_source_key.pt;
      switch(i) {
      case 0:
      case 1:
      case 2:
      case 6:
      case 7:
      case 8:
      case 9:
      case 11:
      case 16:
      case 17:
      case 18:
      case 21:
      case 22:
      case 24: {
        fore_output << point.x << ","
                    << point.y << ","
                    << (int32_t)focused_image.at<cv::Vec3b>(point).val[0] << ","
                    << (int32_t)focused_image.at<cv::Vec3b>(point).val[1] << ","
                    << (int32_t)focused_image.at<cv::Vec3b>(point).val[2] << std::endl;
        std::string filename = "foresampling_" + std::to_string(i) + ".cvs";
        fore_groupoutput.open(filename, std::ios::app);
        if(!fore_groupoutput.is_open()) {
          LOG(ERROR) << "Can not open " << filename << std::endl;
        }
        fore_groupoutput << point.x << ","
                         << point.y << ","
                         << (int32_t)focused_image.at<cv::Vec3b>(point).val[0] << ","
                         << (int32_t)focused_image.at<cv::Vec3b>(point).val[1] << ","
                         << (int32_t)focused_image.at<cv::Vec3b>(point).val[2] << std::endl;
        fore_groupoutput.close();
        break;
      }
      case 3:
      case 4:
      case 5:
      case 10:
      case 12:
      case 13:
      case 14:
      case 15:
      case 19:
      case 20:
      case 23: {
        back_output << point.x << ","
                    << point.y << ","
                    << (int32_t)focused_image.at<cv::Vec3b>(point).val[0] << ","
                    << (int32_t)focused_image.at<cv::Vec3b>(point).val[1] << ","
                    << (int32_t)focused_image.at<cv::Vec3b>(point).val[2] << std::endl;
        std::string filename = "backsampling_" + std::to_string(i) + ".cvs";
        back_groupoutput.open(filename, std::ios::app);
        if(!back_groupoutput.is_open()) {
          LOG(ERROR) << "Can not open " << filename << std::endl;
        }
        back_groupoutput << point.x << ","
                         << point.y << ","
                         << (int32_t)focused_image.at<cv::Vec3b>(point).val[0] << ","
                         << (int32_t)focused_image.at<cv::Vec3b>(point).val[1] << ","
                         << (int32_t)focused_image.at<cv::Vec3b>(point).val[2] << std::endl;
        back_groupoutput.close();
        break;
      }
      }
    }
  }





  /* Finalize */
  google::InstallFailureSignalHandler();


  return EXIT_SUCCESS;
}

