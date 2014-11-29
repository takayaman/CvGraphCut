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
#include <glog/logging.h>

#include "graph.h"

/*=== Local Define / Local Const ============================================*/

/*=== Local Variable ========================================================*/

/*=== Local Function Define =================================================*/

/*=== Local Function Implementation =========================================*/

/*=== Global Function Implementation ========================================*/

int main(int argc, char *argv[]) {
  /* Initialize */
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;

  cvgraphcut_base::Graph graph;
  cvgraphcut_base::Vertex *vertex_a = graph.addVertex("a");
  cvgraphcut_base::Vertex *vertex_b = graph.addVertex("b");
  cvgraphcut_base::Vertex *vertex_c = graph.addVertex("c");
  cvgraphcut_base::Vertex *vertex_d = graph.addVertex("d");
  cvgraphcut_base::Vertex *vertex_e = graph.addVertex("e");
  cvgraphcut_base::Vertex *vertex_f = graph.addVertex("f");

  graph.addFlowPath(vertex_a, vertex_b, 6);
  graph.addFlowPath(vertex_a, vertex_c, 8);
  graph.addFlowPath(vertex_b, vertex_d, 6);
  graph.addFlowPath(vertex_b, vertex_e, 3);
  graph.addFlowPath(vertex_c, vertex_d, 3);
  graph.addFlowPath(vertex_c, vertex_e, 3);
  graph.addFlowPath(vertex_d, vertex_f, 8);
  graph.addFlowPath(vertex_e, vertex_f, 6);

  graph.m_source = vertex_a;
  graph.m_sink = vertex_f;

  graph.searchMaxFlow();
  // 頂点のvisitの初期化が必要
  graph.reset();
  graph.searchMinCut( graph.m_source );
  graph.displayMinCut();

  /* Finalize */
  google::InstallFailureSignalHandler();

  return EXIT_SUCCESS;
}

