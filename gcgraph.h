/*=============================================================================
 * Project : CvGraphCut
 * Code : gcgraph.h
 * Written : N.Takayama, UEC
 * Date : 2014/12/ 3
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Definition of cvgraphcut_base::GCgraph
 * graph for graphcut. This code is modified version of gcgraph of opencv.
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

#ifndef _CVGRAPHCUT_CVGRAPHCUT_GCGRAPH_H_
#define _CVGRAPHCUT_CVGRAPHCUT_GCGRAPH_H_

/*=== Include ===============================================================*/

#include <stdint.h>
#include <glog/logging.h>

#include <opencv2/opencv.hpp>

/*=== Define ================================================================*/

/*=== Class Definition  =====================================================*/

namespace cvgraphcut_base {

/** Graph for cutting based on maxflow algorithm.
 * The algorithm using in this class is based on [Boykov'04]
 * [Boykov'04] Yuri Boykov and Vladimir Kolmogrov,
 * "An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision",
 * IEEE Trans. on PAMI, Vol.26, No.9, pp.1124-1137, Sep. 2004.
 */
class GCGraph {
 public:
  /** Defoult constructor.
   *
   */
  GCGraph(void);

  /** Constructor with reserve vertices and edges area.
   * @param vertexcount
   * @param edgecount
   */
  GCGraph(uint8_t vertexcount, uint8_t edgecount);

  /** Default destructor
   *
   */
  ~GCGraph(void);

  /** Copy constructor
   * @param rhs Right hand side
   */
  GCGraph(const GCGraph& rhs);

  /** Assignment operator
   * @param rhs Right hand side
   * @return pointer of this object
   */
  GCGraph& operator=(const GCGraph& rhs);

  /** Reserve vertices and edges area
   * @param vertexcount
   * @param edgecount
   * @return None
   */
  void create(uint8_t vertexcount, uint8_t edgecount);

  /** Add Vertex to graph
   * @return Index of added vertex
   */
  int32_t addVertex(void);

  /** Add an bi-direction edge between two vertices and assign weight to edge.
   * @param i Index of one vertex
   * @param j Index of other vertex
   * @param weight A weight of edge directed from i to j.
   * @param reverseweight A weight of edge directed from j to i.
   * @return None
   */
  void addEdges(int32_t i, int32_t j, double_t weight, double_t reverseweight);

  /** Add two directional edges and assign weights.
   * One edge is directed from source to i, and other is from i to sink.
   * @param i Index of a vertex.
   * @param sourceweight A weight of edge directed from source to i.
   * @param sinkweight A weight of edge directed from i to sink.
   * @return None
   */
  void addTerminalWeights(int32_t i, double_t sourceweight, double_t sinkweight);

  /** Run maxflow algorithm to cut a graph.
   * @return Total flow.
   */
  double_t maxFlow(void);

  /** Return whether the vertex is in source segment.
   * @param index_vertex Index of vertex
   * @return true : in source, false : in sink
   */
  bool inSourceSegment(int32_t index_vertex);

 private:
  /** Vertex class for cutting graph.
   */
  class GCVertex {
   public:
    GCVertex *next;  /**< next vertex to search */
    int32_t parent;                     /**< parent vertex */
    int32_t firstsearch_edge;           /**< The edge to search first */
    int32_t searchtreeid;               /**< Treeid to search */
    int32_t distance;                   /**< Num of edges from tree root */
    double_t weight;                    /**< A weidght to tree root(source/sink) */
    uint8_t treeid;  /**< 0:Source, other:Sink */
  };

  /** Edge class for cutting graph.
   */
  class GCEdge {
   public:
    int32_t destination_vertex;         /**< Index of destination vertex */
    int32_t nextsearch_edge;            /**< Index of the edge of searching next */
    double_t weight;                    /**< A weight between vertices */
  };

  std::vector<GCVertex> m_vertexes;
  std::vector<GCEdge> m_edges;
  double_t m_flow;

};

/*!
 * Log output operator
 * @param lhs Left hand side
 * @param rhs Right hand side
 * @return Pointer of google::LogSink object
 */
google::LogMessage& operator<<(google::LogMessage& lhs, const GCGraph& rhs);

}  // namespace cvgraphcut_base


#endif  // _CVGRAPHCUT_CVGRAPHCUT_GCGRAPH_H_
