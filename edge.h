/*=============================================================================
 * Project : CvGraphCut
 * Code : edge.h
 * Written : N.Takayama, UEC
 * Date : 2014/11/29
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Definition of cvgraphcut_base::Edge
 * Edge of image graph to segment image
 *===========================================================================*/

#ifndef _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_EDGE_H_
#define _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_EDGE_H_

/*=== Include ===============================================================*/

#include <stdint.h>
#include <glog/logging.h>

#include "vertex.h"

/*=== Define ================================================================*/

/*=== Class Definition  =====================================================*/

namespace cvgraphcut_base {

class Vertex;

class Edge {
 public:
  /*!
  * Defoult constructor
  */
  Edge(void);

  Edge(Vertex *startvertex, Vertex *endvertex, double size);

  /*!
  * Default destructor
  */
  ~Edge(void);

  /*!
  * Copy constructor
  */
  Edge(const Edge& rhs);

  /*!
  * Assignment operator
  * @param rhs Right hand side
  * @return pointer of this object
  */
  Edge& operator=(const Edge& rhs);

  double addFlow(double flow);
  void setFlow(double flow);
  double getFlow(void) const;

  void setReverseEdge(Edge* reverseedge);
  Edge* getReverseEdge(void) const;
  double getSize(void) const;


 public:
  Vertex* m_startvertex;
  Vertex* m_endvertex;

 private:
  Edge* m_reverseedge;
  double m_flow;
  double m_size;

};

/*!
 * Log output operator
 * @param lhs Left hand side
 * @param rhs Right hand side
 * @return Pointer of google::LogSink object
 */
google::LogMessage& operator<<(google::LogMessage& lhs, const Edge& rhs);

}  // namespace cvgraphcut_base


#endif  // _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_EDGE_H_
