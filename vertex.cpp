/*=============================================================================
 * Project : CvGraphCut
 * Code : vertex.cpp
 * Written : N.Takayama, UEC
 * Date : 2014/11/29
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Implementation of cvgraphcut_base::Vertex
 * Vertex of image graph to segment image
 *===========================================================================*/

/*=== Include ===============================================================*/

#include <vertex.h>
#include <stdint.h>
#include <glog/logging.h>

/*=== Local Define / Local Const ============================================*/

/*=== Class Implementation ==================================================*/

namespace cvgraphcut_base {


/*--- Constructor / Destructor / Initialize ---------------------------------*/

/* Defoult constructor */
Vertex::Vertex(void)
    : m_parent(NULL),
      m_parentpath(NULL),
      m_point(cv::Point2d(0, 0)),
      m_value(0.0),
      is_visited(false)
{
    m_vectorflowpath.clear();
}

Vertex::Vertex(std::string name)
    : m_parent(NULL),
      m_parentpath(NULL),
      m_point(cv::Point2d(0, 0)),
      m_value(0.0),
      is_visited(false)
{
    m_vectorflowpath.clear();
}

/* Default destructor */
Vertex::~Vertex(void) {
    while(!m_vectorflowpath.empty()){
        if(NULL != m_vectorflowpath.back())
            delete m_vectorflowpath.back();
        m_vectorflowpath.pop_back();
    }
}

/*  Copy constructor */
Vertex::Vertex(const Vertex& rhs) {
}

/* Assignment operator */
Vertex& Vertex::operator=(const Vertex& rhs) {
  if (this != &rhs) {
    // TODO(N.Takayama): implement copy
  }
  return *this;
}

bool Vertex::operator ==(Vertex& rhs){
    return this->m_point == rhs.m_point;
}

/*--- Operation -------------------------------------------------------------*/
void Vertex::addFlowPath(Vertex *endvertex, double size)
{
    Edge* forwardedge = new Edge(this, endvertex, size);
    m_vectorflowpath.push_back(forwardedge);
    Edge* reverseedge = new Edge(endvertex, this, -size);
    reverseedge->setReverseEdge(forwardedge);
    forwardedge->setReverseEdge(reverseedge);
    endvertex->m_vectorflowpath.push_back(reverseedge);
}

void Vertex::visit(void)
{
    is_visited = true;
}

void Vertex::reset(void)
{
    is_visited = false;
    m_value = 0.0;
}

/*  Log output operator */
google::LogMessage& operator<<(google::LogMessage& lhs, const Vertex& rhs) {
  lhs.stream() << "cvgraphcut_base::Vertex{" <<
      // TODO(N.Takayama): implement out stream of memder data
      "}" << std::endl;
  return lhs;
}

/*--- Accessor --------------------------------------------------------------*/
void Vertex::setValue(double value)
{
    m_value = value;
}

double Vertex::getValue(void)
{
    return m_value;
}

void Vertex::setPoint(cv::Point2d point)
{
    m_point = point;
}

bool Vertex::isVisited(void)
{
    return is_visited;
}


void Vertex::setVertexType(VertexType type) {
    return m_vertextype;
}

Vertex::VertexType Vertex::getVertexType(void){
    m_vertextype = type;
}


/*--- Event -----------------------------------------------------------------*/

}  // namespace cvgraphcut_base


