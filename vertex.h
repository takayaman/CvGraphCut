/*=============================================================================
 * Project : CvGraphCut
 * Code : vertex.h
 * Written : N.Takayama, UEC
 * Date : 2014/11/29
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Definition of cvgraphcut_base::Vertex
 * Vertex of image graph to segment image
 *===========================================================================*/

#ifndef _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_VERTEX_H_
#define _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_VERTEX_H_

/*=== Include ===============================================================*/

#include <stdint.h>
#include <glog/logging.h>

#include <opencv2/opencv.hpp>

#include "edge.h"

/*=== Define ================================================================*/

/*=== Class Definition  =====================================================*/

namespace cvgraphcut_base {

class Edge;

class Vertex {
public:
    typedef enum VertexType_TAG {
        TYPE_DEFAULT = 0,
        TYPE_NEUTRAL = TYPE_DEFAULT,
        TYPE_SINK,
        TYPE_SOURCE,
        TYPE_NUM,
    }VertexType;

    /*!
    * Defoult constructor
    */
    Vertex(void);

    Vertex(cv::Point2d point);

    /*!
    * Default destructor
    */
    ~Vertex(void);

    /*!
    * Copy constructor
    */
    Vertex(const Vertex& rhs);

    /*!
    * Assignment operator
    * @param rhs Right hand side
    * @return pointer of this object
    */
    Vertex& operator=(const Vertex& rhs);

    bool operator ==(Vertex& rhs);

    /*! 自身を始点として終点との間に双方向エッジを張り,容量を設定する
    * ここで,始点->終点のエッジ容量 : size
    * 終点->始点のエッジ容量 : -size
    */
    void addFlowPath(Vertex *endvertex, double size);
    void visit(void);
    void reset(void);

    void setValue(double value);
    double getValue(void);
    void setPoint(cv::Point2d point);

    bool isVisited(void);

    void setVertexType(VertexType type);
    VertexType getVertexType(void);

public:
    std::vector<Edge*> m_vectorflowpath;
    Vertex* m_parent;
    Edge* m_parentpath;
    cv::Point2d m_point;


private:
    double m_value;
    bool is_visited;
    VertexType m_vertextype;

};

/*!
 * Log output operator
 * @param lhs Left hand side
 * @param rhs Right hand side
 * @return Pointer of google::LogSink object
 */
google::LogMessage& operator<<(google::LogMessage& lhs, const Vertex& rhs);

}  // namespace cvgraphcut_base


#endif  // _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_VERTEX_H_
