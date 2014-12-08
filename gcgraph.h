/*=============================================================================
 * Project : CvGraphCut
 * Code : gcgraph.h
 * Written : N.Takayama, UEC
 * Date : 2014/12/ 3
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Definition of cvgraphcut_base::Gcgraph
 * graph for graphcut
 *===========================================================================*/

#ifndef _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_GCGRAPH_H_
#define _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_GCGRAPH_H_

/*=== Include ===============================================================*/

#include <stdint.h>
#include <glog/logging.h>

#include <opencv2/opencv.hpp>

/*=== Define ================================================================*/

/*=== Class Definition  =====================================================*/

namespace cvgraphcut_base {


class GCGraph {
public:
    /*!
    * Defoult constructor
    */
    GCGraph(void);

    GCGraph(uint8_t vertexcount, uint8_t edgecount);

    /*!
    * Default destructor
    */
    ~GCGraph(void);

    /*!
    * Copy constructor
    */
    GCGraph(const GCGraph& rhs);

    /*!
    * Assignment operator
    * @param rhs Right hand side
    * @return pointer of this object
    */
    GCGraph& operator=(const GCGraph& rhs);

    void create(uint8_t vertexcount, uint8_t edgecount);
    int32_t addVertex(void);
    void addEdges(int32_t i, int32_t j, double_t weight, double_t reverseweight);
    void addTerminalWeights(int32_t i, double_t sourceweight, double_t sinkweight);
    double_t maxFlow(void);
    bool inSourceSegment(int32_t index_vertex);

private:
    class GCVertex {
    public:
        GCVertex *next; // 次の頂点?
        int32_t parent;
        int32_t firstsearch_edge;
        int32_t searchtreeid;
        int32_t distance;
        double_t weight;
        uint8_t treeid; // 0 だったらSource, 他はSink
    };

    class GCEdge {
    public:
        int32_t destination_vertex;
        int32_t nextsearch_edge;
        double_t weight;
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


#endif  // _HOME_TAKAYAMAN_DOCUMENTS_PROGRAMMING_OPENCV_CVGRAPHCUT_CVGRAPHCUT_GCGRAPH_H_
