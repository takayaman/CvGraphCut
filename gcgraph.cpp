/*=============================================================================
 * Project : CvGraphCut
 * Code : gcgraph.cpp
 * Written : N.Takayama, UEC
 * Date : 2014/12/ 3
 * Copyright (c) 2014 N.Takayama <takayaman@uec.ac.jp>
 * Implementation of cvgraphcut_base::Gcgraph
 * graph for graph cut
 *===========================================================================*/

/*=== Include ===============================================================*/

#include <gcgraph.h>
#include <stdint.h>
#include <glog/logging.h>

/*=== Local Define / Local Const ============================================*/

/*=== Class Implementation ==================================================*/

namespace cvgraphcut_base {


/*--- Constructor / Destructor / Initialize ---------------------------------*/

/* Defoult constructor */
GCGraph::GCGraph(void)
    : m_flow(0.0) {
    m_vertexes.clear();
    m_edges.clear();
}

GCGraph::GCGraph(uint8_t vertexcount, uint8_t edgecount) {
    create(vertexcount, edgecount);
}

/* Default destructor */
GCGraph::~GCGraph(void) {
}

/*  Copy constructor */
GCGraph::GCGraph(const GCGraph& rhs) {
}

/* Assignment operator */
GCGraph& GCGraph::operator=(const GCGraph& rhs) {
  if (this != &rhs) {
    // TODO(N.Takayama): implement copy
  }
  return *this;
}

void GCGraph::create(uint8_t vertexcount, uint8_t edgecount) {
    m_vertexes.reserve(vertexcount);
    m_edges.reserve(edgecount + 2);
    m_flow = 0.0;
}

/*--- Operation -------------------------------------------------------------*/
int32_t GCGraph::addVertex(void) {
    GCVertex vertex;
    memset(&vertex, 0, sizeof(GCVertex));
    m_vertexes.push_back(vertex);
    return static_cast<int32_t>(m_vertexes.size() - 1);
}

void GCGraph::addEdges(int32_t i, int32_t j, double_t weight, double_t reverseweight) {
    /* 引数チェック */
    CV_Assert(0 <= i && i < static_cast<int32_t>(m_vertexes.size()));
    CV_Assert(0 <= j && j < static_cast<int32_t>(m_vertexes.size()));
    CV_Assert(0 <= weight && 0 <= reverseweight);
    CV_Assert(i != j);

    /* 少なくとも頂点間の双方向グラフ分を確保 */
    if(!m_edges.size())
        m_edges.resize(2);

    GCEdge fromI, toI;
    /* i -> jへのエッジ追加 */
    fromI.destination_vertex = j; /* 終点設定 */
    /* 頂点に設定してある初期探索エッジを次の探索エッジとして設定 */
    fromI.nextsearch_edge = m_vertexes[i].firstsearch_edge;
    fromI.weight = weight;
    /* 自身を頂点の初期探索エッジに更新 */
    m_vertexes[i].firstsearch_edge = static_cast<int32_t>(m_edges.size());
    m_edges.push_back(fromI);

    /* j -> iへのエッジ追加 */
    toI.destination_vertex = i;
    toI.nextsearch_edge = m_vertexes[j].firstsearch_edge;
    toI.weight = reverseweight;
    m_vertexes[j].firstsearch_edge = static_cast<int32_t>(m_edges.size());
    m_edges.push_back(toI);
}

void GCGraph::addTerminalWeights(int32_t i, double_t sourceweight, double_t sinkweight) {
    CV_Assert(0 <= i && static_cast<int32_t>(m_vertexes.size()));

    double_t offset_weight = m_vertexes[i].weight;
    if(0 < offset_weight)
        sourceweight += offset_weight;
    else
        sinkweight -= offset_weight;
    /* 少い方が優先される? */
    m_flow += (sourceweight < sinkweight) ? sourceweight : sinkweight;
    /* 頂点weightが正 : source側, 負 : sink側 */
    m_vertexes[i].weight = sourceweight - sinkweight;
}


double_t GCGraph::maxFlow(void) {
    const int32_t TERMINAL = -1, ORPHAN = -2;
    GCVertex stub, *nilvertex = &stub;
    /* first : 探索頂点群の先頭, last : 末尾 (queue) */
    GCVertex *first = nilvertex, *last = nilvertex;
    int32_t current_ts = 0;
    stub.next = nilvertex;
    GCVertex *p_vertex = &m_vertexes[0];
    GCEdge *p_edge = &m_edges[0];

    std::vector<GCVertex *> orphans; // 探索パスから外れた頂点

    /* 探索パス(queue)の初期化 */
    for(size_t i =0; i < m_vertexes.size(); i++) {
        GCVertex *vertex = p_vertex + i;
        vertex->searchtreeid = 0;
        if(0 != vertex->weight) {
            /* 最初だけstub.next,fist.next,last.next
             * が同じところを指しているので一斉に更新される
             */
            last->next = vertex;
            last = vertex;
            //last = last->next = vertex;

            vertex->distance = 1;
            vertex->parent = TERMINAL;
            /* weightが正 : source側(0, false), 負 : sink側(other, true) */
            vertex->treeid = vertex->weight < 0;
        } else
            vertex->parent = 0;
    }
    first = first->next;
    last->next = nilvertex;
    nilvertex->next = 0;

    /* 最大流探索開始
     * Step1. 探索パス構築
     * Step2. パスにフローを流す
     * Step3. ツリーの修正
    */
    for(;;) {
        GCVertex *vertex0, *vertex1;
        int32_t edgeid_0 = -1, edgeid_i = 0, edgeid_j = 0;
        double_t minweight, weight;
        uint8_t vertex_treeid; /* 0(false) : Source, other(true) Sink */

        /* Step1. SourceとSinkから探索パスを伸ばす.この操作は互いのパスが出会うまで続ける */
        while(nilvertex != first) {
            vertex0 = first;
            if(vertex0->parent) {
                vertex_treeid = vertex0->treeid;
                for(edgeid_i = vertex0->firstsearch_edge; edgeid_i != 0; edgeid_i = p_edge[edgeid_i].nextsearch_edge) {
                    /* XORによる奇遇判定 s->t, t->sへの方向をvertex_treeと合わせる */
                    int32_t debug = edgeid_i^vertex_treeid;
                    if(0 == p_edge[edgeid_i^vertex_treeid].weight)
                        continue;
                    /* vertex0の隣接頂点 */
                    vertex1 = p_vertex + p_edge[edgeid_i].destination_vertex;
                    /* vertex1がツリーに所属していない場合 */
                    if(!vertex1->parent) {
                        vertex1->treeid = vertex_treeid;
                        /* vertex1 -> vertex0へのエッジを親とする */
                        vertex1->parent = edgeid_i ^ 1;
                        /* vertex0と同じ探索ツリーに追加 */
                        vertex1->searchtreeid = vertex0->searchtreeid;
                        /* 探索上vertex0を経由するので距離が1増える */
                        vertex1->distance = vertex1->distance + 1;
                        if(!vertex1->next) {
                            vertex1->next = nilvertex;
                            last = last->next = vertex1;
                        }
                        continue;
                    }
                    /* vertex1が違うツリーに所属している場合,ツリーの成長を止める */
                    if(vertex1->treeid != vertex_treeid) {
                        edgeid_0 = edgeid_i ^ vertex_treeid;
                        break;
                    }
                    /* vertex1を加える過程で値の整合性が取れなくなった場合修正する
                     * 高速化のため?
                     */
                    if(vertex1->distance > vertex0->distance + 1 && vertex1->searchtreeid <= vertex0->searchtreeid) {
                        /* parentの再設定 */
                        vertex1->parent = edgeid_i ^ 1;
                        vertex1->searchtreeid = vertex0->searchtreeid;
                        vertex1->distance = vertex0->distance + 1;
                    }
                }
                /* SourceツリーとSinkツリーが出会った */
                if(0 < edgeid_0)
                    break;
            }
            /* 頂点をアクティブリストから除外 */
            first = first->next;
            vertex0->next = 0;
        }
        /* アクティブリストが無くなった */
        if(0 >= edgeid_0)
            break;

        /* パスに沿って最小のエッジの重みを探索 */
        minweight = p_edge[edgeid_0].weight;
        CV_Assert(0 < minweight);

        /* k = 1: source tree, k = 0: destination tree */
        for(int32_t k = 1; k >= 0; k--) {
            for(vertex0 = p_vertex + p_edge[edgeid_0^k].destination_vertex; ; vertex0 = p_vertex + p_edge[edgeid_i].destination_vertex) {
                edgeid_i = vertex0->parent;
                if(0 > edgeid_i)
                    break;
                weight = p_edge[edgeid_i^k].weight;
                minweight = MIN(minweight, weight);
            }
            weight = fabs(vertex0->weight);
            minweight = MIN(minweight, weight);
            CV_Assert(0 < minweight);
        }

        /* パスに沿ってエッジの重みを修正し,orphansを修正する */
        p_edge[edgeid_0].weight -= minweight;
        p_edge[edgeid_0^1].weight += minweight;
        m_flow += minweight;

        /* k = 1: source tree, k = 0: destination tree */
        for(int32_t k = 1; k >= 0; k--) {
            for(vertex0 = p_vertex + p_edge[edgeid_0^k].destination_vertex; ; vertex0 = p_vertex + p_edge[edgeid_i].destination_vertex) {
                edgeid_i = vertex0->parent;
                if(0 > edgeid_i)
                    break;
                p_edge[edgeid_i^(k^1)].weight += minweight;
                p_edge[edgeid_i^k].weight -= minweight;
                if(0 == p_edge[edgeid_i^k].weight) {
                    orphans.push_back(vertex0);;
                    vertex0->parent = ORPHAN;
                }
            }
            vertex0->weight = vertex0->weight + minweight * (1 - k * 2);
            if(0 == vertex0->weight) {
                orphans.push_back(vertex0);
                vertex0->parent = ORPHAN;
            }
        }

        /* orphansに対して新たなparentsを見つけることで探索パスを保存する */
        current_ts++;
        while(!orphans.empty()) {
            GCVertex *vertex2 = orphans.back();
            orphans.pop_back();

            int32_t distance, mindistance = INT32_MAX;
            edgeid_0 = 0;
            vertex_treeid = vertex2->treeid;

            for(edgeid_i = vertex2->firstsearch_edge; edgeid_i != 0; edgeid_i = p_edge[edgeid_i].nextsearch_edge) {
                if(0 == p_edge[edgeid_i^(vertex_treeid^1)].weight)
                    continue;
                vertex1 = p_vertex + p_edge[edgeid_i].destination_vertex;
                if(vertex1->treeid != vertex_treeid || vertex1->parent == 0)
                    continue;
                /* rootまでの距離を計算 */
                for(distance = 0 ; ; ) {
                    if(vertex1->searchtreeid == current_ts) {
                        distance += vertex1->distance;
                        break;
                    }
                    edgeid_j = vertex1->parent;
                    distance++;
                    if(0 > edgeid_j) {
                        if(edgeid_j == ORPHAN)
                            distance = INT32_MAX - 1;
                        else {
                            vertex1->searchtreeid = current_ts;
                            vertex1->distance = 1;
                        }
                        break;
                    }
                    vertex1 = p_vertex + p_edge[edgeid_j].destination_vertex;
                }

                /* distanceを更新 */
                if(INT32_MAX > ++distance) {
                    if(distance < mindistance) {
                        mindistance = distance;
                        edgeid_0 = edgeid_i;
                    }
                    for(vertex1 = p_vertex + p_edge[edgeid_i].destination_vertex; vertex1->searchtreeid != current_ts; vertex1 = p_vertex + p_edge[vertex1->parent].destination_vertex) {
                        vertex1->searchtreeid = current_ts;
                        vertex1->distance = --distance;
                    }
                }
            }

            vertex2->parent = edgeid_0;
            if(0 < vertex2->parent) {
                vertex2->searchtreeid = current_ts;
                vertex2->distance = mindistance;
                continue;
            }

            /* parentが見つからない */
            vertex2->searchtreeid = 0;
            for(edgeid_i = vertex2->firstsearch_edge; edgeid_i != 0; edgeid_i = p_edge[edgeid_i].nextsearch_edge) {
                vertex1 = p_vertex + p_edge[edgeid_i].destination_vertex;
                GCEdge debug = p_edge[edgeid_i];
                edgeid_j = vertex1->parent;
                if(vertex1->treeid != vertex_treeid || !edgeid_j)
                    continue;
                if(p_edge[edgeid_i^(vertex_treeid^1)].weight && !vertex1->next) {
                    vertex1->next = nilvertex;
                    last = last->next = vertex1;
                }
                if(0 < edgeid_j && (p_vertex + p_edge[edgeid_j].destination_vertex) == vertex2) {
                    orphans.push_back(vertex1);
                    vertex1->parent = ORPHAN;
                }
            }
        }
    }
    return m_flow;
}

/*  Log output operator */
google::LogMessage& operator<<(google::LogMessage& lhs, const GCGraph& rhs) {
  lhs.stream() << "cvgraphcut_base::Gcgraph{" <<
      // TODO(N.Takayama): implement out stream of memder data
      "}" << std::endl;
  return lhs;
}

/*--- Accessor --------------------------------------------------------------*/
bool GCGraph::inSourceSegment(int32_t index_vertex) {
    CV_Assert(0 <= index_vertex && static_cast<int32_t>(m_vertexes.size()) > index_vertex);
    return m_vertexes[index_vertex].treeid == 0;
}

/*--- Event -----------------------------------------------------------------*/

}  // namespace cvgraphcut_base


