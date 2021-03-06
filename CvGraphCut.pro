INCLUDEPATH += /home/takayaman/opencv3/build/include \
  /home/takayaman/opencv3/build/include/opencv \
  /home/takayaman/opencv3/build/include/opencv2 \
  /usr/include

LIBS += `pkg-config --libs opencv` \
  -L/home/takayaman/opencv3/build/lib \
  -L/usr/lib/x86_64-linux-gnu \
  -lglog

QMAKE_CXXFLAGS += -std=c++11

TARGET = CvGraphCut

HEADERS += \
    vertex.h \
    graph.h \
    edge.h \
    gcgraph.h \
    siftmatcher.h \
    siftgraphbuilder.h \
    siftmatchpare.h \
    siftdata.h \
    util.h \
    gmm.h

SOURCES += \
    vertex.cpp \
    graph.cpp \
    edge.cpp \
    gcgraph.cpp \
    siftmatcher.cpp \
    siftgraphbuilder.cpp \
    siftmatchpare.cpp \
    siftdata.cpp \
    util.cpp \
    main.cpp \
    gmm.cpp
