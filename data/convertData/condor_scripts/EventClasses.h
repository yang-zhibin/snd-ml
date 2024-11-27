#define EventClasses_h
#include <vector>
#include "TObject.h"

class Id : public TObject {
public:
    int runId, eventId, partitionId, isMC;
    Id(int runId=-1, int eventId=-1, int partitionId=-1, int isMC=-1) : 
    runId(runId), eventId(eventId), partitionId(partitionId), isMC(isMC) {}
    virtual ~Id() {}

    void clear() {
        runId = -1;
        eventId = -1;
        partitionId = -1;
        isMC = -1;
    }

    ClassDef(Id, 1)  
};

class Label : public TObject {
public:
    int pdgCode;    // PDG code identifying the particle type
    float px, py, pz; // momentum components
    float x, y, z;    // coordinates of the first interaction point
    double scifi_avg_ver, scifi_avg_hor, DS_avg_ver, DS_avg_hor;
    double scifi_avg_x_pos, scifi_avg_y_pos, DS_avg_x_pos, DS_avg_y_pos;
    int veto1, veto2;
    int scifi1, scifi2, scifi3, scifi4, scifi5;
    int us1, us2, us3, us4, us5;
    int ds1, ds2, ds3, ds4;

    Label(int pdgCode = 0,
          float px = 0.0, float py = 0.0, float pz = 0.0,
          float x = 0.0, float y = 0.0, float z = 0.0)
    : pdgCode(pdgCode), px(px), py(py), pz(pz),
      x(x), y(y), z(z),
      scifi_avg_ver(-1.0), scifi_avg_hor(-1.0),
      DS_avg_ver(-1.0), DS_avg_hor(-1.0),
      scifi_avg_x_pos(-100.0), scifi_avg_y_pos(-100.0),
      DS_avg_x_pos(-100.0), DS_avg_y_pos(-100.0) ,
      veto1(0), veto2(0),
      scifi1(0), scifi2(0), scifi3(0), scifi4(0), scifi5(0),
      us1(0), us2(0), us3(0), us4(0), us5(0),
      ds1(0), ds2(0), ds3(0), ds4(0){
    }

    virtual ~Label() {}

    void clear() {
        pdgCode = 0;
        px = py = pz = 0.0;
        x = y = z = 0.0;

        scifi_avg_ver = scifi_avg_hor = -1.0;
        DS_avg_ver = DS_avg_hor = -1.0;
        scifi_avg_x_pos = scifi_avg_y_pos = -100.0;
        DS_avg_x_pos = DS_avg_y_pos = -100.0;

        veto1 = veto2 = 0;
        scifi1 = scifi2 = scifi3 = scifi4 = scifi5 =0;
        us1 = us2 = us3 = us4 = us5 =0;
        ds1 = ds2 = ds3 = ds4 =0;
    }

    ClassDef(Label, 1)
};

class MCTrack {
public:
    int trackId;
    float ratio;
    int pdgCode;
    int mother;
    float px, py, pz; // momentum components
    float x, y, z;    // start positions

    MCTrack(int trackId = 0, float ratio = 0.0, int pdgCode = 0, int mother = 0,
            float px = 0.0, float py = 0.0, float pz = 0.0,
            float x = 0.0, float y = 0.0, float z = 0.0)
    : trackId(trackId), ratio(ratio), pdgCode(pdgCode), mother(mother),
      px(px), py(py), pz(pz), x(x), y(y), z(z) {}

    void clear() {
        trackId = 0;
        ratio = 0.0;
        pdgCode = 0;
        mother = 0;
        px = py = pz = 0.0;
        x = y = z = 0.0;
    }
};

class Hit : public TObject {
public:
    bool orientation; // true for vertical (1), false for horizontal (0)
    float x1, y1, z1; // coordinates of one end
    float x2, y2, z2; // coordinates of the other end
    int detType;      // detector type 1: scifi, 2: us, 3: ds
    float hitTime;    // time of the hit
    int detId;         // detector Id

    Hit(bool orientation = true, 
        float x1 = 0.0, float y1 = 0.0, float z1 = 0.0,
        float x2 = 0.0, float y2 = 0.0, float z2 = 0.0,
        int detType = 0, 
        float hitTime = 0.0, 
        int detId = 0) 
    : orientation(orientation), 
      x1(x1), y1(y1), z1(z1), 
      x2(x2), y2(y2), z2(z2), 
      detType(detType), 
      hitTime(hitTime),
      detId(detId){}

    virtual ~Hit() {}

    void clear() {
        orientation = true;
        x1 = y1 = z1 = 0.0;
        x2 = y2 = z2 = 0.0;
        detType = 0;
        hitTime = 0.0;
        detId = 0;
    }

    ClassDef(Hit, 1)
};

class ScifiCluster : public TObject {
public:
    bool orientation; // true for vertical (1), false for horizontal (0)
    float x1, y1, z1; // coordinates of one end
    float x2, y2, z2; // coordinates of the other end
    int detType;      // detector type

    ScifiCluster(bool orientation = true,
                 float x1 = 0.0, float y1 = 0.0, float z1 = 0.0,
                 float x2 = 0.0, float y2 = 0.0, float z2 = 0.0,
                 int detType = 0)
    : orientation(orientation),
      x1(x1), y1(y1), z1(z1),
      x2(x2), y2(y2), z2(z2),
      detType(detType) {
    }

    virtual ~ScifiCluster() {}

    void clear() {
        orientation = true;
        x1 = y1 = z1 = x2 = y2 = z2 = 0.0;
        detType = 0;
    }

    ClassDef(ScifiCluster, 1)
};

class RecoMuon : public TObject {
public:
    float px, py, pz; // momentum components
    float x, y, z;    // position coordinates

    RecoMuon(float px = 0.0, float py = 0.0, float pz = 0.0,
             float x = 0.0, float y = 0.0, float z = 0.0)
    : px(px), py(py), pz(pz),
      x(x), y(y), z(z) {
    }

    virtual ~RecoMuon() {}

    void clear() {
        px = py = pz = 0.0;
        x = y = z = 0.0;
    }

    ClassDef(RecoMuon, 1)
};

class VM_Selection : public TObject {
public:
    bool stage1; // true for signal (1), false for background (0)
    bool stage2; // true for signal (1), false for background (0)

    VM_Selection(bool stage1 = false, bool stage2 = false)
    : stage1(stage1), stage2(stage2) {
    }

    virtual ~VM_Selection() {}

    void clear() {
        stage1 = false;
        stage2 = false;
    }

    ClassDef(VM_Selection, 1)
};
