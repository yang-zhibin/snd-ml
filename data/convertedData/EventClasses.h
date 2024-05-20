//#ifndef EventClasses_h
#define EventClasses_h

#include "TObject.h"

class Id : public TObject {
public:
    Int_t runId, eventId;
    Id(Int_t runId=-1, Int_t eventId=-1) : runId(runId), eventId(eventId) {}
    virtual ~Id() {}

    ClassDef(Id, 1)  
};


class Label : public TObject {
public:
    int pdgCode;    // PDG code identifying the particle type
    float px, py, pz; // momentum components
    float x, y, z;    // coordinates of the first interaction point

    Label(int pdgCode = 0,
          float px = 0.0, float py = 0.0, float pz = 0.0,
          float x = 0.0, float y = 0.0, float z = 0.0)
    : pdgCode(pdgCode), px(px), py(py), pz(pz),
      x(x), y(y), z(z) {
    }

    virtual ~Label() {}

    ClassDef(Label, 1)
};


class Hit : public TObject {
public:
    bool orientation; // true for vertical (1), false for horizontal (0)
    float x1, y1, z1; // coordinates of one end
    float x2, y2, z2; // coordinates of the other end
    int detType;      // detector type 1: scifi, 2: us, 3: ds
    float hitTime;    // time of the hit

    Hit(bool orientation = true, 
        float x1 = 0.0, float y1 = 0.0, float z1 = 0.0,
        float x2 = 0.0, float y2 = 0.0, float z2 = 0.0,
        int detType = 0, 
        float hitTime = 0.0) 
    : orientation(orientation), 
      x1(x1), y1(y1), z1(z1), 
      x2(x2), y2(y2), z2(z2), 
      detType(detType), 
      hitTime(hitTime) {
    }

    virtual ~Hit() {}

    ClassDef(Hit, 1)
};

class ScifiCluster : public TObject {
public:
    bool orientation; // true for vertical (1), false for horizontal (0)
    float x1, y1, z1; // coordinates of one end
    float x2, y2, z2; // coordinates of the other end
    int detType;      // detector type

    // Constructor with default parameter values
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

    ClassDef(ScifiCluster, 1)
};

class RecoMuon : public TObject {
public:
    float px, py, pz; // momentum components
    float x, y, z;    // position coordinates

    // Constructor with default parameter values
    RecoMuon(float px = 0.0, float py = 0.0, float pz = 0.0,
             float x = 0.0, float y = 0.0, float z = 0.0)
    : px(px), py(py), pz(pz),
      x(x), y(y), z(z) {
    }

    virtual ~RecoMuon() {}

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

    ClassDef(VM_Selection, 1)
};



//#endif