#define EventClasses_h
#include <vector>
#include "TObject.h"

class Id : public TObject {
public:
    int runId, eventId, pdgCode, isMC;
    Id(int runId=-1, int eventId=-1, int pdgCode=-1, int isMC=-1) : 
    runId(runId), eventId(eventId), pdgCode(pdgCode), isMC(isMC) {}
    virtual ~Id() {}

    void clear() {
        runId = -1;
        eventId = -1;
        pdgCode = -1;
        isMC = -1;
    }

    ClassDef(Id, 1)  
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