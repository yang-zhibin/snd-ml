#define EventClasses_h
#include <vector>
#include "TObject.h"

class Id : public TObject {
    public:
        int runId, eventId, pdgCode, isMC;
        
        Id(int runId = -999, int eventId = -999, int pdgCode = -999, int isMC = -999)
            : runId(runId), eventId(eventId), pdgCode(pdgCode), isMC(isMC) {}
    
        virtual ~Id() {}
    
        void clear() {
            runId = -999;
            eventId = -999;
            pdgCode = -999;
            isMC = -999;
        }
    
        ClassDef(Id, 1)
    };
    
class Hit : public TObject {
    public:
        bool orientation;   // true for vertical (1), false for horizontal (0)
        float x1, y1, z1;   // coordinates of one end
        float x2, y2, z2;   // coordinates of the other end
        int detType;        // detector type # 1: scifi, 2: veto, 3: us, 4: ds
        float hitTime;      // time of the hit
        int detId;          // detector Id
        float qdc;          // hit qdc 
    
        Hit(bool orientation = true, 
            float x1 = -999.0, float y1 = -999.0, float z1 = -999.0,
            float x2 = -999.0, float y2 = -999.0, float z2 = -999.0,
            int detType = -999,
            float hitTime = -999.0,
            int detId = -999,
            float qdc = -999.0)
        : orientation(orientation),
            x1(x1), y1(y1), z1(z1),
            x2(x2), y2(y2), z2(z2),
            detType(detType),
            hitTime(hitTime),
            detId(detId), qdc(qdc) {}
    
        virtual ~Hit() {}
    
        void clear() {
            orientation = true;
            x1 = y1 = z1 = -999.0;
            x2 = y2 = z2 = -999.0;
            detType = -999;
            hitTime = -999.0;
            detId = -999;
            qdc = -999.0;
        }
    
        ClassDef(Hit, 1)
    };

struct ScifiMiniPoint {
    int    pdg{-999};
    float  energy_loss{-999.f};
    float  x{-999.f}, y{-999.f}, z{-999.f}; // optional, useful for debugging/analysis
    };
    
class VetoHit : public TObject {
    public:
    float hit_time{ -999.f };
    float energy_loss{ -999.f };
    int   veto_plane{ -1 };
    float qdc{ -999.f };
    std::vector<ScifiMiniPoint> scifiPoints;
    
    VetoHit() = default;
    virtual ~VetoHit() {}
    void clear() {
        hit_time = -999.f; energy_loss = -999.f; veto_plane = -1; qdc = -999.f;
        scifiPoints.clear();
    }
    ClassDef(VetoHit, 2) // bump version
    };