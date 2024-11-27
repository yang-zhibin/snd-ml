Convert SND software dependent raw data into regular root data

converted data format (https://github.com/SND-LHC/sndsw/blob/024bea15aa2c1dde4c67530f27354a22c652db8e/python/SndlhcDigi.py#L22)
- id:
    - EventHeader.runId
    - EventHeader.eventId

- Lable (truth info):(https://github.com/SND-LHC/sndsw/blob/024bea15aa2c1dde4c67530f27354a22c652db8e/python/shipEvent_ex.py#L14)
    - pdg code
    - pz
    - first interaction point
- Hits (training features): (https://github.com/SND-LHC/sndsw/blob/024bea15aa2c1dde4c67530f27354a22c652db8e/python/SndlhcDigi.py#L25)
    - 0 is vertical (1) or horizontal (0)
    - x1, y1, z1, x2, y2, z2
    - det type
    - hitTime
- scifiCluster (https://github.com/SND-LHC/sndsw/blob/024bea15aa2c1dde4c67530f27354a22c652db8e/python/SndlhcDigi.py#L42)

- recoMuon

- hits2MCPoints (https://github.com/SND-LHC/sndsw/blob/024bea15aa2c1dde4c67530f27354a22c652db8e/python/SndlhcDigi.py#L29)



converted data format 
- id:
    - EventHeader.runId
    - EventHeader.eventId

- Lable (truth info):
    - pdg code
    - px,py,pz
    - x,y,z(first interaction point)
- Hits (training features): 
    - 0 is vertical (1) or horizontal (0)
    - x1, y1, z1, 
    - x2, y2, z2
    - det type
    - hitTime
- scifiCluster 
    - 0 is vertical (1) or horizontal (0)
    - x1, y1, z1, 
    - x2, y2, z2,
    - det type

- recoMuon
    - px,py,pz
    - x,y,z

- vm_selection
    - stage1 (signal=1, background=0)
    - stage2 (signal=1, background=0)

- hits2MCPoints
    - x,y,z