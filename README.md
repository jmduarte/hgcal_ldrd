# hgcal_ldrd

Code repository for HGCal LDRD

Setup:
```
wget https://raw.githubusercontent.com/jmduarte/hgcal_ldrd/cleanup/setup.sh
source setup.sh
cd hgcal_ldrd
mkdir -p data/ntup
xrdcp root://eoscms.cern.ch//eos/cms/store/group/dpg_hgcal/comm_hgcal/sitong/D41/tau_flatPt/Pythia8PtGun_sitong_20190724/NTUP/partGun_PDGid15_x1000_Pt3.0To100.0_NTUP_1.root data/ntup/
```
