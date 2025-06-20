# Pipeline for rosetta docking with density, crosslink and energy scoring

This pipeline combines docking, pose clustering, and ranking. Docking is performed with Rosetta in centroid mode and, after clustering and ranking, in local mode. Ranking is based on valid crosslinks using Xwalk for scoring, fit to a 3DEM map using Rosetta `elec_dens_fast` protocol and in case of local docking mode, the Rosetta interface energy score.

### Installation

Required third-party software:
- Rosetta ([link](https://rosettacommons.org/)) 
- Xwalk ([link](https://www.xwalk.org/))

Rosetta executables need to be in the PATH

In case of Xwalk add XWALK_BIN to the environment:
```
export XWALK_BIN=\path\to\xwalk
```

Required python libraries (Tested version in brackets, other versions might also work):
- pandas (1.5.3)
- scikit-learn (1.2.2)
- matplotlib (3.7.1)
- scipy (1.10.1)
- biopython (1.84)
- pyRMSD (4.3.2)



### Usage


To get an overview about all options type:
```
python3 docking_pipeline.py --help
```

Example input using crosslinking information and 3DEM map (with an effective resolution of 12 Angstrom):
```
--pdb_file model.pdb \
--map_file  map.mrc \
--resolution 12 \
--xlink_file xlinks.json \
--chains1 CDEFIJKLMN \
--chains2 OPQR \
--cluster_center=combined_score \
--centroid_score_labels=elec_dens_fast,valid_xlinks \
--centroid_score_weights=0.5,0.5 \
--local_score_labels=I_sc,elec_dens_fast,valid_xlinks \
--local_score_weights=0.33,0.33,0.33 \
--nproc 50 \
--dist_measure sas_dist \
--xl_threshold_distance 34
```

This will perform centroid docking of chains OPQR against chains CDEFIJKLMN (not moving) followed by superposition r.m.s.d. clustering using the normalized average score of `elec_dens_fast,valid_xlinks` (equal weights) to find cluster centers. The top 5 cluster centers are subjected to local full-atom docking followed by superposition r.m.s.d. clustering and ranking of cluster centers by the average score of `I_sc,elec_dens_fast,valid_xlinks` (equal weights). To determine valid crosslinks a maximum SAS (solvent accesible surface) distance of 34 Angstrom is used. The number of counted valid crosslinks defines the `valid_xlinks` score. For 3DEM map fit scoring the `elec_dens_fast` protocol of Rosetta is used which outputs the `elec_dens_fast` score.

### Crosslink definition file

The crosslink definition needs to be supplied in json format as follows:

```
[
    [residue_chain1, chain_label1, residue_chain2, chain_label2, group_id]
]
```

Example:
```
[
  [357, "O", 57, "E", 1],
  [357, "O", 75, "C", 2],
  [357, "R", 164, "N", 3]
]
```

In case of homomers, the same group_id specifies different redundant crosslink possibilities. In the below example, crosslinking possibilities between a homotetramer (in chains O, P, Q, R) and other subunits (in chains E, C) are defined: 
```
[
  [357, "O", 57, "E", 1],
  [357, "P", 57, "E", 1],
  [357, "Q", 57, "E", 1],
  [357, "R", 57, "E", 1],
  [357, "O", 75, "C", 2],
  [357, "P", 75, "C", 2],
  [357, "Q", 75, "C", 2],
  [357, "R", 75, "C", 2],
]
```