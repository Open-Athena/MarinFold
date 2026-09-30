| Date       | UniRef30 | PDB          | Environmental DB   |
|------------|----------|--------------|--------------------|
| Release    | 2103     | PDB70 200916 | BFD/MGnify         |
| 2022-03-04 | 2103     | PDB70 200916 | ColabFoldDB 202108 |
| 2022-07-13 | 2202     | PDB70 220313 | ColabFoldDB 202108 |
| 2023-06-12 | 2302     | PDB100 230517| ColabFoldDB 202108 |
| 2023-07-27 | 2202     | PDB70 220313 | ColabFoldDB 202108 |
| 2023-07-31 | 2302     | PDB100 230517| ColabFoldDB 202108 |
| 2025-08-04 | 2302*    | PDB100 230517| ColabFoldDB 202108 |


## 2025-08-04: Updated UniRef100_2302 taxonomy/pairing files
We updated the taxonomy/pairing files for the UniRef100. Previously, the pairing was done on the UniRef LCA. We switched this to the taxon identifier of the representative sequence. We also updated the `_taxonomy` file, which is **not** directly used in the ColabFold search. About ~8000 taxa were missing in this, resulting in some hits being assigned unclassified labels, instead of a more specific label.

We updated the `setup_databases.sh` script to first download the base files with the old pairing/taxonomy file and then download a second archive that only replaces these files. If you don't want to re-download all databases you can download only the two changed files [in this archive](https://opendata.mmseqs.org/colabfold/uniref30_2302_newtaxonomy.tar.gz).