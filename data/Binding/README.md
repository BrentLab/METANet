# TF Binding data

* ChIP-seq data for human transcription factors (TFs) were downloaded from the REMAP 2020 database [1].
* `DatabaseExtract_v_1.01.txt` lists TFs  defined by Lambert et al [2].
* `REMAP2020_TFs_list.txt` lists 799 Lambert TFs with high-confidence binding peaks.
* `Remap_binding_scores.bed` is the ground truth file used for training
  METANets. This file was generated using scripts in `scripts/remap_labels`.
  This file is too large for GitHub: it has been deposited in Zenodo.


REFERENCES
==========

[1] Cheneby J, Menetrier Z, Mestdagh M, Rosnet T, Douida A, Rhalloussi W, Bergon
A, Lopez F, Ballester B: ReMap 2020: a database of regulatory regions from an
integrative analysis of Human and Arabidopsis DNA-binding sequencing
experiments. Nucleic Acids Res 2020, 48(D1):D180-D188.
https://remap2020.univ-amu.fr/

[2] Lambert SA, Jolma A, Campitelli LF, Das PK, Yin Y, Albu M, Chen X, Taipale
J, Hughes TR, Weirauch MT: The Human Transcription Factors. Cell 2018,
172(4):650-665. https://humantfs.ccbr.utoronto.ca/
