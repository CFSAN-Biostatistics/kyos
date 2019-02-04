Synthetic Data Creation Scripts
===============================

This directory contains the scripts we used to create synthetic data when developing Kyos.


Dependencies
------------

These scripts need the following external software::

  snp-mutator
  art_illumina
  qarrayrun
  CFSAN SNP Pipeline

The scripts need the following reference genome::

  CFSAN000189


Environment
-----------

Most of these scripts will run on a Linux workstation, but the ``qsub-create_tabular.sh``
script runs in a Grid Engine environment.


Step-by-Step Procedure
----------------------

To create the synthetic data, run the scripts in this order::

  run-snp-mutator.sh
  run-art.sh
  run-snp-pipeline.sh
  qsub-create_tabular.sh
  run-merge-tabular.sh


Output Files
------------

After completing these steps, you should have the following 3 files::

  tabular-features-train
  tabular-features-validate
  tabular-features-test
