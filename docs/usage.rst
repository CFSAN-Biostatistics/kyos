========
Usage
========

.. highlight:: bash

Before you can use Kyos to call variants, you will need to prepare the input datasets.
The workflow begins with one or more BAM files.  If you don't already have BAM files,
you could use the `CFSAN SNP Pipeline <http://snp-pipeline.readthedocs.io/en/latest/readme.html>`_
to create the BAM files.

When extracting features from BAM files, you will need to supply the known-truth if you intend to
use the tabulated features for training and testing the neural network.  The ``-t`` command line
option to the ``tabulate`` command adds an extra ``Truth`` column to the output tsv file.

Kyos is currently dependent upon `SNP Mutator <http://snp-mutator.readthedocs.io/en/latest/readme.html>`_
to generate the known-truth datasets for supervised learning. A future version will use VCF files
instead.

To extract tabular data from a BAM file::

    kyos tabulate -t TRUTH_FILE input.bam output.tsv ref.fasta

To merge multiple tabulated files::

    kyos merge file1.tsv file2.tsv file3.tsv ... > train.tsv

To train a neural network model::

    kyos train train.tsv validate.tsv model.h5

To test a neural network model::

    kyos test model.h5 test.tsv