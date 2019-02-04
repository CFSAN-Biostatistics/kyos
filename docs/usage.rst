========
Usage
========

.. highlight:: bash

Kyos is dependant upon snp-mutator.

To extract tabular data from a BAM file::

    kyos tabulate -t TRUTH_FILE --force_truth input_file output_file ref_file

To merge multiple tabulated files::

    kyos merge INFILE ...

To train Kyos::

    kyos train FTR FTR MODEL

To test Kyos::

    kyos test --vcf VCF MODEL FTR