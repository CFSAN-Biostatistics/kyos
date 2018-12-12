#!/bin/bash


for dir in train validate test
do
    DIR=SyntheticData/$dir
    cfsan_snp_pipeline run -c snppipeline.mapq10.conf -Q grid -m soft -o $DIR/snp-pipeline-mapq10 -s $DIR/art_fastq $DIR/replicates/unmutated.fasta
done
