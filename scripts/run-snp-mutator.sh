#!/bin/bash

# Create mutated fasta replicates

mkdir -p SyntheticData/train/replicates
mkdir -p SyntheticData/validate/replicates
mkdir -p SyntheticData/test/replicates

POOL_SIZE=100000
NUM_REPLICATES=1000
REPLICATES_PER_GROUP=100
NUM_SNPS=500
NUM_DELETIONS=0
NUM_INSERTIONS=0

POOL_SIZE=100000
NUM_REPLICATES=200
REPLICATES_PER_GROUP=40
NUM_SNPS=500
NUM_DELETIONS=0
NUM_INSERTIONS=0

RSEED=1
cd SyntheticData/train/replicates
snpmutator -r $RSEED --mono -v truth.vcf -o summary.tsv --metrics metrics --ref unmutated.fasta --seqid chrom -p $POOL_SIZE -g $REPLICATES_PER_GROUP -n $NUM_REPLICATES -s $NUM_SNPS -d $NUM_DELETIONS -i $NUM_INSERTIONS ../../../reference/CFSAN000189.fasta
cd -

RSEED=2
cd SyntheticData/validate/replicates
snpmutator -r $RSEED --mono -v truth.vcf -o summary.tsv --metrics metrics --ref unmutated.fasta --seqid chrom -p $POOL_SIZE -g $REPLICATES_PER_GROUP -n $NUM_REPLICATES -s $NUM_SNPS -d $NUM_DELETIONS -i $NUM_INSERTIONS ../../../reference/CFSAN000189.fasta
cd -

RSEED=3
cd SyntheticData/test/replicates
snpmutator -r $RSEED --mono -v truth.vcf -o summary.tsv --metrics metrics --ref unmutated.fasta --seqid chrom -p $POOL_SIZE -g $REPLICATES_PER_GROUP -n $NUM_REPLICATES -s $NUM_SNPS -d $NUM_DELETIONS -i $NUM_INSERTIONS ../../../reference/CFSAN000189.fasta
cd -
