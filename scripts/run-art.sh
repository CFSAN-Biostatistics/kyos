#!/bin/bash

# Run art_illimuna to create fastq files for all the replicates

rm SyntheticData/train/art-cmd.log
rm SyntheticData/validate/art-cmd.log
rm SyntheticData/test/art-cmd.log

# Uncomment to run on this computer
#python run-art.py SyntheticData/train/replicates    SyntheticData/train/art_fastq       1 SyntheticData/train/art-cmd.log
#python run-art.py SyntheticData/validate/replicates SyntheticData/validate/art_fastq 1001 SyntheticData/validate/art-cmd.log
#python run-art.py SyntheticData/test/replicates     SyntheticData/test/art_fastq     2001 SyntheticData/test/art-cmd.log

# Run on grid engine
echo 'python run-art.py SyntheticData/train/replicates    SyntheticData/train/art_fastq       1 SyntheticData/train/art-cmd.log'    | qsub -N art1 -cwd -j y -V
echo 'python run-art.py SyntheticData/validate/replicates SyntheticData/validate/art_fastq 1001 SyntheticData/validate/art-cmd.log' | qsub -N art2 -cwd -j y -V
echo 'python run-art.py SyntheticData/test/replicates     SyntheticData/test/art_fastq     2001 SyntheticData/test/art-cmd.log'     | qsub -N art3 -cwd -j y -V
