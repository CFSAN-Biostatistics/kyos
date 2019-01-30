#!/bin/bash

# Run the create_tabular.py script on all the BAM files.

# Step 1: create files with a 3-column list of parameters to feed to create_tabular.py, one line per bam file
for dir in train validate test
do
    rm create_tabular_array_params_$dir &> /dev/null

    reference=SyntheticData/$dir/replicates/unmutated.fasta
    outDir=SyntheticData/$dir/tabular
    bamGlob=SyntheticData/$dir/snp-pipeline-mapq10/samples/*/reads.sorted.deduped.indelrealigned.bam
    for inFile in $bamGlob
    do
        dirName=$(dirname $inFile)
        sampleName=$(basename $dirName)
        outFile=$outDir/$sampleName.tsv
        echo $inFile $outFile $reference >> create_tabular_array_params_$dir
    done
done

# Step 2: Submit the array job passing the 3 parameters in the create_tabular_array_params file
echo 'qarrayrun SGE_TASK_ID create_tabular_array_params_train kyos tabulate --rseed 1 --truth SyntheticData/train/replicates/summary.tsv --tnfract 1.0 {1} {2} {3}' | \
      qsub -N tab1 -t 1-$(cat create_tabular_array_params_train | wc -l) -cwd -j y -V

echo 'qarrayrun SGE_TASK_ID create_tabular_array_params_validate kyos tabulate --truth SyntheticData/validate/replicates/summary.tsv {1} {2} {3}' | \
      qsub -N tab2 -t 1-$(cat create_tabular_array_params_validate | wc -l) -cwd -j y -V

echo 'qarrayrun SGE_TASK_ID create_tabular_array_params_test kyos tabulate --truth SyntheticData/test/replicates/summary.tsv --force_truth {1} {2} {3}' | \
      qsub -N tab3 -t 1-$(cat create_tabular_array_params_test | wc -l) -cwd -j y -V


