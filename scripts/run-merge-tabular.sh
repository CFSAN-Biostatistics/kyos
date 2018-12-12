#!/bin/bash

# Merge the tabulated feature files together into a long narrow file.

for dir in train validate test
do
    echo merging $dir tabulated feature files
    kyos merge SyntheticData/$dir/tabular/*.tsv > SyntheticData/$dir/tabular-features-$dir
done


for dir in train validate test
do
    lines=$(wc -l SyntheticData/$dir/tabular-features-$dir | cut -d ' ' -f 1)
    echo $lines Observations in SyntheticData/$dir/tabular-features-$dir
done
