#!/usr/bin/env python

"""
Run ART-Illumina on all the fasta files in an input directory.
Generate fastq files in separate output directories per replicate
structured as needed by CFSAN SNP Pipeline.
"""

from __future__ import print_function

import numpy as np
import os
import subprocess
import sys


def draw_random_normal(mu, std, minimum, maximum):
    """Draw samples from a modified random normal distribution with the tails
    chopped off at a specified minimum and maximum values.
    """
    while True:
        draw = np.random.normal(mu, std)
        if draw >= minimum and draw <= maximum:
            return draw


def run_art_bash(in_directory, out_directory, filename, seed, coverage, log_file_path=None):
    inputFile = os.path.join(in_directory, filename + ".fasta")
    outputDir = os.path.join(out_directory, filename)
    subprocess.call("mkdir -p " + outputDir, shell=True)
    outputFile = os.path.join(outputDir, filename + "_")
    run_string = "art_illumina --noALN -i %s -p -l 250 -ss MSv1 -f %.1f -m 500 -s 85 -rs %d -o %s" % (inputFile, coverage, seed, outputFile)
    if log_file_path:
        with open(log_file_path, 'a') as f:
            print(run_string, file=f)
    subprocess.call(run_string, shell=True)


def main():
    arguments = sys.argv
    in_directory = arguments[1]
    out_directory = arguments[2]
    first_rseed = int(arguments[3])
    log_file_path = arguments[4]
    mu = 40
    std = 20
    minimum = 15
    maximum = 150
    rseed = first_rseed
    np.random.seed(rseed)
    for root, dirs, files in os.walk(in_directory):
        files = [f for f in files]
        for filename in sorted(files):
            if not filename.endswith(".fasta"):
                continue
            if filename.startswith("unmutated"):
                continue
            coverage = draw_random_normal(mu, std, minimum, maximum)
            run_art_bash(in_directory, out_directory, filename.replace(".fasta", ""), rseed, coverage, log_file_path)
            rseed += 1


if __name__ == "__main__":
    main()
