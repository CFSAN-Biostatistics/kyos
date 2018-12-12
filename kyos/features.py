# -*- coding: utf-8 -*-

"""
This module extracts features useful for detecting variants from a BAM file.
"""

from __future__ import print_function
from __future__ import absolute_import

from Bio import SeqIO
from collections import Counter
from collections import namedtuple
import csv
import errno
import os
import pysam
import random
import sys

# Tuple to hold metrics per base
PerBaseTuple = namedtuple("PerBaseTuple", ['A', 'T', 'C', 'G', 'N', 'DEL', 'INS'])


feature_names = ["SampleName", "Chrom", "Position", "A", "T", "C", "G", "N", "a", "t", "c", "g", "n", "Insertion", "Deletion", "MapqA", "MapqT", "MapqC", "MapqG", "MapqN", "MapqDel", "MapqIns",
                 "BaseQualA", "BaseQualT", "BaseQualC", "BaseQualG", "BaseQualN", "BaseQualDel", "BaseQualIns", "RefBase"]

first_ftr_idx = feature_names.index("A")
last_ftr_idx = feature_names.index("BaseQualIns")


def mkdir_p(path):
    """Python equivalent of bash mkdir -p.

    Parameters
    ----------
    path : str
        Directory path to create.

    Raises
    ------
    OSError if the directory does not already exist and cannot be created
    """
    if path == "":
        return
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def read_summary_file(summary_file):
    """Read a SNP Mutator summary file to get the truth alleles.

    Parameters
    ----------
    summary_file : str
        Path to SNP Mutator summary file

    Returns
    ------
    dict
        key = str(sample_name+position)
        value = truth allele
    """
    truth = {}

    with open(summary_file, "r") as f:
        for line in f:
            z = line.split()
            key = z[0] + " " + z[1]
            truth_var = z[3]
            if truth_var.endswith("_insertion"): #Strip off reference from insertions
                truth_var = truth_var[1:]
            truth[key] = [z[2].upper(), truth_var]

    return truth


def check_mutated(pileup_base_counter, ref_seq_dict, chrom, position):
    """Determines whether there is a mutation occuring at the position determined by the pileup column

    Parameters
    ----------
    pileup_base_counter : dict
        Contains the bases as the keys and the number of occurences as the values
    ref_seq_dict : dict of str
        Dictionary of contigs.  Key=contig name, value=sequence string.
    chrom : str
        Name of the contig
    position : int
        Zero-based position relative to the start of the contig

    Returns
    -------
    bool
        True if there is a mutation, false if there is not a mutation
    """
    if len(pileup_base_counter) == 0:
        return False

    ref_base = ref_seq_dict[chrom][position]

    keys = pileup_base_counter.keys()
    if len(keys) > 2:
        return True

    elif "+" in keys or "-" in keys or "*" in keys or "N" in keys or "n" in keys:
        return True
    elif len(keys) == 2:
        if keys[0].upper() != keys[1].upper():
            return True
        else:
            if keys[0].upper() != ref_base:
                return True
    else:
        if keys[0].upper() != ref_base:
            return True
    return False


def convert_to_csv(observations, file_path):
    """Writes the tabular data to a csv file

    Parameters
    ----------
    observations : list of dictionaries
        Contains the data of each mutated place as a row in the list
    file_path : string
        Path to the output file

    """
    out_dir = os.path.dirname(file_path)
    mkdir_p(out_dir)
    with open(file_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=feature_names, delimiter='\t')

        writer.writeheader()
        for row in observations:
            writer.writerow(row)


def mean(qualities):
    """Calculates the average of a list which might be empty.

    Parameters
    ----------
    qualtiies : list
        list of numbers

    Returns
    -------
    float
        Float representing the mean numbers in the the list; zero if the list is empty.
    """
    if len(qualities) == 0:
        return 0

    de_phred = []

    for x in qualities:
        de_phred.append(100-(100*(10**(-1.0*x/10.0))))

    return sum(de_phred) / float(len(de_phred))


def get_average_qualities(pileup_alleles, pileup_qualities):
    """Gets the average quality for each A,T,C,G,N, and deletions.

    Parameters
    ----------
    pileup_alleles : pileup column
        Pileup column containing a list of bases.
    pileup_qualities : pileup column
        Pileup column containing a list of quality scores for each allele.  The quality score
        could be either base quality or mapping quality.

    Returns
    -------
    PerBaseTuple
        Named tuple, PerBaseTuple, containing the average quality for each of (A,T,C,G,N,deletion)
    """
    qualA = []
    qualT = []
    qualC = []
    qualG = []
    qualN = []
    qualDel = []
    qualIns = []

    for x in range(len(pileup_alleles)):
        if pileup_alleles[x].upper() == "A":
            qualA.append(pileup_qualities[x])
        elif pileup_alleles[x].upper() == "T":
            qualT.append(pileup_qualities[x])
        elif pileup_alleles[x].upper() == "C":
            qualC.append(pileup_qualities[x])
        elif pileup_alleles[x].upper() == "G":
            qualG.append(pileup_qualities[x])
        elif pileup_alleles[x].upper() == "N":
            qualN.append(pileup_qualities[x])
        elif "+" in pileup_alleles[x]:
            qualIns.append(pileup_qualities[x])
        elif "-" in pileup_alleles[x] or "*" in pileup_alleles[x]:
            qualDel.append(pileup_qualities[x])
        else:
            print("Unknown pileup_allele", pileup_alleles[x], file=sys.stderr)

    return PerBaseTuple(mean(qualA), mean(qualT), mean(qualC), mean(qualG), mean(qualN), mean(qualDel), mean(qualIns))


def read_ref_file(reference_path):
    """Read a fasta reference and return a dictionary of contigs.

    Parameters
    ----------
    reference_path : str
        Path to reference fasta file.

    Returns
    -------
    ref_seq_dict : dict
        Dictionary with key=contig name and value=string of bases in the contig.
    """
    ref_seq_dict = {}
    for seqrecord in SeqIO.parse(reference_path, "fasta"):
        ref_seq_dict[seqrecord.id] = str(seqrecord.seq).upper()
    return ref_seq_dict


def create_tabular_data(input_file, output_file, reference_file, truth_file=None, tn_fraction=1.0, force_truth=None, rseed=None):
    # TODO: This function needs to also examine the summary file or the VCF file.
    #       This is the truth dataset.
    #       Every mutated position needs to be included in the output file, even if
    #       there is no evidence of variation at the position.
    # TODO: Determine if the pileup iterator skips some positions.  Could it skip
    #       a position in the truth dataset?
    """Read a BAM file, extract informative features, and write the features to
    a TSV file.

    Parameters
    ----------
    input_file : str
        Path to input BAM file
    output_file : str
        Path to output TSV file
    reference_file : str
        Path to reference fasta file
    truth_file : str
        Path to the file containing the truth allele
    tn_fraction : float
        Fraction of True Negatives to write to the output file.
    force_truth : bool
        Extract all known variant observations even if there is no coverage. Requires a summary file.
    rseed : int
        Random seed to ensure reproducible results.  Set to zero for non-deterministic results.
    """
    if rseed:
        random.seed(rseed)

    input_dir = os.path.dirname(input_file)
    sample_name = os.path.basename(input_dir)

    if truth_file:
        truth_dict = read_summary_file(truth_file)
        feature_names.append("Truth")

    ref_seq_dict = read_ref_file(reference_file)

    samfile = pysam.AlignmentFile(input_file, "rb")

    observations = []

    all_positions = set()

    for pileupcolumn in samfile.pileup(until_eof=True):

        min_base_quality = 13
        pileupcolumn.set_min_base_quality(min_base_quality)

        pileup_bases = pileupcolumn.get_query_sequences(mark_matches=True, add_indels=True)

        observation = {}  # dictionary of features keyed by feature_names

        for x in feature_names:
            if x not in observation:
                observation[x] = 0

        contig = pileupcolumn.reference_name

        observation["Chrom"] = contig
        observation["RefBase"] = ref_seq_dict[contig][pileupcolumn.pos]
        observation["SampleName"] = sample_name
        observation["Position"] = pileupcolumn.pos + 1
        observation["Truth"] = observation["RefBase"]

        if truth_file:
            key = sample_name + " " + str(pileupcolumn.pos + 1)
            mutated_position_flag = key in truth_dict
            if mutated_position_flag:
                observation["Truth"] = truth_dict[key][1].upper()
            else:
                observation["Truth"] = observation["RefBase"]

        if len(pileup_bases) == 0:
            # no coverage at this position
            print(pileupcolumn.pos, "No coverage at base quality", min_base_quality, file=sys.stderr)
            if force_truth and mutated_position_flag:
                print("wow")
                observations.append(observation)
                all_positions.add(sample_name + " " + str(pileupcolumn.pos + 1))
            continue

        base_counts = Counter(pileup_bases)

        is_mutated = check_mutated(base_counts, ref_seq_dict, contig, pileupcolumn.pos)

        in_summary = force_truth and mutated_position_flag

        '''if in_summary:
            print(in_summary, pileupcolumn.pos, key)'''

        if is_mutated or in_summary:
            if tn_fraction < 1.0 and not mutated_position_flag and random.random() > tn_fraction:
                continue

            for key in base_counts:
                if key in feature_names:
                    observation[key] = base_counts[key]
                else:
                    # A regex pattern `\+[0-9]+[ACGTNacgtn]+' indicates there is an insertion between this reference position
                    # and the next reference position. The length of the insertion is given by the integer in the pattern,
                    # followed by the inserted sequence.
                    if "+" in key:
                        observation["Insertion"] += 1

                    # A regex pattern `-[0-9]+[ACGTNacgtn]+' represents a deletion from the reference.
                    # An * (asterisk) is a placeholder for a deleted base in a multiple basepair deletion that was mentioned
                    # in a previous line by the -[0-9]+[ACGTNacgtn]+ notation.
                    elif "-" in key or "*" in key:
                        observation["Deletion"] += 1

                    else:
                        print(1 + pileupcolumn.pos, "Unexpected base:", key, pileup_bases, file=sys.stderr)

            mapping_qualities = pileupcolumn.get_mapping_qualities()
            base_qualities = pileupcolumn.get_query_qualities()

            ave_mapping_quality = get_average_qualities(pileup_bases, mapping_qualities)
            ave_base_quality = get_average_qualities(pileup_bases, base_qualities)

            observation["MapqA"] = ave_mapping_quality.A
            observation["MapqT"] = ave_mapping_quality.T
            observation["MapqC"] = ave_mapping_quality.C
            observation["MapqG"] = ave_mapping_quality.G
            observation["MapqN"] = ave_mapping_quality.N
            observation["MapqDel"] = ave_mapping_quality.DEL
            observation["MapqIns"] = ave_mapping_quality.INS

            observation["BaseQualA"] = ave_base_quality.A
            observation["BaseQualT"] = ave_base_quality.T
            observation["BaseQualC"] = ave_base_quality.C
            observation["BaseQualG"] = ave_base_quality.G
            observation["BaseQualN"] = ave_base_quality.N
            observation["BaseQualDel"] = ave_base_quality.DEL
            observation["BaseQualIns"] = ave_base_quality.INS

            # Fill in missing data
            for x in feature_names:
                if x not in observation:
                    observation[x] = 0

            observations.append(observation)
            all_positions.add(sample_name + " " + str(pileupcolumn.pos + 1))

    for key, value in truth_dict.iteritems():
        try:
            splitted = key.split()
            position = int(splitted[1])
            if sample_name == splitted[0] and key not in all_positions:
                observation = {}  # dictionary of features keyed by feature_names

                for x in feature_names:
                    if x not in observation:
                        observation[x] = 0

                contig = ""

                observation["Chrom"] = "chrom"
                observation["RefBase"] = value[0]
                observation["SampleName"] = splitted[0]
                observation["Position"] = position
                observation["Truth"] = value[1]

                observations.append(observation)
        except Exception:
            print(key, value)

    samfile.close()

    convert_to_csv(observations, output_file)


def merge(input_paths):
    """Merge multiple tabulated files and write to stdout.

    Parameters
    ----------
    input_paths : list of str
        List of tsv file paths
    """
    file_num = 0
    for path in input_paths:
        file_num += 1
        with open(path) as f:
            if file_num > 1:
                next(f)  # skip header line in all files after first
            for line in f:
                sys.stdout.write(line)

    sys.stdout.flush()
