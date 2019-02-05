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
import logging
import os
import pysam
import random
import sys

# Tuple to hold metrics per base
PerBaseTuple = namedtuple("PerBaseTuple", ['A', 'T', 'C', 'G', 'N', 'DEL', 'INS'])

# Tuple to hold truth dict value
TruthTuple = namedtuple("TruthTuple", ['ref_base', 'variant'])


feature_names = ["SampleName", "Chrom", "Position", "A", "T", "C", "G", "N", "a", "t", "c", "g", "n", "Insertion", "Deletion", "MapqA", "MapqT", "MapqC", "MapqG", "MapqN", "MapqDel", "MapqIns",
                 "BaseQualA", "BaseQualT", "BaseQualC", "BaseQualG", "BaseQualN", "BaseQualDel", "BaseQualIns", "RefBase"]

# Expose feature names to the nn module
first_ftr_name = "A"
last_ftr_name = "BaseQualIns"
read_count_feature_names = ["A", "T", "C", "G", "N", "a", "t", "c", "g", "n", "Insertion", "Deletion"]
map_quality_feature_names = ["MapqA", "MapqT", "MapqC", "MapqG", "MapqN", "MapqDel", "MapqIns"]
base_quality_feature_names = ["BaseQualA", "BaseQualT", "BaseQualC", "BaseQualG", "BaseQualN", "BaseQualDel", "BaseQualIns"]
target_label_name = "Truth"


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

    The SNP Mutator summary file looks like this:

    Replicate              Position  OriginalBase  NewBase
    CFSAN000189_mutated_1  25183     g             T
    CFSAN000189_mutated_1  34442     a             aA_insertion
    CFSAN000189_mutated_1  42998     g             A
    CFSAN000189_mutated_1  92950     a             _deletion
    CFSAN000189_mutated_1  94829     g             C

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
        f.next()  # skip first row with headers
        for line in f:
            replicate, pos, ref_base, truth_variant = line.split()
            key = replicate + " " + pos
            if truth_variant.endswith("_insertion"):  # Strip off reference base from insertions
                truth_variant = truth_variant[1:]
            truth[key] = TruthTuple(ref_base.upper(), truth_variant)

    return truth


def any_variant_evidence(pileup_base_counter, ref_seq_dict, chrom, position):
    """Determine whether there is any evidence of variation occuring at the position determined by the pileup column.

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
        True if there is any evidence of variant, false if there is not evidence
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
    """Gets the average quality for each A,T,C,G,N, insertions and deletions.

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
        Named tuple, PerBaseTuple, containing the average quality for each of (A,T,C,G,N,insertion,deletion)
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
                # TODO: test this for insertions and deletions, not sure this will work properly
                observation["Truth"] = truth_dict[key].variant.upper()
            else:
                observation["Truth"] = observation["RefBase"]

        if len(pileup_bases) == 0:
            # no coverage at this position
            logging.debug("%d No coverage at base quality %d" % (pileupcolumn.pos + 1, min_base_quality))
            if force_truth and mutated_position_flag:
                observations.append(observation)
                all_positions.add(sample_name + " " + str(pileupcolumn.pos + 1))
                logging.debug("Added no-coverage known true mutation for %s at %d" % (sample_name, pileupcolumn.pos + 1))
            continue

        base_counts = Counter(pileup_bases)

        is_any_variant_evidence = any_variant_evidence(base_counts, ref_seq_dict, contig, pileupcolumn.pos)

        in_summary = force_truth and mutated_position_flag

        '''if in_summary:
            print(in_summary, pileupcolumn.pos, key)'''

        if is_any_variant_evidence or in_summary:
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
            replicate, position_str = key.split()
            position = int(position_str)
            if sample_name == replicate and key not in all_positions:
                observation = {}  # dictionary of features keyed by feature_names

                for x in feature_names:
                    if x not in observation:
                        observation[x] = 0

                contig = ""

                observation["Chrom"] = "chrom"
                observation["RefBase"] = value.ref_base
                observation["SampleName"] = sample_name
                observation["Position"] = position
                observation["Truth"] = value.variant

                observations.append(observation)
                logging.debug("Added missing known true mutation for %s at %d" % (sample_name, position))
        except Exception:
            logging.exception("Exception with truth dict.  key: %s value: %s", repr(key), repr(value))

    samfile.close()

    logging.debug("Converting to csv...")
    convert_to_csv(observations, output_file)
    logging.debug("Tabulate finished.")


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
