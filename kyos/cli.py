#!/usr/bin/env python

"""Console script for Kyos."""

from __future__ import print_function
from __future__ import absolute_import

import argparse
import logging
import sys

from kyos import features
from kyos import nn
from kyos.__init__ import __version__


def parse_arguments(system_args):
    """Parse command line arguments.

    Parameters
    ----------
    system_args : list
        List of command line arguments, usually sys.argv[1:].

    Returns
    -------
    Namespace
        Command line arguments are stored as attributes of a Namespace.
    """
    def non_negative_int(value):
        try:
            ivalue = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a number >= 0")
        if ivalue < 0:
            raise argparse.ArgumentTypeError("Must be >= 0")
        return ivalue

    def positive_int(value):
        try:
            ivalue = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a number greater than 0")
        if ivalue <= 0:
            raise argparse.ArgumentTypeError("Must be greater than 0")
        return ivalue

    description = """Tools for haploid variant calling with Deep Neural Networks."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--version", action="version", version="%(prog)s version " + __version__)
    subparsers = parser.add_subparsers(dest="subparser_name", help=None, metavar="subcommand")
    subparsers.required = True

    formatter_class = argparse.ArgumentDefaultsHelpFormatter

    '''help_str = "Extract and tabulate informative features from a BAM file."
    description = help_str + "  Pileup positions with evidence of variation are written to the output file."
    subparser = subparsers.add_parser("extract", formatter_class=formatter_class, description=description, help=help_str)
    subparser.add_argument(dest="bam_path", type=str, metavar="BAMFILE", help="Input sorted BAM file.")
    subparser.add_argument(dest="ref_path", type=str, metavar="REFFILE", help="Input reference fasta file.")
    subparser.add_argument("--truth", dest="truth_file_path", type=str, metavar="TRUTH", help="Optional input truth file.  When specified, the last column of the output TSV file will be the truth call.")
    subparser.add_argument("--tnfract", dest="tn_fract", type=float, default=1.0, metavar="FRACTION", help="Fraction of true negative observations to retain in the output.")
    subparser.add_argument(dest="output_path", type=str, metavar="OUTFILE", help="Output tabulated feature file.")
    subparser.set_defaults(func=extract_command)

    description = "Merge multiple tabulated files and write to stdout."
    subparser = subparsers.add_parser("merge", formatter_class=formatter_class, description=description, help=description)
    subparser.add_argument(dest="input_paths", type=str, metavar="INFILE", help="Input tabulated feature files.", nargs='+')
    subparser.set_defaults(func=merge_command)'''

    help_str = "Extract and tabulate informative features from a BAM file."
    description = """Generate tabular data containing places of mutations from a reference genome. Takes a BAM
           file. Outputs the positions, number of forward and reverse ACTG, deletions, insertions and
           reference skipped, and the qualities of the ACTG SNP's."""

    subparser = subparsers.add_parser("tabulate", formatter_class=formatter_class, description=description, help=help_str)

    subparser.add_argument(dest="input_file",       type=str,    help="Input sorted bam file.")
    subparser.add_argument(dest="output_file",      type=str,    help="Output tabular data file.")
    subparser.add_argument(dest="ref_file",         type=str,    help="Input reference file.")
    subparser.add_argument("-t", "--truth", dest="truth_file", type=str, help="SNP Mutator summary file containing truth allele.", default=None)
    subparser.add_argument("-f", "--tnfract", dest="tnfract", type=float, help="Fraction of True Negatives to write to the output file.", default=1.0)
    subparser.add_argument("-s", "--rseed", dest="rseed", type=int, help="Random seed to ensure reproducible results when using --tnfract.  Set to zero for non-deterministic results.", default=1)
    subparser.add_argument("--force_truth", dest="force_truth", action='store_true', help="Extract all known variant observations even if there is no coverage. Requires a summary file.", default=False)
    subparser.set_defaults(func=tabulate_command)

    description = "Merge multiple tabulated files and write to stdout."
    subparser = subparsers.add_parser("merge", formatter_class=formatter_class, description=description, help=description)
    subparser.add_argument(dest="input_paths", type=str, metavar="INFILE", help="Input tabulated feature files.", nargs='+')
    subparser.set_defaults(func=merge_command)

    description = "Train a neural network to detect variants."
    subparser = subparsers.add_parser("train", formatter_class=formatter_class, description=description, help=description)
    subparser.add_argument(dest="train_file_path", type=str, metavar="FTR", help="Input tabulated feature file for training.")
    subparser.add_argument(dest="validate_file_path", type=str, metavar="FTR", help="Input tabulated feature file for validation during training.")
    subparser.add_argument(dest="model_file_path", type=str, metavar="MODEL", help="Output trained model.")
    subparser.add_argument("--rseed", dest="rseed", type=int, help="Random seed to ensure reproducible results.  Set to non-zero value for reproducible results.", default=0)
    subparser.set_defaults(func=train_command)

    description = "Test a neural network model when the truth is known."
    subparser = subparsers.add_parser("test", formatter_class=formatter_class, description=description, help=description)
    subparser.add_argument(dest="model_file_path", type=str, metavar="MODEL", help="Input trained model.")
    subparser.add_argument(dest="test_file_path", type=str, metavar="FTR", help="Input tabulated feature file for testing.")
    # subparser.add_argument("--vcf", dest="vcf_file_path", type=str, metavar="VCF", help="Optional output VCF file.")
    subparser.set_defaults(func=test_command)

    description = "Call variants when the truth is unknown."
    subparser = subparsers.add_parser("call", formatter_class=formatter_class, description=description, help=description)
    subparser.add_argument(dest="model_file_path", type=str, metavar="MODEL", help="Input trained model.")
    subparser.add_argument(dest="ftr_file_path", type=str, metavar="FTR", help="Input tabulated feature file.")
    subparser.add_argument(dest="vcf_file_path", type=str, metavar="VCF", help="Output VCF file.")
    subparser.set_defaults(func=call_command)

    args = parser.parse_args(system_args)
    return args


def train_command(args):
    """Train a neural network to detect variants.

    Parameters
    ----------
    args : Namespace
        Command line arguments stored as attributes of a Namespace, usually
        parsed from sys.argv
    """
    nn.train(args.train_file_path, args.validate_file_path, args.model_file_path, args.rseed)


def test_command(args):
    """YYY multiple files and write to stdout.

    Parameters
    ----------
    args : Namespace
        Command line arguments stored as attributes of a Namespace, usually
        parsed from sys.argv
    """
    nn.test(args.model_file_path, args.test_file_path)  # , args.vcf_file_path)


def call_command(args):
    """YYY multiple files and write to stdout.

    Parameters
    ----------
    args : Namespace
        Command line arguments stored as attributes of a Namespace, usually
        parsed from sys.argv
    """
    nn.call(args.model_file_path, args.ftr_file_path, args.vcf_file_path)


def merge_command(args):
    """Merge multiple tabulated files and write to stdout.

    Parameters
    ----------
    args : Namespace
        Command line arguments stored as attributes of a Namespace, usually
        parsed from sys.argv
    """
    features.merge(args.input_paths)


def tabulate_command(args):
    """YYY multiple files and write to stdout.

    Parameters
    ----------
    args : Namespace
        Command line arguments stored as attributes of a Namespace, usually
        parsed from sys.argv
    """
    if args.tnfract < 1.0 and args.truth_file is None:
        print("You must specify a truth file when requesting a subset of the true negative observations.", file=sys.stderr)
        exit(1)
    features.create_tabular_data(args.input_file, args.output_file, args.ref_file, args.truth_file, args.tnfract, args.force_truth, args.rseed)


def run_command_from_args(args):
    """Run a subcommand with previously parsed arguments in an argparse namespace.

    This function is intended to be used for unit testing.

    Parameters
    ----------
    args : Namespace
        Command line arguments are stored as attributes of a Namespace.
        The args are obtained by calling parse_argument_list().

    Returns
    -------
    Returns 0 on success if it completes with no exceptions.
    """
    return args.func(args)  # this executes the function previously associated with the subparser with set_defaults


def run_from_line(line):
    """Run a command with a command line.

    This function is intended to be used for unit testing.

    Parameters
    ----------
    line : str
        Command line.

    Returns
    -------
    Returns 0 on success if it completes with no exceptions.
    """
    argv = line.split()
    args = parse_arguments(argv)
    return args.func(args)  # this executes the function previously associated with the subparser with set_defaults


def main():
    """This is the main function which is turned into an executable
    console script by the setuptools entry_points.  See setup.py.

    To run this function as a script, first install the package:
        $ python setup.py develop
        or
        $ pip install --user kyos

    Parameters
    ----------
    This function must not take any parameters

    Returns
    -------
    The return value is passed to sys.exit().
    """
    enable_log_timestamps = True
    if enable_log_timestamps:
        logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)
    else:
        logging.basicConfig(format="%(message)s", level=logging.DEBUG)
    args = parse_arguments(sys.argv[1:])
    return args.func(args)  # this executes the function previously associated with the subparser with set_defaults


# This snippet lets you run the cli without installing the entrypoint.
if __name__ == "__main__":
    sys.exit(main())
