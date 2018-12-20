import argh

from data.gtex import GTEx
from data.expecto import Expecto

# from cli.train import train_keras
# from data import create_lmdb

def wc_l(fname):
    """Get the number of lines of a text-file using unix `wc -l`
    """
    import os
    import subprocess
    return int((subprocess.Popen('wc -l {0}'.format(fname), shell=True, stdout=subprocess.PIPE).stdout).readlines()[0].split()[0])


def main():
    # assembling:
    # parser = argh.ArghParser()
    # parser.add_commands([wc_l, train_keras, create_lmdb])
    # parser.add_commands([wc_l])
    # argh.dispatch(parser)

    # gtex = GTEx()
    #
    # print("1. Load GTEx data:")
    # gtex.load_count_matrix("../data.csv", sep=",", varanno="../anno.csv", obsanno="../anno_obs.csv")
    # gtex.get_count_matrix()
    #
    #
    # print("2. Filter data:")

if __name__ == "__main__":
    main()


