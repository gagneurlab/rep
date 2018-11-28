import argh
# from rep.cli.train import train_keras
# from rep.data import create_lmdb

def wc_l(fname):
    """Get the number of lines of a text-file using unix `wc -l`
    """
    import os
    import subprocess
    return int((subprocess.Popen('wc -l {0}'.format(fname), shell=True, stdout=subprocess.PIPE).stdout).readlines()[0].split()[0])


def main():
    # assembling:
    parser = argh.ArghParser()
    # parser.add_commands([wc_l, train_keras, create_lmdb])
    parser.add_commands([wc_l])
    argh.dispatch(parser)
