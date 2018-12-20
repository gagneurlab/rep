"""
Abstract class for loading / saving the data in anndata format
"""
import os
import abc

import anndata
import pandas as pd


class BackedError(Exception):
    pass

# Abstract Class
class AbstractData(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self,anndataobj = None):

        self.annobj = anndataobj


    def load_vars(self, csvfile, sep=","):
        """
        Load var (columns) description for the summarized experiment
        :param annobj:
        :param csvfile:
        :param sep:
        """

        if csvfile:
            varaux = pd.DataFrame(pd.read_csv(os.path.abspath(csvfile), header=None, delimiter=sep, index_col=0))
            self.annobj.var = varaux
            self.annobj.var_names = list(varaux.index)


    def load_obs(self, csvfile, sep=","):
        """
        Load obs (rows) description for the summarized experiment
        :param annobj:
        :param csvfile:
        :param sep:
        """
        if csvfile:
            obsaux = pd.DataFrame(pd.read_csv(os.path.abspath(csvfile), header=None, delimiter=sep, index_col=0))
            self.annobj.obs = obsaux
            self.annobj.obs_names = list(obsaux.index)


    def load_count_matrix(self,filename, sep=",", varanno=None, obsanno=None):
        """
        Load count matrix and put this into a summarized experiment.
        Add anndata.var (col description) and anndata.obs (row description) annotation

            ---- var ---
        |   T1_s1,T2_s2,T3_s,T4_1
        |   G1,10,20,30,40
        obs G2,5,10,20,30
        |   G3,6,7,8,9

        varanno input example:
        T1_s1  F  Tissue1  Sample1  rnaseq
        T2_s2  M  Tissue2  Sample2  rnaseq
        T3_s1  F  Tissue3  Sample1  rnaseq
        T4_s2  M  Tissue4  Sample2  rnaseq

        obsanno input example:
        G1,hg19,t1,chr1,1111,-
        G2,hg19,t2,chr2,2222,-
        G3,hg19,t3,chr3,3333,-

        :param filename: .csv file containing a n_obs x n_vars count matrix
        :param sep: separator, this should be the same for count_marix, varanno and obsanno
        :param varanno: additional annotation for cols (e.g. sample_tissue description)
        :param obsanno: additional annotation for rows (e.g. gene id description)
        """
        abs_path = os.path.abspath(filename)

        # read count matrix
        self.annobj = anndata.read_csv(abs_path, delimiter=sep)

        # read var data (samples description)
        self.load_vars(varanno, sep)

        # read obs data (index description e.g gene annotation)
        self.load_obs(obsanno, sep)



    def load_anndata_from_file(self,filename, backed=False, varanno=None, obsanno=None, sep=","):
        """
        Load anndata format specific data into an anndata object.
        :param filename: .h5ad file containing n_obs x n_vars count matrix and further annotations
        :param backed: default False - see anndata.read_h5ad documentation https://media.readthedocs.org/pdf/anndata/latest/anndata.pdf
                       if varanno and obsanno are provided please set backed = r+
        :param varanno: additional annotation for cols (e.g. sample_tissue description)
        :param obsanno: additional annotation for rows (e.g. gene id description)
        :param sep: separator for varanno and obsanno files
        :return: anndata object
        """
        abspath = os.path.abspath(filename)
        self.annobj = anndata.read_h5ad(abspath, backed=backed)
        try:

            if varanno or obsanno:
                if backed != 'r+':
                    raise BackedError

        except BackedError:
            print("Exception [varanno] or [obsanno] provided! Please set [backed='r+']")
            exit(1)

        # read var data
        self.load_vars(varanno, sep)
        # read obs data
        self.load_obs(annobj, obsanno, sep)


    def save(self, outname=None):
        """
        Write .h5ad-formatted hdf5 file and close a potential backing file. Default gzip file
        :param annobj:
        :param outname: name of the output file (needs to end with .h5ad)
        :return output filename (if none specified then this will be random generated)
        """

        if outname:
            abspath = os.path.abspath(outname)
            name = abspath
        else:
            name = os.path.abspath("tmp" + str(int(time.time())) + ".h5ad")

        # convert header to string (avoid bug)
        self.annobj.var_names = [str(v) for v in annobj.var_names]
        self.annobj.obs_names = [str(o) for o in annobj.obs_names]
        self.annobj.var.rename(index={r: str(r) for r in list(annobj.var.index)},
                          columns={c: str(c) for c in list(annobj.var.columns)},
                          inplace=True)
        self.annobj.obs.rename(index={r: str(r) for r in list(annobj.obs.index)},
                          columns={c: str(c) for c in list(annobj.obs.columns)},
                          inplace=True)

        self.annobj.write(name)

        return name



    def print_anndata(self):
        print("anndata.X")
        print(self.annobj.X)
        print("anndata.var")
        print(self.annobj.var_names)
        print(self.annobj.var)
        print("anndata.obs")
        print(self.annobj.obs_names)
        print(self.annobj.obs)
        print()

    @property
    def annobj(self):
        return self.annobj

    @property
    def varanno(self):
        return self.annobj.var

    @property
    def obsanno(self):
        return self.annobj.obs

    def get_count_matrix(self):
        return self.annobj.X

    @annobj.setter
    def annobj(self,annobj):
        self.annobj = annobj

    @varanno.setter
    def varanno(self,varanno,sep):
        """
        Set anndata.var
        :param varanno: filename
        :param sep: seprator
        :return: bool
        """
        if self.annobj:
            self.load_vars(varanno,sep)
            return True
        return False # failed to set

    @obsanno.setter
    def obsanno(self,obsanno,sep):
        """
        Set anndata.obs
        :param obsanno: filename
        :param sep: seprator
        :return: bool
        """
        if self.annobj:
            self.load_obs(obsanno,sep)
            return True
        return False # failed to set
