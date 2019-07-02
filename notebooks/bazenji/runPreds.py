#/bin/python


class rangeInterval(object): #renamed to avoid conflict with genomic interval
    """
    stores window-range-info
    as usual in python, an [ ) interval"""
    def __init__(self, n_up,n_down,anchor=None):
        if n_up==n_down:
            raise AttributeError("An interval without elements is not supported. Your upper and lower bound(exclusive) are the same.")
            
        
        self.lower=n_up
        self.upper=n_down
        self.anchor=anchor
        if anchor is None:
            self.anchor=int((self.lower+self.upper)/2)
    def setAnchor(self,newAnchor):
        self.lower=newAnchor-(self.anchor-self.lower)
        self.upper=newAnchor+(self.upper-self.anchor)
        self.anchor=newAnchor
    def __str__(self):
        return "[{},{},{})".format(self.lower,self.anchor,self.upper)
    def __repr__(self):
        return str(self)
    def __len__(self):
        return self.upper-self.lower


from kipoiseq.extractors import MultiSampleVCF
class AllelicMultiSampleVCF(MultiSampleVCF):
    """supports variant-query according to genotype
    #TODO clean up code repetition... all in one method...
    """
    def _has_variant_homo(self, variant, sample_id):
        return variant.gt_types[self.sample_mapping[sample_id]] == 2 #might have to use ==3 since 2 is "unknown/*" ?
    def _has_variant_hetero(self, variant, sample_id):
        return variant.gt_types[self.sample_mapping[sample_id]] == 1
    
    def fetch_variants_homo(self, interval, sample_id=None):
        for v in self(self._region(interval)):
            if sample_id is None or self._has_variant_homo(v, sample_id):
                yield v
    def fetch_variants_hetero(self, interval, sample_id=None):
        for v in self(self._region(interval)):
            if sample_id is None or self._has_variant_hetero(v, sample_id):
                yield v      
    

from kipoiseq.extractors import BaseExtractor,VariantSeqExtractor
import pdb
class AllelicVCFSeqExtractor(BaseExtractor):
    def __init__(self, fasta_file, vcf_file):
        self.fasta_file = fasta_file
        self.vcf_file = vcf_file
        self.variant_extractor = VariantSeqExtractor(fasta_file)
        self.vcf = AllelicMultiSampleVCF(vcf_file)
  
    def get_allelic_variants(self,homoVars,heteroVars):
        """
        hetero-allelic variants are only used on first allele, homo- on both
        """
        return [homoVars+heteroVars,homoVars]
        
    def extract(self, interval, anchor=None, sample_id=None, fixed_len=True):
        
        print("getting variants...")
        homoVars=list(self.vcf.fetch_variants_homo(interval, sample_id)) #when mutating them, lists are way easier and not more costly than generators
        heteroVars=list(self.vcf.fetch_variants_hetero(interval, sample_id))
        
        print("sorting variants, homo:{}, hetero:{}...".format(len(homoVars),len(heteroVars)))

        vars1,vars2=self.get_allelic_variants(homoVars,heteroVars)

        print("sortED variants, 1st:{}, 2nd:{}...".format(len(vars1),len(vars2)))
        print("getting mutated seq...")
        
        allele1=self.variant_extractor.extract(
                interval, variants=vars1,
                anchor=anchor, fixed_len=fixed_len)        
        allele2=self.variant_extractor.extract(
                interval, variants=vars2,
                anchor=anchor, fixed_len=fixed_len)
        
        return [allele1,allele2]




class ZarrVariantPredWriter(object):
    def __init__(self,outputFile,genes,samples,cAlleles,genePositions,predSize):
        """set up data array, annotate with attributes
        cAlleles and predSize are given as raw numbers/dimension size
        genes, samples and genePositions are given as elements
        
        """
        import zarr
        self.filename=outputFile
        self.cGenes=len(genes)
        self.cSamples=len(samples)
        self.cGenePositions=len(genePositions)
        self.cAlleles=cAlleles
        self.cPredSize=predSize
        #Allele got canceled, just append allele 2 to allele1
        self.data=zarr.open(self.filename, mode='w', shape=(self.cGenes,self.cSamples,self.cAlleles,self.cGenePositions,self.cPredSize),
                 dtype='f8') #TODO: precision right? chunksize left out for now, are there default values? "chunks=(1000, 1000),"
        
        self.data.attrs['genes'] = self.as_idx_dict(genes)
        self.data.attrs["samples"]=self.as_idx_dict(samples)
        self.data.attrs["genePositions"]=self.as_idx_dict(genePositions)
    @staticmethod    
    def as_idx_dict(lis):
        """for O1 lookup"""
        dic={}
        for idx,elem in enumerate(lis):
            dic[elem]=idx
        return dic
        
    def batch_write(self,annotatedPreds):
        """plug in predictions in right position, given by metadata and annotation-array

        gene, sample and allele can differ from prediction to prediction
        expects all locations on gene in one go
        """
        print("....writing batch")
        for inputIdx,predPerInput in enumerate(annotatedPreds["preds"]):
            geneId=annotatedPreds["metadata"]["gene_id"][inputIdx]
            sampleId=annotatedPreds["metadata"]["sample_id"][inputIdx]
            
            alleleIdx=annotatedPreds["metadata"]["allele_id"][inputIdx]
            geneIdx=self.data.attrs["genes"][geneId]
            sampleIdx=self.data.attrs["samples"][sampleId]
            self.data[geneIdx,sampleIdx,alleleIdx]=predPerInput

        return
    def close(self):
        """
        According to https://zarr.readthedocs.io/en/stable/api/creation.html:
        Notes
        There is no need to close an array. Data are automatically flushed to the file system."""
        pass 


from torch.utils.data import Dataset
#formerly "repDataloader"
class RepDataset(Dataset):
    
    from pybedtools import Interval
    
    def __init__(self,refGenFasta,vcfFile,tssFile,n_upstream,n_downstream,sample_ids,gene_ids=None):
        """set instance-vars, read TSS completely, load (dummy) VCFfastaExtractor"""

        import pandas as pd

        from kipoiseq.transforms import ReorderedOneHot
        self.transform=ReorderedOneHot()
        
        
        self.vcfFile=vcfFile
        self.refGenFasta=refGenFasta
        self.varExtractor=None
        #AllelicVCFSeqExtractor(self.refGenFasta,vcfFile)#VcfFastaDummy(vcfFile,refGenFasta)
        
        self.sampleIDs=sample_ids
        self.interval=None
        self.refInterval=rangeInterval(n_upstream,n_downstream)
        
        
        #TODO use only prot-coding? Separate file or bound to schema of example file?
        self.tssCollection=pd.read_csv(tssFile) # needs field "TSS" with RegGen position and "strand" with "+" or "-"
        if gene_ids:
            self.geneIDs= gene_ids
            tssCollection=tssCollection.loc[tssCollection['id'].isin( gene_ids)]
            self.geneIDs=self.tssCollection["id"].values #maybe specified IDs not in file...
        else:
            self.geneIDs=self.tssCollection["id"].values
        


    def __len__(self):
        return len(self.sampleIDs)*len(self.geneIDs)
    def __getitem__(self,idx):
        import numpy as np
        import copy
        """transform idx to sample- and tss-id, then get both alleles from extractor"""
        
        #each worker needs own file handle:
        if self.varExtractor==None:
            self.varExtractor=AllelicVCFSeqExtractor(self.refGenFasta,self.vcfFile)
        if self.interval==None:
            self.interval=copy.deepcopy(self.refInterval)
        
        sampleIdx=int(idx/len(self.tssCollection))
        tssIdx=int(idx%len(self.tssCollection))
        
        
        chromo=self.tssCollection.loc[tssIdx,"seqnames"]
        strand=self.tssCollection.loc[tssIdx,"strand"]
        tssStart=self.tssCollection.loc[tssIdx,"TSS"]
        
        self.interval.setAnchor(tssStart)
        genomic_interval=self.getGenomicInterval(chromo,strand)  
        alleles= self.varExtractor.extract(genomic_interval,self.interval.anchor,self.sampleIDs[sampleIdx]) #strand-info where?
        
        #formatting output
        gene_id=self.tssCollection.loc[tssIdx,"id"]
        sample_id=self.sampleIDs[sampleIdx]
        returnBatch={}
        returnBatch["inputs"]=[self.transform(alleles[0].upper()),self.transform(alleles[1].upper())]
        #need np.array-conversion, or it will concat on other dimension for batchsize>1
        returnBatch["metadata"]={"gene_id":np.array([gene_id]*2),"sample_id":np.array([sample_id]*2),"allele_id":np.array([0,1])} 
        return returnBatch
    
    def getGenomicInterval(self,chromosome,strand):
        """converts internal interval to bedInterval as used in extractor"""
        from pybedtools import Interval
        
        return Interval(chromosome, start=self.interval.lower, end=self.interval.upper, strand=strand, otherfields=None)
    def get_gene_ids(self):
        return self.geneIDs
        


#params
basenjiSeqSize=131072 
tssFile="testTSS.csv"
refGen='/s/genomes/human/hg38/hg38.fa'
vcfFile="/data/ouga/home/ag_gagneur/reinharj/REP/rep/notebooks/bazenji/test.vcf.gz"
outfile="test.zarr"

sample1="GTEX-QV31-0003-SM-5URDH" #homo-neg at first snp (1_10146_AC_A_b37) 
sample2="GTEX-R55E-0003-SM-5URBX" #hetero at first snp (1:10146)
sample5="GTEX-N7MS-0009-SM-5JK3E" #homo-neg for snp at 1_10177_A_C_b37 (sample 1/2 homo-unknown there)
sample23="GTEX-OXRK-0004-SM-5JK32" #homo-pos for 1_10327_T_C_b37
samples=[sample1,sample2,sample5,sample23]

cAlleles=2
predSize=4229
binsOnSeq=[int(960/2)] #must be continous!




##MAIN routine!
def pred_and_store():
    inputData = RepDataset(refGen,vcfFile,tssFile,-int(basenjiSeqSize/2),int(basenjiSeqSize/2),samples)
    geneIDs= inputData.get_gene_ids()
    
    from torch.utils import data
    from kipoi.data_utils import numpy_collate_concat
    generator = data.DataLoader(inputData,batch_size=1, collate_fn=numpy_collate_concat,num_workers=30)

    import kipoi
    model = kipoi.get_model('Basenji')



    writer=ZarrVariantPredWriter(outfile,geneIDs,samples,cAlleles,binsOnSeq,predSize)


    for batch in generator:
        preds=model.predict_on_batch(batch['inputs'])
        #TODO choose/compute right bin AFTER TSS, not over. For now, just use middle
        #see https://github.com/kipoi/models/issues/87 for info n dimensions

        #remember: id_inputvec(batchsize,here allele1 and 2), id_genomicInterval, id_features/"tissues"
        tssExpr=preds[:,binsOnSeq[0]:(binsOnSeq[-1]+1),:]
        batch["preds"]=tssExpr
        writer.batch_write(batch)

    writer.close()
pred_and_store()


