
### Goal
Parse and clean *recount2*, gene counts to AnnData format.

Use first `../../data/raw/gtex/recount/splitGtexData.R` - to split in counts, rowData, colData (the data was available as .RData object)

### TODO
- Get GTEx metadata:
  - check individuals if they have WGS/WXS or RNA-SEQ whole blood
  - integrate GTEx metadata (Assay, Gender, Sample Id, Individual ...) into the recount2 sample_description
- Create our custom recount2 metadata containing: Assay, Gender, Sample id, Tissue, Parent Tissues, Individual ...
- Create the Summarized experiment  + store data as HDF5 format

### Conclusions
`GTEx` metadata - short analysis:
- Total number of individuals: 723
- #individuals having WGS or WES: 679
- #individuals having Whole Blood RNA-Seq: 427
- #individuals having at least 2 Tissues available: 544
- #individuals having at least 2 Tissues available + having WGS or WES: 501

`recount2` Summarized experiment: `/s/project/rep/processed/gtex/recount/recount_gtex.h5ad`



```python
import sys
import os

import pandas as pd
import numpy as np

from rep import preprocessing as p
```

#### 1. GTEx metadata - short analysis


```python
# add metadata patient related - used from gtex v7
metadata = '/s/project/gtex-processed/gene_counts_v7/SraRunTable.txt'
gtex_meta = p.load_df(metadata,delimiter="\t",header=0,index_col=None)
```


```python
# get individual information
gtex_meta_short = gtex_meta[['Sample_Name_s','sex_s','Assay_Type_s','Instrument_s','body_site_s','histological_type_s']]
gtex_meta_short.loc[:,'Sample_Name'] = pd.Series(gtex_meta_short['Sample_Name_s']).str.split("_",expand=True)[0]
aux = pd.Series(gtex_meta_short['Sample_Name']).str.split("-",expand=True)
gtex_meta_short.loc[:,'Individual'] = aux[0].map(str) + "-" + aux[1]
gtex_meta_unique = gtex_meta_short.iloc[:,1:]
gtex_meta_unique.drop_duplicates(inplace=True)
gtex_meta_unique.iloc[:3,:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sex_s</th>
      <th>Assay_Type_s</th>
      <th>Instrument_s</th>
      <th>body_site_s</th>
      <th>histological_type_s</th>
      <th>Sample_Name</th>
      <th>Individual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>RNA-Seq</td>
      <td>Illumina MiSeq</td>
      <td>Skin - Sun Exposed (Lower leg)</td>
      <td>Skin</td>
      <td>GTEX-WEY5-1826-SM-5CHRT</td>
      <td>GTEX-WEY5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>male</td>
      <td>RNA-Seq</td>
      <td>Illumina MiSeq</td>
      <td>Testis</td>
      <td>Testis</td>
      <td>GTEX-SUCS-1326-SM-5CHQI</td>
      <td>GTEX-SUCS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>male</td>
      <td>RNA-Seq</td>
      <td>Illumina MiSeq</td>
      <td>Thyroid</td>
      <td>Thyroid</td>
      <td>GTEX-SUCS-0226-SM-5CHQG</td>
      <td>GTEX-SUCS</td>
    </tr>
  </tbody>
</table>
</div>



- Count total number of individuals


```python
subset = gtex_meta_unique[['body_site_s','Assay_Type_s','Individual']]
subset.drop_duplicates(inplace=True)
subset[:3]
```

    /opt/modules/i12g/anaconda/3-5.0.1/envs/rep/lib/python3.6/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>body_site_s</th>
      <th>Assay_Type_s</th>
      <th>Individual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Skin - Sun Exposed (Lower leg)</td>
      <td>RNA-Seq</td>
      <td>GTEX-WEY5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Testis</td>
      <td>RNA-Seq</td>
      <td>GTEX-SUCS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Thyroid</td>
      <td>RNA-Seq</td>
      <td>GTEX-SUCS</td>
    </tr>
  </tbody>
</table>
</div>




```python
indiv = subset['Individual'].drop_duplicates()
indiv.count()
```




    723



- Count individuals having WGS or WES


```python
count_dnaseq_individuals = subset[subset['Assay_Type_s'].isin(['WGS','WXS'])]['Individual'].drop_duplicates()
count_dnaseq_individuals.count()
```




    679



- Count individuals having whole blood RNA-Seq


```python
c = subset[subset['Assay_Type_s'].isin(['RNA-Seq'])]
count_rnaseq_individuals = c[c['body_site_s'].isin(['Whole Blood'])]['Individual'].drop_duplicates()
count_rnaseq_individuals.count()
```




    427



- Count individuals having at least two tissues available


```python
c = subset[subset['Assay_Type_s'].isin(['RNA-Seq'])]
count_tissues = c.groupby(['Individual','Assay_Type_s'], sort=True).size()
count_filter = count_tissues[count_tissues > 1]
len(count_filter)
```




    544



- Count indiv with at least two tissues + having WGS/WES


```python
tokeep = list(set(count_dnaseq_individuals.tolist()) & set(count_filter.index.get_level_values('Individual')))
len(tokeep)
```




    501




```python
# this array will be further used to complete the recount2 sample data information
gtex_tokeep = gtex_meta_unique[gtex_meta_unique['Individual'].isin(tokeep)]
gtex_tokeep.rename(columns={"histological_type_s": "Parent_Tissue", "body_site_s": "Tissue"},inplace=True)
gtex_tokeep[:3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sex_s</th>
      <th>Assay_Type_s</th>
      <th>Instrument_s</th>
      <th>Tissue</th>
      <th>Parent_Tissue</th>
      <th>Sample_Name</th>
      <th>Individual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>RNA-Seq</td>
      <td>Illumina MiSeq</td>
      <td>Skin - Sun Exposed (Lower leg)</td>
      <td>Skin</td>
      <td>GTEX-WEY5-1826-SM-5CHRT</td>
      <td>GTEX-WEY5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>male</td>
      <td>RNA-Seq</td>
      <td>Illumina MiSeq</td>
      <td>Testis</td>
      <td>Testis</td>
      <td>GTEX-SUCS-1326-SM-5CHQI</td>
      <td>GTEX-SUCS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>male</td>
      <td>RNA-Seq</td>
      <td>Illumina MiSeq</td>
      <td>Thyroid</td>
      <td>Thyroid</td>
      <td>GTEX-SUCS-0226-SM-5CHQG</td>
      <td>GTEX-SUCS</td>
    </tr>
  </tbody>
</table>
</div>



- Individuals without DNA-seq


```python
# list individuals without WGS or WXS
black = list(set(indiv.tolist())-set(count_dnaseq_individuals.tolist()))
```

#### 2. GTEx to AnnData object
GTEx keep only individuals with WES/WGS having at least 2 tissues available.
Parse to the recommanded format


```python
data_path = os.readlink(os.path.join("..","..","data"))
path = os.path.join(data_path,"raw","gtex","recount","version2")
counts_file = "recount_gene_counts.csv"
rowdata_file = "recount_rowdata.csv"
coldata_file = "recount_coldata.csv"
```


```python
# might take longer to create a large object
annobj = p.create_anndata(os.path.join(path,counts_file), sep="\t", samples_anno=os.path.join(path,coldata_file),genes_anno=os.path.join(path,rowdata_file))
```


```python
# parse samples metadata to our standardize form: Individual, Parent_Tissue, Tissue, Indiv_Seq_Assay, Gender
sample_description = annobj.var
sample_description[:3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>project</th>
      <th>sample</th>
      <th>experiment</th>
      <th>mapped_read_count</th>
      <th>avg_read_length</th>
      <th>sampid</th>
      <th>smatsscr</th>
      <th>smts</th>
      <th>smtsd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SRR660824</th>
      <td>SRP012682</td>
      <td>SRS389722</td>
      <td>SRX222703</td>
      <td>170790002</td>
      <td>152</td>
      <td>GTEX-QMR6-1926-SM-32PL9</td>
      <td>3.0</td>
      <td>Lung</td>
      <td>Lung</td>
    </tr>
    <tr>
      <th>SRR2166176</th>
      <td>SRP012682</td>
      <td>SRS1036203</td>
      <td>SRX1152700</td>
      <td>191059974</td>
      <td>500</td>
      <td>GTEX-T5JC-0011-R11A-SM-5S2RX</td>
      <td>NaN</td>
      <td>Brain</td>
      <td>Brain - Cerebellar Hemisphere</td>
    </tr>
    <tr>
      <th>SRR606939</th>
      <td>SRP012682</td>
      <td>SRS333474</td>
      <td>SRX199032</td>
      <td>159714774</td>
      <td>136</td>
      <td>GTEX-POMQ-0326-SM-2I5FO</td>
      <td>1.0</td>
      <td>Heart</td>
      <td>Heart - Left Ventricle</td>
    </tr>
  </tbody>
</table>
</div>




```python
aux = pd.Series(sample_description['sampid']).str.split("-",expand=True)
sample_description.loc[:,'Individual'] = aux[0].map(str) + "-" + aux[1]
sample_description.rename(index=str, columns={"smts": "Parent_Tissue", "smtsd": "Tissue","sampid":"Sample_Name"},inplace=True)
sample_description[:3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>project</th>
      <th>sample</th>
      <th>experiment</th>
      <th>mapped_read_count</th>
      <th>avg_read_length</th>
      <th>Sample_Name</th>
      <th>smatsscr</th>
      <th>Parent_Tissue</th>
      <th>Tissue</th>
      <th>Individual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SRR660824</th>
      <td>SRP012682</td>
      <td>SRS389722</td>
      <td>SRX222703</td>
      <td>170790002</td>
      <td>152</td>
      <td>GTEX-QMR6-1926-SM-32PL9</td>
      <td>3.0</td>
      <td>Lung</td>
      <td>Lung</td>
      <td>GTEX-QMR6</td>
    </tr>
    <tr>
      <th>SRR2166176</th>
      <td>SRP012682</td>
      <td>SRS1036203</td>
      <td>SRX1152700</td>
      <td>191059974</td>
      <td>500</td>
      <td>GTEX-T5JC-0011-R11A-SM-5S2RX</td>
      <td>NaN</td>
      <td>Brain</td>
      <td>Brain - Cerebellar Hemisphere</td>
      <td>GTEX-T5JC</td>
    </tr>
    <tr>
      <th>SRR606939</th>
      <td>SRP012682</td>
      <td>SRS333474</td>
      <td>SRX199032</td>
      <td>159714774</td>
      <td>136</td>
      <td>GTEX-POMQ-0326-SM-2I5FO</td>
      <td>1.0</td>
      <td>Heart</td>
      <td>Heart - Left Ventricle</td>
      <td>GTEX-POMQ</td>
    </tr>
  </tbody>
</table>
</div>




```python
new = sample_description.merge(gtex_tokeep,how='left').fillna("").set_index(sample_description.index)
new.rename(index=str, columns={"sex_s": "Gender"},inplace=True)
new[:3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>project</th>
      <th>sample</th>
      <th>experiment</th>
      <th>mapped_read_count</th>
      <th>avg_read_length</th>
      <th>Sample_Name</th>
      <th>smatsscr</th>
      <th>Parent_Tissue</th>
      <th>Tissue</th>
      <th>Individual</th>
      <th>Gender</th>
      <th>Assay_Type_s</th>
      <th>Instrument_s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SRR660824</th>
      <td>SRP012682</td>
      <td>SRS389722</td>
      <td>SRX222703</td>
      <td>170790002</td>
      <td>152</td>
      <td>GTEX-QMR6-1926-SM-32PL9</td>
      <td>3</td>
      <td>Lung</td>
      <td>Lung</td>
      <td>GTEX-QMR6</td>
      <td>male</td>
      <td>RNA-Seq</td>
      <td>Illumina HiSeq 2000</td>
    </tr>
    <tr>
      <th>SRR2166176</th>
      <td>SRP012682</td>
      <td>SRS1036203</td>
      <td>SRX1152700</td>
      <td>191059974</td>
      <td>500</td>
      <td>GTEX-T5JC-0011-R11A-SM-5S2RX</td>
      <td></td>
      <td>Brain</td>
      <td>Brain - Cerebellar Hemisphere</td>
      <td>GTEX-T5JC</td>
      <td>male</td>
      <td>RNA-Seq</td>
      <td>Illumina HiSeq 2000</td>
    </tr>
    <tr>
      <th>SRR606939</th>
      <td>SRP012682</td>
      <td>SRS333474</td>
      <td>SRX199032</td>
      <td>159714774</td>
      <td>136</td>
      <td>GTEX-POMQ-0326-SM-2I5FO</td>
      <td>1</td>
      <td>Heart</td>
      <td>Heart - Left Ventricle</td>
      <td>GTEX-POMQ</td>
      <td>female</td>
      <td>RNA-Seq</td>
      <td>Illumina HiSeq 2000</td>
    </tr>
  </tbody>
</table>
</div>




```python
new.shape, len(list(set(new['Individual'].tolist()) & set(tokeep))), len(list(set(new['Individual'].tolist())))
```




    ((9662, 13), 501, 551)




```python
# assign right assay to the experiments
new.loc[:,'Indiv_Seq_Assay'] = pd.Series('WGS', index=new.index)
for indiv in tokeep:
    assay = list(set(gtex_tokeep[gtex_tokeep['Individual'] == indiv]['Assay_Type_s'].tolist()))
    if 'WGS' not in assay and 'WXS' in assay:
        # replace assay to whole exome
        new.loc[new['Individual']  == indiv, 'Indiv_Seq_Assay'] = 'WES'
    elif 'WGS' not in assay and 'WXS' not in assay:
        print("Error - individual has not DNA-seq: ",indiv)
```


```python
# remove the list of experiments which do not have WGS/WES
new_filtered = new[new['Individual'].isin(tokeep)]
```


```python
annobj.var = new
filtered_annobj = annobj[:,new_filtered.index.tolist()]
filtered_annobj.var_names = filtered_annobj.var.index.tolist()

for indiv in tokeep:
    gender = filtered_annobj.var.loc[filtered_annobj.var['Individual'] == indiv][['Individual','Gender']].drop_duplicates()
    if gender.shape[0] != 1:
        genders = gender['Gender'].tolist()
        if 'male' in genders and 'female' in genders: print("Erroorrrr")
        elif 'male' in genders:
            filtered_annobj.var.loc[filtered_annobj.var['Individual'] == indiv,'Gender'] = 'male'
        elif 'female' in genders: 
            filtered_annobj.var.loc[filtered_annobj.var['Individual'] == indiv,'Gender'] = 'female'
        else:
            print("could not found gender ", genders, " ", indiv)
```


```python
filtered_annobj.var.shape, new.shape, new_filtered.shape, len(tokeep), filtered_annobj.var['Individual'].drop_duplicates().shape
```




    ((8911, 14), (9662, 14), (8911, 14), 501, (501,))




```python
# save RepAnnData object to HDF5 format
import rep.preprocessing_new as pn
output_file = os.path.join(data_path,"processed","gtex","recount","recount_gtex.h5ad")
tmp = filtered_annobj.transpose()
repobj = pn.RepAnnData(X=tmp.X, samples_obs=tmp.obs, genes_var=tmp.var)
repobj.save(outname=output_file)
```

    /s/project/rep/processed/gtex/recount/recount_gtex.h5ad


    ... storing 'symbol' as categorical
    ... storing 'project' as categorical
    ... storing 'smatsscr' as categorical
    ... storing 'Parent_Tissue' as categorical
    ... storing 'Tissue' as categorical
    ... storing 'Individual' as categorical
    ... storing 'Gender' as categorical
    ... storing 'Assay_Type_s' as categorical
    ... storing 'Instrument_s' as categorical
    ... storing 'Indiv_Seq_Assay' as categorical





    '/data/nasif12/home_if12/giurgiu/rep_gagneur/rep/notebooks/preprocessing/tmp1547640828.h5ad'


