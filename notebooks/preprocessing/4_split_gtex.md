
### Goal

Balanced split of the data. Save into files the individuals for train, test and valid.
501 available individuals have (i) #tissues >= 2 and (ii) DNA-seq

###  TODO
- split dataset by individuals and taking care to have a balanced set in terms of #samples (some invidivuals tent to have more samples as the others), by gender and by WGS/WES

### Conclusions

Under `/s/project/rep/processed/gtex/recount` are the list of individuals for train, valid, test.

`train_individuals.txt`,`valid_individuals.txt`, `test_individuals.txt` - list of individuals

`train_individuals.csv`,`valid_individuals.csv`, `test_individuals.csv` - description of samples for these individuals

#### Split summary:

Expected split: [0.6, 0.2, 0.2]

train  #individuals =  286  fraction =  0.5708582834331337<br>
valid  #individuals =  100  fraction =  0.1996007984031936<br>
test  #individuals =  115  fraction =  0.22954091816367264<br>

train  #samples =  (5389, 14)  fraction =  0.6047581640668837<br>
valid  #samples =  (1820, 14)  fraction =  0.204241948153967<br>
test  #samples =  (1702, 14)  fraction =  0.19099988777914936<br>

train  female =  0.36713286713286714 male =  0.6328671328671329<br>
valid  female =  0.35 male =  0.65<br>
test  female =  0.34782608695652173 male =  0.6521739130434783<br>


train  WGS =  0.958041958041958 WES =  0.04195804195804196<br>
valid  WGS =  0.95 WES =  0.05<br>
test  WGS =  0.9217391304347826 WES =  0.0782608695652174<br>


```python
import os
import sys

import pandas as pd
import numpy as np

import rep.preprocessing_new as p
from rep.constants import ANNDATA_CST as a

%aimport
```

    Modules to reload:
    all-except-skipped
    
    Modules to skip:
    



```python
file = os.path.join(os.readlink(os.path.join("..","..","data")),"processed","gtex","recount","recount_gtex.h5ad")
gtex = p.RepAnnData.read_h5ad(file)
```


```python
# check if every samples have at least 2 tissues
count_tissues = gtex.samples[['Individual','Tissue']].groupby(['Individual'], sort=True).size()
count_filter = count_tissues[count_tissues > 1]
len(count_filter)
```




    501




```python
gtex.samples[['Individual','Gender']].drop_duplicates().shape
```




    (501, 2)




```python
%time (train,valid,test) = p.split_by_individuals(gtex,groupby=['Gender','Indiv_Seq_Assay'])
```

    Total individuals: 501
    Individual split before balancing:  298 98 105
    Individual split after balancing:  286 100 115
    CPU times: user 5.56 s, sys: 0 ns, total: 5.56 s
    Wall time: 5.56 s



```python
#Total individuals: 501
#Individual split before balancing:  298 98 105
#('Individual split: ', 286, 100, 115)
```

#### Check the uniformity of the datasets


```python
samples_info = gtex.samples
samples_info[['Gender','Individual']].groupby('Gender').size()
```




    Gender
    female    3376
    male      5535
    dtype: int64




```python
df_train = samples_info[samples_info['Individual'].isin(train)]
df_valid = samples_info[samples_info['Individual'].isin(valid)]
df_test = samples_info[samples_info['Individual'].isin(test)]
```

- check woman and man ratio


```python
states = ['train','valid','test']
dict_states = {'train':df_train,'valid':df_valid,'test':df_test}
dict_states_indiv = {'train':train,'valid':valid,'test':test}
total_len = len(train) + len(test) + len(valid)

print("Expected split: [0.6, 0.2, 0.2]")

for s in states: print(s," #individuals = ", len(dict_states_indiv[s]), " fraction = " ,(len(dict_states_indiv[s])/total_len))

print()
for s in states: print(s," #samples = ", dict_states[s].shape, " fraction = ", dict_states[s].shape[0]/samples_info.shape[0])

print()
for s in states:
    df = dict_states[s]
    l = dict_states_indiv[s]
    print(s," female = ", (df[df['Gender'] == 'female'][['Individual','Gender']].drop_duplicates().shape[0] / len(l)), "male = ", (df[df['Gender'] == 'male'][['Individual','Gender']].drop_duplicates().shape[0] / len(l)))

print()
for s in states:
    df = dict_states[s]
    l = dict_states_indiv[s]
    print(s," WGS = ", (df[df['Indiv_Seq_Assay'] == 'WGS'][['Individual','Indiv_Seq_Assay']].drop_duplicates().shape[0] / len(l)), "WES = ", (df[df['Indiv_Seq_Assay'] == 'WES'][['Individual','Indiv_Seq_Assay']].drop_duplicates().shape[0] / len(l))) 
```

    Expected split: [0.6, 0.2, 0.2]
    train  #individuals =  286  fraction =  0.5708582834331337
    valid  #individuals =  100  fraction =  0.1996007984031936
    test  #individuals =  115  fraction =  0.22954091816367264
    
    train  #samples =  (5389, 14)  fraction =  0.6047581640668837
    valid  #samples =  (1820, 14)  fraction =  0.204241948153967
    test  #samples =  (1702, 14)  fraction =  0.19099988777914936
    
    train  female =  0.36713286713286714 male =  0.6328671328671329
    valid  female =  0.35 male =  0.65
    test  female =  0.34782608695652173 male =  0.6521739130434783
    
    train  WGS =  0.958041958041958 WES =  0.04195804195804196
    valid  WGS =  0.95 WES =  0.05
    test  WGS =  0.9217391304347826 WES =  0.0782608695652174



```python
# save individuals
path = os.path.join(os.readlink(os.path.join("..","..","data")),"processed","gtex","recount")
for s in states:
    with open(os.path.join(path,s + "_individuals.txt"), 'w') as f: 
        for item in dict_states_indiv[s]:
            f.write("%s\n" % item)
```


```python
# save data information per training
for s in states:
    file = os.path.join(path,s + "_individuals.csv")
    dict_states[s].to_csv(file, encoding='utf-8', sep="\t")
```
