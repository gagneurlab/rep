
class ANNDATA_CST:
    GENES = 'genes'
    SAMPLES = 'samples'

class GTEX_CST:
    INDIVIDUAL = 'Individual'
    FROM_TISSUE = 'From_tissue'
    TO_TISSUE = 'To_tissue'
    FROM_PARENT_TISSUE = 'From_parent_tissue'
    TO_PARENT_TISSUE = 'To_parent_tissue'
    INDIV_SEQ_ASSAY = 'Indiv_Seq_Assay'
    TISSUE = 'Tissue'
    PARENT_TISSUE = 'Parent_Tissue'
    GENDER = 'Gender'
    FROM_SAMPLE = 'From_sample'
    TO_SAMPLE = 'To_sample'
    TYPE = 'Type'
    INDEX = 'Index'
    
class METADATA_CST:
    GENE_METADATA = 'gene_metadata'
    INDIV_TISSUE_METADATA = 'patient_tissue_metadata'


# rank by number of samples
TOP_INDIVIDUALS = ['GTEX-12WSD',
 'GTEX-ZAB4',
 'GTEX-13OW6',
 'GTEX-131YS',
 'GTEX-NPJ8',
 'GTEX-11GSP',
 'GTEX-YFC4',
 'GTEX-13OVJ',
 'GTEX-13OW8',
 'GTEX-12ZZX',
 'GTEX-T5JC',
 'GTEX-ZUA1',
 'GTEX-WY7C',
 'GTEX-T6MN',
 'GTEX-WZTO',
 'GTEX-Y3IK',
 'GTEX-Q2AG',
 'GTEX-13NYB',
 'GTEX-1313W',
 'GTEX-11TT1',
 'GTEX-Y114',
 'GTEX-11DXX',
 'GTEX-RU72',
 'GTEX-N7MS',
 'GTEX-WYVS',
 'GTEX-XV7Q',
 'GTEX-11EQ9',
 'GTEX-13OVL',
 'GTEX-13G51',
 'GTEX-13O3O',
 'GTEX-TSE9',
 'GTEX-13PVQ',
 'GTEX-XUJ4',
 'GTEX-YEC4',
 'GTEX-U3ZN',
 'GTEX-13N1W',
 'GTEX-13O3Q',
 'GTEX-ZVT2',
 'GTEX-WYJK',
 'GTEX-X4EP',
 'GTEX-12BJ1',
 'GTEX-WHSE',
 'GTEX-ZDXO',
 'GTEX-11WQK',
 'GTEX-13O61',
 'GTEX-WFON',
 'GTEX-11ZUS',
 'GTEX-11P7K',
 'GTEX-11TUW',
 'GTEX-ZDYS']

TOP_HIGH_VARIANCE_GENES = ['ENSG00000171195.10',
                         'ENSG00000124233.11',
                         'ENSG00000169344.15',
                         'ENSG00000172179.11',
                         'ENSG00000259384.6',
                         'ENSG00000204414.11',
                         'ENSG00000279857.1',
                         'ENSG00000135346.8',
                         'ENSG00000164822.4',
                         'ENSG00000164816.7']
TOP_LOW_VARIANCE_GENES = ['ENSG00000262526.2',
                         'ENSG00000131143.8',
                         'ENSG00000173812.10',
                         'ENSG00000104904.12',
                         'ENSG00000170315.13',
                         'ENSG00000105185.11',
                         'ENSG00000123349.13',
                         'ENSG00000156976.15',
                         'ENSG00000197746.13',
                         'ENSG00000172757.12']

TOP_TISSUES = ['Muscle - Skeletal',
             'Skin - Sun Exposed (Lower leg)',
             'Adipose - Subcutaneous',
             'Lung',
             'Artery - Tibial',
             'Thyroid',
             'Nerve - Tibial',
             'Esophagus - Mucosa',
             'Cells - Transformed fibroblasts',
             'Esophagus - Muscularis',
             'Heart - Left Ventricle',
             'Skin - Not Sun Exposed (Suprapubic)',
             'Artery - Aorta',
             'Adipose - Visceral (Omentum)']
    