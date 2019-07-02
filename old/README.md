# REP

RNA Expression prediction

- [gdocs](https://drive.google.com/drive/folders/1uGBIctJlo-uGSfgI2SA5mH7Q4NYU7tDj)
- [website](https://i12g-gagneurweb.in.tum.de/project/rep/)

## Folder structure

- rep: python package  
  - cli/ - CLI commands
  - exp/ - experiment-specific code
- src: contains script, ipython notebooks (i.e. other code, not part of the python package)
- data:
  - raw: raw data, not modified by any script
  - processed: output of the different commands

## Setup

### 1. Install rep

```bash
# Create 'rep' conda environment
conda env create -f conda-env.yml
source activate rep

# Install all necessary dependencies
conda install -c conda-forge implicit

# Install rep
pip install -e .
```

## Extending

See [docs/extending.md](docs/extending.md)
