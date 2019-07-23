"""
Convert all ipynb to html
"""
import os
from glob import glob
from pathlib import Path

# config
# exclude = ['src/chipseq/', 'src/chipexo']
exclude = []
TARGET_DIR = 'data/www/code/'
# --------

def startswith_any(x, exclude_list):
    for l in exclude_list:
        if x.startswith(l):
            return True
    return False


rule all:
    input:
        [TARGET_DIR / Path(x).with_suffix(".html")  # ipynb -> html in TARGET_DIR
         for x in glob('notebooks/**/*ipynb', recursive=True)  # all ipynb
         if not startswith_any(x, exclude)]  # exclude


rule compile_ipynb:
    input:
        f = "{path}.ipynb"
    output:
        f = os.path.join(TARGET_DIR, "{path}.html")
    shell:
        """
        jupyter nbconvert --to html '{input.f}' --output-dir ./ --output '{TARGET_DIR}/{wildcards.path}'
        """
