## Extending rep

- all ipynb or other analysis scripts should go to the appropriate folder / sub-folder under `src`. If you work on `chipseq`, 
make a new folder in `src/chipseq` (e.g. `src/chipseq/my_folder`)
- put useful functions specific to your particular experiment to `rep/exp/<exp name>/<my module>.py`
- put functions/classes useful across many experiments / data types to the appropriate modules under rep/ (e.g. models to models.py, layers to layers.py etc).
- try to work on the master branch to avoid diverging branches and be able to immediately share code. 
  - make sure you don't break other people's code in `rep/`
- use pep8 formatting (setup style higlighting / auto-fixing in your text editor, e.g. https://pypi.org/project/autopep8/)
- document functions using google style docstrings. [example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- Keep the data outside of the git repository.
