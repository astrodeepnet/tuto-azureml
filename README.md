ML on Azure for AstroDeep
=========================

## Notes

Here are a few notes on the services offered by Azure and how to use them

- [Azure ML](notes/azure_ml.md)
- [Azure storage](notes/azure_storage.md)

## Notebooks

Before running the notebooks, make sure you computer is [properly set up](#Local-setup)



## Local setup

All informations can be found [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment)

### download workspace configuration JSON file

As specified [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#workspace), a `config.json` with dedicated workspace information can be downloaded from the Azure portal.  

It must then be place in the work directory.

### create a virtual env with `conda`

```bash
conda update conda
conda env create -f environment.yml
```

The virtual environment is called `azure-astrodeep`

### activate the environment
```bash
conda activate azure-astrodeep
```
