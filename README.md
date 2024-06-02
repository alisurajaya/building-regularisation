# Building Footprint Regularisation

This repository contains an algorithm for building footprint regularization. Extracted building footprints from imagery often have jagged edges due to the raster-to-vector conversion process. This methods try to refine the building footprint/outline and removed the jagged edges. The input for this algorithm is a GeoJSON file of building footprints extracted from aerial or satellite imagery.  

## Preparation
### Create new virtual environment (venv)
```
    python -m venv <venv_name>
```

### Activate the venv
In powershell.
```
    <venv_name>\Scripts\activate

``` 

Git bash
```
    source <venv_name>/Scripts/activate
```

### Install dependencies inside the activated venv
```
    pip install -r requirements
```

## Running the project
1. Download sample files from https://drive.google.com/drive/folders/1qtLHroTUvnCM2WQc3M-8IdzOxD2V-m_A?usp=sharing
2. Change the file_path and output_geojson variable values to fit your file name and directory. e.g files/renggang
3. Run the program by typing : python <file_name.py>


## Notes
1. all-regulatisation-time-mp.py (using multiprocessing. distribute the data rowly to the each core processor)
2. regularisation-v2 or main.py (using multiprocessing. distribute the data batchly (500 rows) to the core processor)
