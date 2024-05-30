# Relative Rotation Regressor

- This deep learning model estimates the relative rotation between two images.
- The model estimates the relative rotation between two images through vanishing point estimation under the Manhattan assumption.
- Relative Rotation Regressor is an application of the Vanishing Point Detector. If you are interested in the Vanishing Point Detector that detects vanishing points, you can view it at this [github.](https://github.com/MMSS9402/Vanishing-Point-Detector).
- You can view the paper at this [Link.](https://kookmin.dcollection.net/public_resource/pdf/200000737077_20240530151846.pdf)


## installation
```shell
pip install -r requirements.txt
```

## Dataset
MatterPort3D dataset Download [Jin et al.](https://github.com/jinlinyi/SparsePlanes/blob/main/docs/data.md)

## Data Preprocessing
```shell
cd scripts/data_preprocessing
python LSD2.py
```

## train script
```shell
python run.py 
```

## train test script
```shell
python run.py experiment=cuti_run_test.yaml
```

