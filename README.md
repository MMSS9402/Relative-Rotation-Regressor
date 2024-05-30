# CuTi

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

