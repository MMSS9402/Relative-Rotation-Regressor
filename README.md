# Relative Rotation Regressor
- 두 장의 이미지 사이의 상대적인 회전을 구하는 딥러닝 모델입니다.
- 이 딥러닝 모델은 맨하탄 가정 하에서 소실점 추정을 통해 두 이미지 간의 상대적인 회전을 추정합니다.


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

