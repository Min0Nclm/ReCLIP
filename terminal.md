1.PreScale

python precompute_features.py --data_source data/brainmri
python precompute_features.py --data_source data/chexpert

2.Train

python train.py --config_path config/brainmri.yaml --num_support_samples 1
python train.py --config_path config/busi.yaml --num_support_samples 1
python train.py --config_path config/chexpert.yaml --num_support_samples 1