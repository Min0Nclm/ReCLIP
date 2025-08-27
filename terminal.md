1.PreScale

python precompute_features.py --dataset chexpert
python precompute_features.py --dataset chexpert
python precompute_features.py --dataset chexpert

2.Train

python train.py --config_path config/brainmri.yaml --num_support_samples 1
python train.py --config_path config/busi.yaml --num_support_samples 1
python train.py --config_path config/chexpert.yaml --num_support_samples 1