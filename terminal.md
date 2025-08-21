1.PreScale

python precompute_features.py --config_path config/brainmri.yaml

2.Train

python train.py --config_path config/brainmri.yaml --num_support_samples 16
python train.py --config_path config/brainmri.yaml --num_support_samples 8