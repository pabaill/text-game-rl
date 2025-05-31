python SAC/train.py --csv {csv}

<!-- learned reward shaping, with and without curriculum -->

python SAC/online_train.py --game_path jericho/z-machine-games-master/jericho-game-suite/zork1.z5 --wandb_proj 224r --wandb_entity cadicamo --learn_reward_shaping True --curriculum_enabled True --pretrain_critic_enabled True

python SAC/online_train.py --game_path jericho/z-machine-games-master/jericho-game-suite/zork1.z5 --wandb_proj 224r --wandb_entity cadicamo --learn_reward_shaping True --curriculum_enabled False --pretrain_critic_enabled True --episodes 101

<!-- copy over checkpoints -->

scp -i 224R_aws_key.pem -r "ubuntu@ec2-54-191-56-70.us-west-2.compute.amazonaws.com:~/text-game-rl/checkpoints/online/nocurr*" ./text-game-rl/checkpoints/online/reward_shaping_curriculum_disabled/

scp -i 224R_aws_key.pem -r ubuntu@ec2-35-95-65-119.us-west-2.compute.amazonaws.com:~/text-game-rl/checkpoints/online/from_beginning_potential_based_reward_shaping ./text-game-rl/checkpoints/online/from_beginning_potential_based_reward_shaping


<!-- learn from beginning with potential based reward shaping -->
python SAC/online_train.py --game_path jericho/z-machine-games-master/jericho-game-suite/zork1.z5 --wandb_proj 224r --wandb_entity cadicamo --random_reset False --max_ep_len 100 --episodes 101