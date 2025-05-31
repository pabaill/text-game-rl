python SAC/train.py --csv {csv}

<!-- learned reward shaping, with and without curriculum -->

python SAC/online_train.py --game_path jericho/z-machine-games-master/jericho-game-suite/zork1.z5 --wandb_proj 224r --wandb_entity cadicamo --learn_reward_shaping True --curriculum_enabled True --pretrain_critic_enabled True

python SAC/online_train.py --game_path jericho/z-machine-games-master/jericho-game-suite/zork1.z5 --wandb_proj 224r --wandb_entity cadicamo --learn_reward_shaping True --curriculum_enabled False --pretrain_critic_enabled True --episodes 101

<!-- copy over checkpoints -->

scp -i 224R_aws_key.pem -r "ubuntu@ec2-54-191-56-70.us-west-2.compute.amazonaws.com:~/text-game-rl/checkpoints/online/nocurr*" ./text-game-rl/checkpoints/online/reward_shaping_curriculum_disabled/
