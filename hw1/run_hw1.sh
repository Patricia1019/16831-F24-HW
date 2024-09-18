# Ant-v2
python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --eval_batch_size 2000 --num_agent_train_steps_per_iter 1500 \
    --video_log_freq -1

# Humanoid-v2
python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Humanoid.pkl \
    --env_name Humanoid-v2 --exp_name bc_humanoid --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
    --eval_batch_size 2000 --num_agent_train_steps_per_iter 1500 \
    --video_log_freq -1

# Walker2d-v2
python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Walker2d.pkl \
    --env_name Walker2d-v2 --exp_name bc_walker2d --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Walker2d-v2.pkl \
    --eval_batch_size 2000 --num_agent_train_steps_per_iter 1500 \
    --video_log_freq -1

# Hopper-v2
python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Hopper.pkl \
    --env_name Hopper-v2 --exp_name bc_hopper --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Hopper-v2.pkl \
    --eval_batch_size 2000 --num_agent_train_steps_per_iter 1500 \
    --video_log_freq -1

# HalfCheetah-v2
python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/HalfCheetah.pkl \
    --env_name HalfCheetah-v2 --exp_name bc_halfCheetah --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_HalfCheetah-v2.pkl \
    --eval_batch_size 2000 --num_agent_train_steps_per_iter 1500 \
    --video_log_freq -1

# Ant-v2 tuning parameters
python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --eval_batch_size 2000 --num_agent_train_steps_per_iter 500 \
    --video_log_freq -1

python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --eval_batch_size 2000 --num_agent_train_steps_per_iter 700 \
    --video_log_freq -1

python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --eval_batch_size 2000 --num_agent_train_steps_per_iter 900 \
    --video_log_freq -1

python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --eval_batch_size 2000 --num_agent_train_steps_per_iter 1100 \
    --video_log_freq -1

python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --eval_batch_size 2000 --num_agent_train_steps_per_iter 1300 \
    --video_log_freq -1

python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --eval_batch_size 2000 --num_agent_train_steps_per_iter 1700 \
    --video_log_freq -1

python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --eval_batch_size 2000 --num_agent_train_steps_per_iter 1900 \
    --video_log_freq -1

python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --eval_batch_size 2000 --num_agent_train_steps_per_iter 2100 \
    --video_log_freq -1

python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --eval_batch_size 2000 --num_agent_train_steps_per_iter 2300 \
    --video_log_freq -1

# dagger
python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name dagger_ant --n_iter 10 \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --eval_batch_size 2000 --num_agent_train_steps_per_iter 1500 \
    --video_log_freq -1 --do_dagger

python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Hopper.pkl \
    --env_name Hopper-v2 --exp_name dagger_hopper --n_iter 10 \
    --expert_data rob831/expert_data/expert_data_Hopper-v2.pkl \
    --eval_batch_size 2000 --num_agent_train_steps_per_iter 1500 \
    --video_log_freq -1 --do_dagger