python run_parallel.py \
    --server kaist \
    --group_name final_test \
    --exp_name simbav2_bc \
    --config_name offline_rl \
    --agent_config simbaV2_bc \
    --env_type d4rl_mujoco \
    --device_ids 0 1 2 3 4 5 6 7 \
    --num_seeds 5 \
    --num_exp_per_device 1 \