python run_parallel.py \
    --server kaist \
    --group_name final_test \
    --exp_name simbaV2_no_hb \
    --agent_config simbaV2 \
    --env_type hb_locomotion \
    --device_ids 0 1 2 3 4 5 6 7 \
    --num_seeds 5 \
    --num_exp_per_device 4 \
    --overrides recording_per_interaction_step=9999999