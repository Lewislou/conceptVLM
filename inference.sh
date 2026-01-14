torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=1 \
    --master_port=10293 \
    eval/vqa/evaluate_vqa.py --checkpoint "/home/louw/InternVL/internvl_chat/work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora"
	
	
	
	GPUS=1 sh evaluate.sh "/home/louw/InternVL/internvl_chat/work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora"