export CUDA_VISIBLE_DEVICES=0

lm_eval --model hf \
    --model_args pretrained=openai-community/gpt2 \
    --tasks hellaswag,wikitext,lambada_openai \
    --device cuda:0 \
    --batch_size 512

lm_eval --model hf \
    --model_args pretrained=openai-community/gpt2-xl \
    --tasks hellaswag,wikitext,lambada_openai \
    --device cuda:0 \
    --batch_size 64