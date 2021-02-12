python src/compressed_sensing.py --checkpoint-path models/ncsnv2_ffhq/checkpoint_80000.pth --net ncsnv2 --dataset ffhq-69000 --num-input-images 19 --batch-size 18 --ncsnv2-configs-file ./ncsnv2/configs/ffhq.yml --measurement-type superres --noise-std 0 --downsample 32 --model-types langevin --save-images --save-stats --print-stats --checkpoint-iter 1 --gif --gif-iter 20 --gif-dir ncsnv-cs-langevin-gif --cuda --mloss-weight 1.0 --learning-rate 9e-6 --sigma-init 348 --sigma-final 0.01 --L 2311 --T 3
