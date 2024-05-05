@echo off
call E:\anaconda3\Scripts\activate.bat E:\anaconda3\envs\paddle_11.2
cd/d D:\github\Online_Tournament\PaddleDetection-2.5

python tools/infer.py ^
    -c configs/yolov3/yolov3_mobilenet_v1_ssld_270e_voc.yml ^
    -o weights=output/yolov3_mobilenet_v1_ssld_270e_voc/best_model_0409.pdparams ^
    --infer_dir=D:\github\Online_Tournament\photo\block_scene ^
    --output_dir=D:\github\Online_Tournament\photo\images_predict