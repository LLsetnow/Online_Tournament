@echo off
call E:\anaconda3\Scripts\activate.bat E:\anaconda3\envs\paddle_11.2
cd/d D:\github\Online_Tournament\PaddleDetection-2.5

python tools/infer.py ^
    -c configs\yolov3\yolov3_mobilenet_v1_ssld_200e_voc.yml ^
    -o weights=D:\github\Online_Tournament\model\offline_good_v6\best_model ^
    --infer_dir=D:\github\Online_Tournament\my_dataset\test\images ^
    --output_dir=D:\github\Online_Tournament\my_dataset\test\output

cmd