@echo off
SET dataset=v1_3
SET model=souhu_99321_v15_ep105_v1_3

call E:\anaconda3\Scripts\activate.bat E:\anaconda3\envs\paddle_11.2

python threshold_count.py ^
    --model=D:\github\Online_Tournament\model\%model% ^
    --dataset=D:\github\Online_Tournament\dataset\%dataset% ^
    --output=D:\github\Online_Tournament\model\%model%\analyse\threshold.json ^
    --start_threshold=0.0 ^
    --end_threshold=1.0 ^
    --step=0.01

cmd