@echo off


REM 激活conda环境
call E:\anaconda3\Scripts\activate.bat E:\anaconda3\envs\paddle_11.2

REM 定义数据集和标签
set dataset_folder=D:\github\Online_Tournament\my_dataset
set dataset=v5_1
set label=11


REM 定义标签列表
set label_list={'bomb': 1, 'bridge': 2, 'safety': 3, 'cone': 4, 'crosswalk': 5, 'danger': 6, 'evil': 7, 'block': 8, 'patient': 9, 'prop': 10, 'spy': 11, 'thief': 12, 'tumble': 13}

REM 调用Python解释器执行脚本
python copy_paste.py ^
    --input_dir=%dataset_folder%/%dataset%/images ^
    --json_path=%dataset_folder%/%dataset%/train.json ^
    --output_json=%dataset_folder%/%dataset%/train.json ^
    --muti_obj ^
    --category_id=%label% ^
    --copypaste_ratio=0.20

REM 切换到工具目录
cd /d D:\github\Online_Tournament\tools

REM 打开新的命令提示符窗口
cmd
