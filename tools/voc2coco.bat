@echo
SET dataset=good_v2
call E:\anaconda3\Scripts\activate.bat E:\anaconda3\envs\paddle_11.2
cd/d D:\github\Online_Tournament\PaddleDetection-2.5
python tools/x2coco.py ^
        --dataset_type voc ^
        --voc_anno_dir D:\github\Online_Tournament\my_dataset\%dataset% ^
        --voc_anno_list D:\github\Online_Tournament\my_dataset\%dataset%\val.txt ^
        --voc_label_list D:\github\Online_Tournament\my_dataset\%dataset%\label_list.txt ^
        --voc_out_name D:\github\Online_Tournament\my_dataset\%dataset%\val.json


python tools/x2coco.py ^
        --dataset_type voc ^
        --voc_anno_dir D:\github\Online_Tournament\my_dataset\%dataset% ^
        --voc_anno_list D:\github\Online_Tournament\my_dataset\%dataset%\train.txt ^
        --voc_label_list D:\github\Online_Tournament\my_dataset\%dataset%\label_list.txt ^
        --voc_out_name D:\github\Online_Tournament\my_dataset\%dataset%\train.json

cd/d D:\github\Online_Tournament\tools
cmd