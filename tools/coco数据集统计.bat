@echo
SET dataset=good_v9
call E:\anaconda3\Scripts\activate.bat E:\anaconda3\envs\paddle_11.2
python dataset_count4coco.py -i D:\github\Online_Tournament\my_dataset\%dataset%

cd/d D:\github\Online_Tournament\tools
cmd