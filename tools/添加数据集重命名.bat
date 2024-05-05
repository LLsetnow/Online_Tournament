@echo off
call E:\anaconda3\Scripts\activate.bat E:\anaconda3\envs\paddle_11.2

python rename_images.py ^
		--input=D:\github\Online_Tournament\my_dataset\18_voc_dataset\choose ^
		--output=D:\github\Online_Tournament\my_dataset\18_voc_dataset\output
