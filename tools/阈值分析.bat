@echo off
SET dataset=v1
SET model=ppyoloe99255

call E:\anaconda3\Scripts\activate.bat E:\anaconda3\envs\paddle_11.2
python threshold_analysis.py ^
			--model=D:\github\Online_Tournament\model\%model% ^
			--dataset=D:\github\Online_Tournament\my_dataset\%dataset%

cmd
		
