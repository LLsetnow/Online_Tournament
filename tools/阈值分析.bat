@echo off
SET dataset=v1_3
SET model=souhu_99321_v15_ep105_v1_3

call E:\anaconda3\Scripts\activate.bat E:\anaconda3\envs\paddle_11.2
python threshold_analysis.py ^
			--model=D:\github\Online_Tournament\model\%model% ^
			--dataset=D:\github\Online_Tournament\dataset\%dataset%

cmd
		
