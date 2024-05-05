@echo off
call E:\anaconda3\Scripts\activate.bat E:\anaconda3\envs\paddle_11.2
cd/d D:\github\Online_Tournament\submmison
python predict.py data.txt result.json
		
