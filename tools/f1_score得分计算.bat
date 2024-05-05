@echo
SET model_name=ppyoloe99255
SET dataset=v1
call E:\anaconda3\Scripts\activate.bat E:\anaconda3\envs\paddle_11.2
python f1_score.py ^
	--model=D:\github\Online_Tournament\model\%model_name% ^
	--dataset=D:\github\Online_Tournament\my_dataset\\%dataset% ^
		--output=D:\github\Online_Tournament\model\%model_name%\analyse\F1_Score.json 

cd/d D:\github\Online_Tournament\tools
cmd
