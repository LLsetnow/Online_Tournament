@echo
call E:\anaconda3\Scripts\activate.bat E:\anaconda3\envs\paddle_11.2
python f1_score_core.py ^
	--model=D:\github\Online_Tournament\model\model97178_yolov3_mobilenet_v1_ssld_270e_voc ^
	--dataset=D:\github\Online_Tournament\my_dataset\\v2 ^
		--output=D:\github\Online_Tournament\model\model97178_yolov3_mobilenet_v1_ssld_270e_voc\analyse\F1_Score.json 

cd/d D:\github\Online_Tournament\tools
