@echo
SET model_name=model97178_yolov3_mobilenet_v1_ssld_270e_voc
SET dataset=tumble

call E:\anaconda3\Scripts\activate.bat E:\anaconda3\envs\paddle_11.2
python add_dataset.py ^
	--model=D:\github\Online_Tournament\model\%model_name% ^
	--images=D:\github\Online_Tournament\my_dataset\%dataset%\all.txt ^
	--output=D:\github\Online_Tournament\my_dataset\%dataset%\annotations

cd/d D:\github\Online_Tournament\tools
