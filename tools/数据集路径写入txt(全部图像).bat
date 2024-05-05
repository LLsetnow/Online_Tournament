@echo
SET dataset=tumble_4
call E:\anaconda3\Scripts\activate.bat E:\anaconda3\envs\paddle_11.2
python dataset_all2txt.py -i D:\github\Online_Tournament\my_dataset\%dataset%
				 