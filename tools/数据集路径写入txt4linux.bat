@echo
SET dataset=good_v10

call E:\anaconda3\Scripts\activate.bat E:\anaconda3\envs\paddle_11.2
python dataset4linux.py -i D:\github\Online_Tournament\my_dataset\%dataset%

cd/d D:\github\Online_Tournament\tools
cmd