@echo
call E:\anaconda3\Scripts\activate.bat E:\anaconda3\envs\paddle_11.2
python data2predict.py -i D:\github\Online_Tournament\my_dataset\v2\train.txt ^
					    -o D:\github\Online_Tournament\submmison\my_submmison\train.txt
python data2predict.py -i D:\github\Online_Tournament\my_dataset\v2\val.txt ^
					    -o D:\github\Online_Tournament\submmison\my_submmison\val.txt
cd/d D:\github\Online_Tournament\tools
