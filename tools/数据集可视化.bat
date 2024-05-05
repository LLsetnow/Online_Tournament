@echo
call E:\anaconda3\Scripts\activate.bat E:\anaconda3\envs\paddle_11.2
python visual_dataset.py ^
		--input D:\github\Online_Tournament\my_dataset\DatasetVocSASU_ForIcarM2023\output_rename ^
		--input_anno D:\github\Online_Tournament\my_dataset\DatasetVocSASU_ForIcarM2023\output_rename ^
		--output D:\github\Online_Tournament\my_dataset\DatasetVocSASU_ForIcarM2023\output_visual ^
		--num 512