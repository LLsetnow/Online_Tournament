@echo off
call E:\anaconda3\Scripts\activate.bat E:\anaconda3\envs\paddle_11.2
cd/d D:\github\Online_Tournament\PaddleDetection-2.5
python tools/eval.py -c configs/ppyoloe/ppyoloe_plus_crn_m_80e_coco.yml ^
					-o weights=output/ppyoloe_plus_crn_m_80e_coco/ap857