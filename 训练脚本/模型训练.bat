@echo off
call E:\anaconda3\Scripts\activate.bat E:\anaconda3\envs\paddle_11.2
cd/d D:\github\Online_Tournament\PaddleDetection-2.5
python tools/train.py -c configs/ppyoloe/ppyoloe_plus_crn_m_80e_coco.yml ^
					--use_vdl=true ^
					--vdl_log_dir=D:\github\Online_Tournament\vdl_log\ppyoloe_plus_crn_m_80e_coco