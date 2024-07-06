@echo off
call E:\anaconda3\Scripts\activate.bat E:\anaconda3\envs\paddle_11.2
cd /d D:\github\Online_Tournament\PaddleDetection-2.5

python tools/export_model.py -c configs/ppyoloe/ppyoloe_plus_crn_m_80e_coco.yml ^
              -o weights=D:\github\Online_Tournament\PaddleDetection-2.5\output\ppyoloe_plus_crn_m_80e_coco\model_final ^
              TestReader.fuse_normalize=true

