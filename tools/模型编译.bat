@echo
cd/d E:
docker start ppnc2.0
docker exec -it ppnc2.0 /bin/bash
export PPNC_HOME=/usr/local/ppnc
source /usr/local/ppnc/scripts/activate_env.sh
cd home/edgeboard/workspace/compiler
python3 compile.py ./config.json
CMD