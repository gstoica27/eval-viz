#!/bin/bash
source /weka/prior-default/jianingz/home/anaconda3/etc/profile.d/conda.sh
conda activate base
cd /weka/prior-default/jianingz/home/project/_GenTraj/viz_website
python -c "from app import app; app.run(host='0.0.0.0', port=8888, debug=False, threaded=True)"
