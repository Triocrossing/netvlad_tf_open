#!/bin/bash
#patch to be aware of "module" inside a job
#OAR -l {host ='igrida-abacus.irisa.fr'}/gpu_device=1,walltime=2:00:0
#OAR -O /srv/tempdd/xwang/logNetvladSPYAML.%jobid%.output
#OAR -E /srv/tempdd/xwang/logNetvladSPYAML.%jobid%.error

# cd /temp_dd/igrida-fs1/xwang/RobotSeason/overcast-reference/ 
cd /srv/tempdd/xwang/RobotCarSeason/sun
zip -r css_sun.zip css