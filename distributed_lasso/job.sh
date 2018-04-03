	#!/bin/sh
### General o:wq

### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J Tunning_Lasso
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=10GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 30GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 01:30 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Error_%J.err 

# Load modules needed by myapplication.x

module load numpy/1.13.1-python-3.6.2-openblas-0.2.20
module load scipy/0.19.1-python-3.6.2
#module load time
#module load multiprocessing
#module load json
#module load collections


# Run my program
python3 tunning.py > output.out
#python3 test.py  > output.out
