### Create file to be executed ###
import os
import shutil

system_id = 'car'
system_id_short = 'C'

n_runs = 10

w_S = 1e-3
info = '{}'.format(1)

offset = 0 if w_S ==0 else 100

seed_list = [29556, 5280, 739, 92, 10, 7298, 14, 264, 22135, 342]

shutil.copy('Template.sh', '{}{}.sh'.format(system_id_short,info))
with open('{}{}.sh'.format(system_id_short,info), 'a') as f:
    for i in range(1,1 + n_runs):
        f.write("nohup python3 -u main.py --system-id='{}' --test-n={}  --seed={} --w-S={} > out/{}{}.txt &\n".format(system_id,offset+i,seed_list[i-1], w_S, system_id_short,offset+i))
    f.write("\necho 'Running scripts in parallel'\n")
    f.write("echo 'PID of this script is:' $$\n")
    f.write("wait\n")
    f.write("echo 'Script done running'\n")

os.makedirs('out', exist_ok=True)