#!/usr/bin/ksh
filename1='nodeIPaddress'
ARRAY=()
num_nodes=0
while read LINE
do
    ARRAY+=("$LINE")
    ((num_nodes++))
done < $filename1


#Transfer files to worker.
filename1='marl_files'
while read line; do
  for ((i=1;i<=num_nodes;i++))
  do
      scp -i ~/AmazonEC2/.ssh/linux_key_pari.pem  $line ubuntu@${ARRAY[i]}:~/marl_coding/experiments/
      echo "Transfer $i to $line Done "
  done
done < $filename1


for ((i=1;i<=num_nodes;i++))
do
    scp -i ~/AmazonEC2/.ssh/linux_key_pari.pem  ../parameters/marl_parameters.json ubuntu@${ARRAY[i]}:~/marl_coding/parameters/
    echo "Transfer $i to $line Done "
done



for ((i=1;i<=num_nodes;i++))
do
    scp -i ~/AmazonEC2/.ssh/linux_key_pari.pem  ../utilities/learner.py ubuntu@${ARRAY[i]}:~/marl_coding/utilities/
    echo "Transfer $i to $line Done "
done




for ((i=1;i<=num_nodes;i++))
do
    scp -i ~/AmazonEC2/.ssh/linux_key_pari.pem  /home/smile/multiagent-particle-envs/multiagent/scenarios/simple_spread.py ubuntu@${ARRAY[i]}:~/multi-envs/multiagent/scenarios/
    echo "Transfer $i to $line Done "
done





filename2='home_files'
while read line; do
  for ((i=1;i<=num_nodes;i++))
  do
      scp -i ~/AmazonEC2/.ssh/linux_key_pari.pem  $line ubuntu@${ARRAY[i]}:~/marl_coding/amazon_scripts/
      echo "Transfer $i to $line Done "
  done
done < $filename2



#Transfer worker load to each worker
# for ((i=1;i<=num_nodes-1;i++))
# do
#     scp -i ~/AmazonEC2/.ssh/linux_key_pari.pem  A_worker$i.h5 ubuntu@${ARRAY[i+1]}:~
#     echo "Transfer A_worker$i to ${ARRAY[i+1]} Done "
# done
#scp -i ~/AmazonEC2/.ssh/linux_key_pari.pem  A_worker3.h5 ubuntu@${ARRAY[4]}:~

# # Transfer files to master.
# filename2='files_for_master_node'
# while read line; do
# scp -i ~/AmazonEC21224/.ssh/linux_key_pari.pem $line  ubuntu@${ARRAY[1]}:~
# echo "Transfer to $line Done "
# done < $filename2


# #Transfer files to worker.
# filename3='files_for_worker_nodes'
# while read line; do
#   for ((i=2;i<=num_nodes;i++))
#   do
#       scp -i ~/AmazonEC21224/.ssh/linux_key_pari.pem  $line ubuntu@${ARRAY[i]}:~
#       echo "Transfer $i to $line Done "
#   done
# done < $filename3
