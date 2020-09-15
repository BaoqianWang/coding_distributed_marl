#!/usr/bin/ksh
ARRAY=()
SchemeArray=()
host_name=""
k=0
l=0
max_straggler=5

while read LINE
do
    ARRAY+=("$LINE")
    ((k=k+1))
done < nodeIPaddress



host_name_uncoded="${ARRAY[1]}"
for((i=2;i<=k;i++))
do
  host_name_uncoded="${host_name_uncoded},${ARRAY[i]}"
done


#Start Different Computation Schemes
for((j=0;j<=max_straggler;j=j+1))
do

  echo "continuous sending"
  sleep 1
  mpirun --mca plm_rsh_no_tree_spawn 1 --mca btl_base_warn_component_unused 0  --host $host_name_uncoded\
   python3 continuous_sending.py

   echo "broadcast"
   sleep 1
   mpirun --mca plm_rsh_no_tree_spawn 1 --mca btl_base_warn_component_unused 0  --host $host_name_uncoded\
    python3 broadcast_test.py

done
