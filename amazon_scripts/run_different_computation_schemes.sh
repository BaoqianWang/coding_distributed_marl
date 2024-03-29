#!/usr/bin/ksh
ARRAY=()
SchemeArray=()
host_name=""
k=0
l=0
max_straggler=0

while read LINE
do
    ARRAY+=("$LINE")
    ((k=k+1))
done < nodeIPaddress


while read schemeLINE
do
    SchemeArray+=("$schemeLINE")
    ((l=l+1))
done < computationScheme


host_name="${ARRAY[1]}"
for((i=2;i<=k;i++))
do
  host_name="${host_name},${ARRAY[i]}"
done


host_name_uncoded="${ARRAY[1]}"
for((i=2;i<=k;i++))
do
  host_name_uncoded="${host_name_uncoded},${ARRAY[i]}"
done


#Start Different Computation Schemes
for((j=0;j<=max_straggler;j=j+1))
do

  sleep 1
  echo "start multiple scheme ..."
  mpirun --mca plm_rsh_no_tree_spawn 1 --mca btl_base_warn_component_unused 0  --host $host_name_uncoded\
  python3 ../experiments/main_uncoded_multiple_training.py --scenario_name simple_spread


  sleep 1
  echo "start multiple broadcast scheme ..."
  mpirun --mca plm_rsh_no_tree_spawn 1 --mca btl_base_warn_component_unused 0  --host $host_name_uncoded\
  python3 ../experiments/main_uncoded_multiple_training_broadcast.py --scenario_name simple_spread

done
