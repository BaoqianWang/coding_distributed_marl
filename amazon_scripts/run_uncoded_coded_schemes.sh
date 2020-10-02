#!/usr/bin/ksh
ARRAY=()
SchemeArray=()
host_name=""
k=0
l=0
max_straggler=2
num_agents=3


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


((num_uncoded_nodes=num_agents+2))
((num_coded_learners=k-2))

host_name_uncoded="${ARRAY[1]}"
for((i=2;i<=num_uncoded_learners;i++))
do
  host_name_uncoded="${host_name_uncoded},${ARRAY[i]}"
done


#Start Different Computation Schemes
for((i=1;i<=l;i++))
do
  for((j=0;j<=max_straggler;j++))
  do

    sleep 1
    echo "start ${SchemeArray[i]}  scheme ..."
    mpirun --mca plm_rsh_no_tree_spawn 1 --mca btl_base_warn_component_unused 0  --host $host_name\
    python3 ../experiments/maddpg_coded_scheme.py --scenario simple_spread --num_straggler $j --num_learners $num_coded_learners --scheme\
    ${SchemeArray[i]} >> ${SchemeArray[i]}_num_straggler_${j}_simple_spread_num_learners_${num_coded_learners}


    sleep 1
    echo "start uncoded scheme ..."
    mpirun --mca plm_rsh_no_tree_spawn 1 --mca btl_base_warn_component_unused 0  --host $host_name_uncoded\
    python3 ../experiments/maddpg_uncoded.py --scenario_name simple_spread --num_straggler $j >> uncoded_num_straggler_${j}_simple_spread_num_learners_${num_agents}

  done
done
