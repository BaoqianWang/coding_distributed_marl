#!/usr/bin/ksh
ARRAY=()
SchemeArray=()
host_name=""
k=8
l=0
max_straggler=4




while read schemeLINE
do
    SchemeArray+=("$schemeLINE")
    ((l=l+1))
done < computationScheme



#Start Different Computation Schemes
for((j=0;j<=max_straggler;j=j+2))
do
  for((i=1;i<=l;i++))
  do
    sleep 1
    echo "start ${SchemeArray[i]} scheme ..."
    mpirun -np 10 python3 main_unified_computation_scheme.py --scheme ${SchemeArray[i]} --num_straggler $j >> ./results/${SchemeArray[i]}_num_learner_${k}_num_straggler_${j}_local
  done

  sleep 1
  echo "start Uncoded scheme ..."
  mpirun -np 7 python3 main_uncoded.py --num_straggler $j >> ./results/uncoded_num_learner_${k}_num_straggler_${j}_local

done
