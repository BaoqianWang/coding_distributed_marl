#!/usr/bin/ksh
ARRAY=()
ARRAYNL=()
k=0
while read LINE
do
    ARRAY+=("$LINE")
    ((k=k+1))
done < scenario_name

l=0
while read LINE1
do
  ARRAYNL+=("$LINE1")
  ((l=l+1))
done < uncoded_num_learner


# host_name_uncoded="${ARRAY[1]}"
# for((i=2;i<=k;i++))
# do
#   host_name_uncoded="${host_name_uncoded},${ARRAY[i]}"
# done


#Start Different Computation Schemes
for((j=1;j<=k;j=j+1))
do

  sleep 1
  echo "${ARRAY[j]}"
  ((num=${ARRAYNL[j]}+2))
  echo $num
  mpirun -n $num python3 ../experiments/maddpg_uncoded.py --scenario ${ARRAY[j]}

done
