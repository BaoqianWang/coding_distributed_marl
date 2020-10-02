#!/usr/bin/ksh
ARRAY=()

k=0
while read LINE
do
    ARRAY+=("$LINE")
    ((k=k+1))
done < scenario_name



# host_name_uncoded="${ARRAY[1]}"
# for((i=2;i<=k;i++))
# do
#   host_name_uncoded="${host_name_uncoded},${ARRAY[i]}"
# done


#Start Different Computation Schemes
for((j=0;j<=k;j=j+1))
do

  sleep 1
  echo "${ARRAY[j]}"
  python3 ../experiments/train.py --scenario ${ARRAY[j]}

done
