#!/usr/bin/ksh
ARRAY=()
disArray=()
l=0


while read line
do
  ARRAY+=("$line")
done < scenario_name

while read schemeLINE
do
    disArray+=("$schemeLINE")
    ((l=l+1))
done < scenario_name_dis


#Start Different Computation Schemes
  for((i=1;i<=l;i++))
  do
    sleep 1
    echo "start ${disArray[i]} scheme ..."
    awk '{print $8}' ./regular/${ARRAY[i]} >> ./regular/${ARRAY[i]}_reward
    awk '{print $10}' ./regular/${ARRAY[i]} >> ./regular/${ARRAY[i]}_time
    awk '{print $8}' ./distributed/${disArray[i]} >> ./distributed/${disArray[i]}_reward
    awk '{print $10}' ./distributed/${disArray[i]} >> ./distributed/${disArray[i]}_time
  done

  for((i=1;i<=l;i++))
  do
    sleep 1
    echo "start ${disArray[i]} scheme ..."
    cat ./regular/${ARRAY[i]}_reward | colrm 6 >> ./regular/${ARRAY[i]}_reward_good
    #cat ./regular/${ARRAY[i]}_reward | colrm 6 >> ./regular/${ARRAY[i]}_reward_good
    cat ./distributed/${disArray[i]}_reward | colrm 6 >> ./distributed/${disArray[i]}_reward_good
  done
