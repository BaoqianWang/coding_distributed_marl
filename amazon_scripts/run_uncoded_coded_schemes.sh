#!/usr/bin/ksh
NodeArray=()
SchemeArray=()
ScenarioArray=()
NumAgentArray=()
NumLearnersArray=()


num_node=0
num_scheme=0
max_straggler=3
num_scenario=0
host_name=""

#Get the number of learners for each scenario in terms of coded scheme
while read learnerLINE
do
  NumLearnersArray+=("$learnerLINE")
done < numLearners


# Get the scenario name
while read scenarioLINE
do
    ScenarioArray+=("$scenarioLINE")
    ((num_scenario=num_scenario+1))
done < scenarioName


# Get the number of agents for each scenario (The number of uncoded learners is equal to the number of agents)
while read numAgentLINE
do
    NumAgentArray+=("$numAgentLINE")
done < numAgents


# Get the node ip address names
while read LINE
do
    NodeArray+=("$LINE")
done < nodeIPaddress

# Get the coded scheme names
while read schemeLINE
do
    SchemeArray+=("$schemeLINE")
    ((num_scheme=num_scheme+1))
done < computationScheme


for((n=1;i<=num_scenario;n++))
do
    ((num_node=${NumLearnersArray[n]}+2))
    host_name="${NodeArray[1]}"
    for((i=2;i<=num_node;i++))
    do
      host_name="${host_name},${NodeArray[i]}"
    done


    ((num_uncoded_nodes=${NumAgentArray[n]}+2))
    host_name_uncoded="${NodeArray[1]}"

    for((i=2;i<=num_uncoded_nodes;i++))
    do
      host_name_uncoded="${host_name_uncoded},${NodeArray[i]}"
    done


    #Start Different Computation Schemes
    for((j=0;j<=max_straggler;j++))
    do
      for((i=1;i<=num_scheme;i++))
      do

        sleep 1
        echo "start ${SchemeArray[i]}  scheme with scenario ${ScenarioArray[n]} straggler $j..."
        mpirun --mca plm_rsh_no_tree_spawn 1 --mca btl_base_warn_component_unused 0  --host $host_name\
        python3 ../experiments/maddpg_coded_scheme.py --scenario ${ScenarioArray[n]} --num_straggler $j --num_learners ${NumLearnersArray[n]} --scheme\
        ${SchemeArray[i]} >> ${SchemeArray[i]}_num_straggler_${j}_${ScenarioArray[n]}_num_learners_${num_coded_learners}

      done

      sleep 1
      echo ""
      echo "start uncoded scheme with scenario ${ScenarioArray[n]} straggler $j..."
      mpirun --mca plm_rsh_no_tree_spawn 1 --mca btl_base_warn_component_unused 0  --host $host_name_uncoded\
      python3 ../experiments/maddpg_uncoded.py --scenario ${ScenarioArray[n]}  --num_straggler $j >> uncoded_num_straggler_${j}_${ScenarioArray[n]}_num_learners_${num_agents}

    done
done
