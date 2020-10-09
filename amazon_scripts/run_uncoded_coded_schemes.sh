#!/usr/bin/ksh
NodeArray=()
SchemeArray=()
ScenarioArray=()
NumAgentArray=()
NumLearnersArray=()
NumVanLDPCpArray=()
NumVanLDPCphoArray=()
NumVanLDPCgammaArray=()

num_node=0
num_scheme=0
max_straggler=1
num_scenario=0
num_LDPC_nodes=0
num_LDPC_learners=0
host_name=""

#Get the number of learners for each scenario in terms of coded scheme
while read learnerLINE
do
  NumLearnersArray+=("$learnerLINE")
done < numLearners

while read VanLDPCpLINE
do
  NumVanLDPCpArray+=("$VanLDPCpLINE")
done < vanLDPC_p

while read VanLDPCphoLINE
do
  NumVanLDPCphoArray+=("$VanLDPCphoLINE")
done < vanLDPC_pho

while read VanLDPCgammaLINE
do
  NumVanLDPCgammaArray+=("$VanLDPCgammaLINE")
done < vanLDPC_gamma



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


for((n=1;n<=num_scenario;n++))
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

      sleep 1
      echo "start centralized scheme with scenario ${ScenarioArray[n]} straggler $j..."
      echo " "
      python3 ../experiments/train.py --scenario simple_spread --num_agents ${NumAgentArray[n]} --num_straggler $j >> centralized_num_straggler_${j}_${ScenarioArray[n]}_num_learners_${NumAgentArray[n]}


      sleep 1
      echo "start uncoded scheme with scenario ${ScenarioArray[n]} straggler $j..."
      echo " "
      mpirun --mca plm_rsh_no_tree_spawn 1 --mca btl_base_warn_component_unused 0  --host $host_name_uncoded\
      python3 ../experiments/maddpg_uncoded.py --scenario simple_spread  --num_agents ${NumAgentArray[n]} --num_straggler $j >> uncoded_num_straggler_${j}_${ScenarioArray[n]}_num_learners_${NumAgentArray[n]}


      for((i=1;i<=num_scheme;i++))
      do

        if (( $i < 4))
            then
            sleep 1
            echo "start ${SchemeArray[i]}  scheme with scenario ${ScenarioArray[n]} straggler $j..."
            echo ""
            mpirun --mca plm_rsh_no_tree_spawn 1 --mca btl_base_warn_component_unused 0  --host $host_name\
            python3 ../experiments/maddpg_coded_scheme.py --scenario simple_spread --num_agents ${NumAgentArray[n]} --num_straggler $j --num_learners ${NumLearnersArray[n]} --scheme\
            ${SchemeArray[i]} >> ${SchemeArray[i]}_num_straggler_${j}_${ScenarioArray[n]}_num_learners_${NumLearnersArray[n]}

          fi

        # if (( $i  == 4 ))
        #     then
        #     echo "start ${SchemeArray[i]}  scheme with scenario ${ScenarioArray[n]} straggler $j..."
        #     echo " "
        #     host_name_LDPC="${NodeArray[1]}"
        #     ((num_LDPC_nodes=${NumVanLDPCpArray[n]}*${NumVanLDPCphoArray[n]}+2))
        #     ((num_LDPC_learners=${NumVanLDPCpArray[n]}*${NumVanLDPCphoArray[n]}))
        #
        #     for((o=2;o<=num_LDPC_nodes;o++))
        #     do
        #       host_name_LDPC="${host_name_LDPC},${NodeArray[o]}"
        #     done
        #
        #     mpirun --mca plm_rsh_no_tree_spawn 1 --mca btl_base_warn_component_unused 0  --host $host_name_LDPC\
        #     python3 ../experiments/maddpg_coded_scheme.py --scenario ${ScenarioArray[n]} --num_straggler $j --num_learners $num_LDPC_learners --scheme VandermondeLDPC\
        #      --vanLDPC_p ${NumVanLDPCpArray[n]}  --vanLDPC_pho ${NumVanLDPCphoArray[n]} --vanLDPC_gamma ${NumVanLDPCgammaArray[n]} >> ${SchemeArray[i]}_num_straggler_${j}_${ScenarioArray[n]}_num_learners_${num_LDPC_learners}
        #
        #     fi

      done

    done
done
