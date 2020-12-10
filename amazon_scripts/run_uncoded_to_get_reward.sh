#!/usr/bin/ksh
NodeArray=()
SchemeArray=()
ScenarioArray=()
NumAgentArray=()
NumLearnersArray=()
NumVanLDPCpArray=()
NumVanLDPCphoArray=()
NumVanLDPCgammaArray=()
num_agents=8

host_name=""

#Get the number of learners for each scenario in terms of coded scheme
while read learnerLINE
do
  NumLearnersArray+=("$learnerLINE")
done < numLearners


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




((num_uncoded_nodes=num_agents+2))
host_name_uncoded="${NodeArray[1]}"

for((i=2;i<=num_uncoded_nodes;i++))
do
  host_name_uncoded="${host_name_uncoded},${NodeArray[i]}"
done



sleep 1
echo "start uncoded scheme with  straggler $j..."
echo " "
mpirun --mca plm_rsh_no_tree_spawn 1 --mca btl_base_warn_component_unused 0  --host $host_name_uncoded\
python3 ../experiments/maddpg_uncoded.py --scenario simple_push  --num_agents $num_agents --num_straggler 0 >> uncoded_simple_push_num_agents_${num_agents}
