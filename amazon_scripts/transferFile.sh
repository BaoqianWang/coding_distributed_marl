#!/bin/sh
# pass the ssh public key of host to ec2 instances


filename='nodeIPaddress'
while read line; do
echo $line

scp -i ~/AmazonEC2/.ssh/linux_key_pari.pem $1 ubuntu@$line:~/marl_coding/utilities/

echo "Transfer $1 to $line Done !"
done < $filename
