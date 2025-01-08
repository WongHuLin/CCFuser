cd /root/mix_moe/ae/
bash /root/mix_moe/mix_moe/examples/test.sh > fig10_data.txt 2>& 1
rm ./output_data/moe_layer_pcie.csv
echo "name,num_expert,total_time,forward_time,backward_time\n">> ./output_data/moe_layer_pcie.csv
cat  fig10_data.txt  | grep Summary  | awk  '{print $1 "," $2 "," $8 "," $12 "," $16}' >> ./output_data/moe_layer_pcie.csv
rm fig10_data.txt 