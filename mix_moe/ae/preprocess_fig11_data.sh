rm ./output_data/e2e_model.csv
echo "name,MOE-GPT,MOE-BERT,MOE-Transformer-xl" > ./output_data/e2e_model.csv
(echo -n "fastermoe," && cat output_data/fig11_data.txt | grep Summary | grep fastermoe | awk '{print $9}' | tr "\n" ','  | sed 's/,$//'  && echo "") >> ./output_data/e2e_model.csv
(echo -n "mix_moe," && cat output_data/fig11_data.txt | grep Summary | grep mix_moe | awk '{print $9}' | tr "\n" ','  | sed 's/,$//'  && echo "") >> ./output_data/e2e_model.csv
(echo -n "naive," && cat output_data/fig11_data.txt | grep Summary | grep naive | awk '{print $9}' | tr "\n" ','  | sed 's/,$//'  && echo "") >> ./output_data/e2e_model.csv
