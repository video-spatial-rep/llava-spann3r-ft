import json
import random
from collections import defaultdict

input_file = "/home/rilyn/project-files/02-pj-cambrians/LLaVA-Video-FT/data_finetune/all_qa_pairs_vsibench_processed_f.json"
output_file = "/home/rilyn/project-files/02-pj-cambrians/LLaVA-Video-FT/data_finetune/all_qa_pairs_1kpc.json"


with open(input_file, "r") as f:
    data = json.load(f)

# Group questions by their `question_type`
category_dict = defaultdict(list)
for item in data:
    category_dict[item["question_type"]].append(item)

# Sample up to 1000 questions per category
sampled_data = []
for category, questions in category_dict.items():
    num_to_sample = min(len(questions), 1000)  # Sample up to 1000
    sampled_data.extend(random.sample(questions, num_to_sample))  # Sample without replacement

# Save the sampled dataset
with open(output_file, "w") as f:
    json.dump(sampled_data, f, indent=4)

print(f"Sampled dataset saved to {output_file}")
