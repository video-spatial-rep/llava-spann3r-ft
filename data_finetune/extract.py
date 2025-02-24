import json

# Define input and output file paths.
input_file = "/home/rilyn/scripts/LLaVA-Video-FT/data_finetune/all_qa_pairs_scannet_updated.json"
output_file = "/home/rilyn/scripts/LLaVA-Video-FT/data_finetune/all_qa_pairs_scannet_f.json"

# Load the JSON data.
with open(input_file, "r") as f:
    data = json.load(f)

# Iterate over each item and update the "video" field if present.
for item in data:
    if "video" in item:
        # Replace the '_128f.mp4' substring with '.mp4'
        item["video"] = item["video"].replace("_128f.mp4", ".mp4")

# Write the updated JSON data to the output file.
with open(output_file, "w") as f:
    json.dump(data, f, indent=4)

print("Replacement complete. Updated file saved to", output_file)
