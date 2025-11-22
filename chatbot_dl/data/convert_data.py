import re

input_path = r"C:\codes\chatbot_dl\data\chatbot_dataset.txt"
output_path = r"C:\codes\chatbot_dl\data\chatbot_dataset_converted.txt"

converted_lines = []
current_q = None

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if line.startswith("Q:"):
            current_q = line.replace("Q:", "").strip()

        elif line.startswith("A:") and current_q:
            answer = line.replace("A:", "").strip()
            converted_lines.append(f"{current_q}\t{answer}")
            current_q = None  # Reset for next pair

# Save output
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(converted_lines))

print("✅ All Q&A converted!")
print("📄 Saved as:", output_path)
