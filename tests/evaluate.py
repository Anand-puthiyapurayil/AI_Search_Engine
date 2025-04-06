from sentence_transformers import SentenceTransformer, util

# Load the pre-trained model
base_model = SentenceTransformer("all-MiniLM-L6-v2")  # Replace with your base model name

# Evaluation data
evaluation_data = [
    {"anchor": "Heavy-Duty Drill Press", 
     "positive": "Industrial-grade drill press for metalworking", 
     "negative": "Bulk Conveyor Belt"},
    {"anchor": "Industrial Welding Service", 
     "positive": "On-site welding for heavy machinery", 
     "negative": "Heavy Equipment Repair Service"},
]

# Evaluate with cosine similarity
results = []
for data in evaluation_data:
    anchor_embedding = base_model.encode(data["anchor"], convert_to_tensor=True)
    positive_embedding = base_model.encode(data["positive"], convert_to_tensor=True)
    negative_embedding = base_model.encode(data["negative"], convert_to_tensor=True)

    positive_score = util.pytorch_cos_sim(anchor_embedding, positive_embedding).item()
    negative_score = util.pytorch_cos_sim(anchor_embedding, negative_embedding).item()

    results.append({"positive_score": positive_score, "negative_score": negative_score})

print("Base Model Results:", results)

# Load the fine-tuned model
fine_tuned_model = SentenceTransformer("output/sbert_finetuned")

# Evaluate with the same process
fine_tuned_results = []
for data in evaluation_data:
    anchor_embedding = fine_tuned_model.encode(data["anchor"], convert_to_tensor=True)
    positive_embedding = fine_tuned_model.encode(data["positive"], convert_to_tensor=True)
    negative_embedding = fine_tuned_model.encode(data["negative"], convert_to_tensor=True)

    positive_score = util.pytorch_cos_sim(anchor_embedding, positive_embedding).item()
    negative_score = util.pytorch_cos_sim(anchor_embedding, negative_embedding).item()

    fine_tuned_results.append({"positive_score": positive_score, "negative_score": negative_score})

print("Fine-Tuned Model Results:", fine_tuned_results)

