# Randomly sample 5*100 datapoints from the aviation dataset for KG construction
# python sample_aviation_data.py

# Generate questions for sensemaking (from OMIn) and actionable (from MaintNet) questions
# Note: The questions are generated using all MaintNet datasets and OMIn datasets - 5*100 randomly sampled datapoints - GS datapoints
# You may modify the dataset used in generate_questions.py
# python generate_questions.py \
#     --output-file ./output/aviation_sensemaking_questions.json \
#     --question-model gpt-4o

# Generate answers for the questions using vanillation, textchunk RAG, and graph RAG methods
# Note: The sample size is set to 20 for demonstration purposes; ignore for full experiment
# python generate_answers.py \
#     --question-files ./output/aviation_sensemaking_questions.json \
#     --output-file ./output/answers_gemma-3-27b-it.json \
#     --kg-path ../kg/output/knowledge_graph.gml \
#     --answer-model google/gemma-3-27b-it \
#     --provider huggingface \
#     --API-provider nebius

# Evaluate the generated answers using three-way evaluation (direct evaluation, pairwise standard comparison, and three-way NLP metrics for actionable questions)
# python run_evaluation.py \
#     --questions-file ./output/aviation_sensemaking_questions.json \
#     --answers-file ./output/answers_phi-4_sample20.json \
#     --output-dir ./evaluation_results/answer_phi-4_sample20_evaluator_gpt-4o \
#     --provider openai \
#     --evaluation-model gpt-4o