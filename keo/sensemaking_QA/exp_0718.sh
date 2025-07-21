# Generate answers for the questions using vanillation, textchunk RAG, and graph RAG methods
# Note: The sample size is set to 20 for demonstration purposes; ignore for full experiment
mkdir -p ./output/kg_gpt-4o
mkdir -p ./evaluation_results
# # kg_gs100
# python run_evaluation.py \
#     --questions-file ./output/aviation_sensemaking_questions.json \
#     --answers-file ./output/answers_gemma-3-27b-it_kg_gs100.json \
#     --output-dir ./evaluation_results/answer_gemma-3-27b-it_evaluator_llama-3.3-70b_kg_gs100 \
#     --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
#     --provider huggingface \
#     --API-provider nebius

# # kg_gpt-4o100 with 100 datapoints, #100
# python run_evaluation.py \
#     --questions-file ./output/aviation_sensemaking_questions.json \
#     --answers-file ./output/kg_gpt-4o/answers_gemma-3-27b-it_kg_gpt-4o100.json \
#     --output-dir ./evaluation_results/answer_gemma-3-27b-it_evaluator_llama-3.3-70b_kg_gpt-4o100 \
#     --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
#     --provider huggingface \
#     --API-provider nebius

# # kg_gpt-4o300 with 100 datapoints, #300
# python run_evaluation.py \
#     --questions-file ./output/aviation_sensemaking_questions.json \
#     --answers-file ./output/kg_gpt-4o/answers_gemma-3-27b-it_kg_gpt-4o300.json \
#     --output-dir ./evaluation_results/answer_gemma-3-27b-it_evaluator_llama-3.3-70b_kg_gpt-4o300 \
#     --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
#     --provider huggingface \
#     --API-provider nebius

# # kg_gpt-4ocum_200 with 200 datapoints
# python run_evaluation.py \
#     --questions-file ./output/aviation_sensemaking_questions.json \
#     --answers-file ./output/kg_gpt-4o/answers_gemma-3-27b-it_kg_gpt-4ocum_200.json \
#     --output-dir ./evaluation_results/answer_gemma-3-27b-it_evaluator_llama-3.3-70b_kg_gpt-4ocum200 \
#     --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
#     --provider huggingface \
#     --API-provider nebius

# # kg_gpt-4ocum_300 with 300 datapoints
# python run_evaluation.py \
#     --questions-file ./output/aviation_sensemaking_questions.json \
#     --answers-file ./output/kg_gpt-4o/answers_gemma-3-27b-it_kg_gpt-4ocum_300.json \
#     --output-dir ./evaluation_results/answer_gemma-3-27b-it_evaluator_llama-3.3-70b_kg_gpt-4ocum300 \
#     --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
#     --provider huggingface \
#     --API-provider nebius

# # kg_gpt-4ocum_400 with 400 datapoints
# python run_evaluation.py \
#     --questions-file ./output/aviation_sensemaking_questions.json \
#     --answers-file ./output/kg_gpt-4o/answers_gemma-3-27b-it_kg_gpt-4ocum_400.json \
#     --output-dir ./evaluation_results/answer_gemma-3-27b-it_evaluator_llama-3.3-70b_kg_gpt-4ocum400 \
#     --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
#     --provider huggingface \
#     --API-provider nebius

# # kg_gpt-4ocum_500 with 500 datapoints
# python run_evaluation.py \
#     --questions-file ./output/aviation_sensemaking_questions.json \
#     --answers-file ./output/kg_gpt-4o/answers_gemma-3-27b-it_kg_gpt-4ocum_500.json \
#     --output-dir ./evaluation_results/answer_gemma-3-27b-it_evaluator_llama-3.3-70b_kg_gpt-4ocum500 \
#     --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
#     --provider huggingface \
#     --API-provider nebius

# mkdir -p ./output/kg_gpt-4o
# mkdir -p ./evaluation_results
# # kg_gs100
# python run_evaluation.py \
#     --questions-file ./output/aviation_sensemaking_questions.json \
#     --answers-file ./output/answers_phi-4-14b_kg_gs100.json \
#     --output-dir ./evaluation_results/answer_phi-4-14b_evaluator_llama-3.3-70b_kg_gs100 \
#     --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
#     --provider huggingface \
#     --API-provider nebius

# # kg_gpt-4o100 with 100 datapoints, #100

# python run_evaluation.py \
#     --questions-file ./output/aviation_sensemaking_questions.json \
#     --answers-file ./output/kg_gpt-4o/answers_phi-4-14b_kg_gpt-4o100.json \
#     --output-dir ./evaluation_results/answer_phi-4-14b_evaluator_llama-3.3-70b_kg_gpt-4o100 \
#     --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
#     --provider huggingface \
#     --API-provider nebius

# # kg_gpt-4o300 with 100 datapoints, #300

# python run_evaluation.py \
#     --questions-file ./output/aviation_sensemaking_questions.json \
#     --answers-file ./output/kg_gpt-4o/answers_phi-4-14b_kg_gpt-4o300.json \
#     --output-dir ./evaluation_results/answer_phi-4-14b_evaluator_llama-3.3-70b_kg_gpt-4o300 \
#     --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
#     --provider huggingface \
#     --API-provider nebius

# # kg_gpt-4ocum_200 with 200 datapoints

# python run_evaluation.py \
#     --questions-file ./output/aviation_sensemaking_questions.json \
#     --answers-file ./output/kg_gpt-4o/answers_phi-4-14b_kg_gpt-4ocum_200.json \
#     --output-dir ./evaluation_results/answer_phi-4-14b_evaluator_llama-3.3-70b_kg_gpt-4ocum200 \
#     --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
#     --provider huggingface \
#     --API-provider nebius

# # kg_gpt-4ocum_300 with 300 datapoints

# python run_evaluation.py \
#     --questions-file ./output/aviation_sensemaking_questions.json \
#     --answers-file ./output/kg_gpt-4o/answers_phi-4-14b_kg_gpt-4ocum_300.json \
#     --output-dir ./evaluation_results/answer_phi-4-14b_evaluator_llama-3.3-70b_kg_gpt-4ocum300 \
#     --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
#     --provider huggingface \
#     --API-provider nebius

# # kg_gpt-4ocum_400 with 400 datapoints

# python run_evaluation.py \
#     --questions-file ./output/aviation_sensemaking_questions.json \
#     --answers-file ./output/kg_gpt-4o/answers_phi-4-14b_kg_gpt-4ocum_400.json \
#     --output-dir ./evaluation_results/answer_phi-4-14b_evaluator_llama-3.3-70b_kg_gpt-4ocum400 \
#     --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
#     --provider huggingface \
#     --API-provider nebius

# kg_gpt-4ocum_500 with 500 datapoints

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_gpt-4o/answers_phi-4-14b_kg_gpt-4ocum_500.json \
    --output-dir ./evaluation_results/answer_phi-4-14b_evaluator_llama-3.3-70b_kg_gpt-4ocum500 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

mkdir -p ./output/kg_gpt-4o
mkdir -p ./evaluation_results
# kg_gs100

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/answers_mistral-nemo-12b-it_kg_gs100.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_llama-3.3-70b_kg_gs100 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

# kg_gpt-4o100 with 100 datapoints, #100

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_gpt-4o/answers_mistral-nemo-12b-it_kg_gpt-4o100.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_llama-3.3-70b_kg_gpt-4o100 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

# kg_gpt-4o300 with 100 datapoints, #300

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_gpt-4o/answers_mistral-nemo-12b-it_kg_gpt-4o300.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_llama-3.3-70b_kg_gpt-4o300 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

# kg_gpt-4ocum_200 with 200 datapoints

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_gpt-4o/answers_mistral-nemo-12b-it_kg_gpt-4ocum_200.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_llama-3.3-70b_kg_gpt-4ocum200 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

# kg_gpt-4ocum_300 with 300 datapoints

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_gpt-4o/answers_mistral-nemo-12b-it_kg_gpt-4ocum_300.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_llama-3.3-70b_kg_gpt-4ocum300 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

# kg_gpt-4ocum_400 with 400 datapoints

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_gpt-4o/answers_mistral-nemo-12b-it_kg_gpt-4ocum_400.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_llama-3.3-70b_kg_gpt-4ocum400 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

# kg_gpt-4ocum_500 with 500 datapoints

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_gpt-4o/answers_mistral-nemo-12b-it_kg_gpt-4ocum_500.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_llama-3.3-70b_kg_gpt-4ocum500 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

mkdir -p ./output/kg_phi4mini-it
mkdir -p ./evaluation_results

# kg_phi4mini-it100 with 100 datapoints, #100

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_gemma-3-27b-it_kg_phi4mini-it100.json \
    --output-dir ./evaluation_results/answer_gemma-3-27b-it_evaluator_llama-3.3-70b_kg_phi4mini-it100 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

# kg_phi4mini-it300 with 100 datapoints, #300

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_gemma-3-27b-it_kg_phi4mini-it300.json \
    --output-dir ./evaluation_results/answer_gemma-3-27b-it_evaluator_llama-3.3-70b_kg_phi4mini-it300 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

# kg_phi4mini-itcum_200 with 200 datapoints

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_gemma-3-27b-it_kg_phi4mini-itcum_200.json \
    --output-dir ./evaluation_results/answer_gemma-3-27b-it_evaluator_llama-3.3-70b_kg_phi4mini-itcum200 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

# kg_phi4mini-itcum_300 with 300 datapoints

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_gemma-3-27b-it_kg_phi4mini-itcum_300.json \
    --output-dir ./evaluation_results/answer_gemma-3-27b-it_evaluator_llama-3.3-70b_kg_phi4mini-itcum300 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

# kg_phi4mini-itcum_400 with 400 datapoints

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_gemma-3-27b-it_kg_phi4mini-itcum_400.json \
    --output-dir ./evaluation_results/answer_gemma-3-27b-it_evaluator_llama-3.3-70b_kg_phi4mini-itcum400 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

# kg_phi4mini-itcum_500 with 500 datapoints

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_gemma-3-27b-it_kg_phi4mini-itcum_500.json \
    --output-dir ./evaluation_results/answer_gemma-3-27b-it_evaluator_llama-3.3-70b_kg_phi4mini-itcum500 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

mkdir -p ./output/kg_phi4mini-it
mkdir -p ./evaluation_results

# kg_phi4mini-it100 with 100 datapoints, #100

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_phi-4-14b_kg_phi4mini-it100.json \
    --output-dir ./evaluation_results/answer_phi-4-14b_evaluator_llama-3.3-70b_kg_phi4mini-it100 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

# kg_phi4mini-it300 with 100 datapoints, #300

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_phi-4-14b_kg_phi4mini-it300.json \
    --output-dir ./evaluation_results/answer_phi-4-14b_evaluator_llama-3.3-70b_kg_phi4mini-it300 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

# kg_phi4mini-itcum_200 with 200 datapoints

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_phi-4-14b_kg_phi4mini-itcum_200.json \
    --output-dir ./evaluation_results/answer_phi-4-14b_evaluator_llama-3.3-70b_kg_phi4mini-itcum200 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

# kg_phi4mini-itcum_300 with 300 datapoints

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_phi-4-14b_kg_phi4mini-itcum_300.json \
    --output-dir ./evaluation_results/answer_phi-4-14b_evaluator_llama-3.3-70b_kg_phi4mini-itcum300 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

# kg_phi4mini-itcum_400 with 400 datapoints

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_phi-4-14b_kg_phi4mini-itcum_400.json \
    --output-dir ./evaluation_results/answer_phi-4-14b_evaluator_llama-3.3-70b_kg_phi4mini-itcum400 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

# kg_phi4mini-itcum_500 with 500 datapoints

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_phi-4-14b_kg_phi4mini-itcum_500.json \
    --output-dir ./evaluation_results/answer_phi-4-14b_evaluator_llama-3.3-70b_kg_phi4mini-itcum500 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

mkdir -p ./output/kg_phi4mini-it
mkdir -p ./evaluation_results

# kg_phi4mini-it100 with 100 datapoints, #100

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_mistral-nemo-12b-it_kg_phi4mini-it100.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_llama-3.3-70b_kg_phi4mini-it100 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

# kg_phi4mini-it300 with 100 datapoints, #300

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_mistral-nemo-12b-it_kg_phi4mini-it300.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_llama-3.3-70b_kg_phi4mini-it300 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

# kg_phi4mini-itcum_200 with 200 datapoints

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_mistral-nemo-12b-it_kg_phi4mini-itcum_200.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_llama-3.3-70b_kg_phi4mini-itcum200 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

# kg_phi4mini-itcum_300 with 300 datapoints

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_mistral-nemo-12b-it_kg_phi4mini-itcum_300.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_llama-3.3-70b_kg_phi4mini-itcum300 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

# kg_phi4mini-itcum_400 with 400 datapoints

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_mistral-nemo-12b-it_kg_phi4mini-itcum_400.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_llama-3.3-70b_kg_phi4mini-itcum400 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius

# kg_phi4mini-itcum_500 with 500 datapoints

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_mistral-nemo-12b-it_kg_phi4mini-itcum_500.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_llama-3.3-70b_kg_phi4mini-itcum500 \
    --evaluation-model meta-llama/Llama-3.3-70B-Instruct \
    --provider huggingface \
    --API-provider nebius