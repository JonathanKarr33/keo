# Generate answers for the questions using vanillation, textchunk RAG, and graph RAG methods
# Note: The sample size is set to 20 for demonstration purposes; ignore for full experiment
mkdir -p ./output/kg_phi4mini-it
mkdir -p ./evaluation_results

# kg_phi4mini-it100 with 100 datapoints, #100
python generate_answers.py \
    --question-files ./output/aviation_sensemaking_questions.json \
    --output-file ./output/kg_phi4mini-it/answers_mistral-nemo-12b-it_kg_phi4mini-it100.json \
    --kg-path ../kg/output/kg_llm/phi4mini_instruct_with_nodes_batches/100/phi4mini_instruct_withprevnodes_100_fixed.gml \
    --answer-model mistralai/Mistral-Nemo-Instruct-2407 \
    --provider huggingface \
    --API-provider nebius

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_mistral-nemo-12b-it_kg_phi4mini-it100.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_gpt-4o_kg_phi4mini-it100 \
    --provider openai \
    --evaluation-model gpt-4o

# kg_phi4mini-it300 with 100 datapoints, #300
python generate_answers.py \
    --question-files ./output/aviation_sensemaking_questions.json \
    --output-file ./output/kg_phi4mini-it/answers_mistral-nemo-12b-it_kg_phi4mini-it300.json \
    --kg-path ../kg/output/kg_llm/phi4mini_instruct_with_nodes_batches/300/phi4mini_instruct_withprevnodes_300_fixed.gml \
    --answer-model mistralai/Mistral-Nemo-Instruct-2407 \
    --provider huggingface \
    --API-provider nebius

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_mistral-nemo-12b-it_kg_phi4mini-it300.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_gpt-4o_kg_phi4mini-it300 \
    --provider openai \
    --evaluation-model gpt-4o

# kg_phi4mini-itcum_200 with 200 datapoints
python generate_answers.py \
    --question-files ./output/aviation_sensemaking_questions.json \
    --output-file ./output/kg_phi4mini-it/answers_mistral-nemo-12b-it_kg_phi4mini-itcum_200.json \
    --kg-path ../kg/output/kg_llm/phi4mini_instruct_with_nodes_batches/cum_200/phi4mini_instruct_withprevnodes_cum_200_fixed.gml \
    --answer-model mistralai/Mistral-Nemo-Instruct-2407 \
    --provider huggingface \
    --API-provider nebius

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_mistral-nemo-12b-it_kg_phi4mini-itcum_200.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_gpt-4o_kg_phi4mini-itcum200 \
    --provider openai \
    --evaluation-model gpt-4o

# kg_phi4mini-itcum_300 with 300 datapoints
python generate_answers.py \
    --question-files ./output/aviation_sensemaking_questions.json \
    --output-file ./output/kg_phi4mini-it/answers_mistral-nemo-12b-it_kg_phi4mini-itcum_300.json \
    --kg-path ../kg/output/kg_llm/phi4mini_instruct_with_nodes_batches/cum_300/phi4mini_instruct_withprevnodes_cum_300_fixed.gml \
    --answer-model mistralai/Mistral-Nemo-Instruct-2407 \
    --provider huggingface \
    --API-provider nebius

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_mistral-nemo-12b-it_kg_phi4mini-itcum_300.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_gpt-4o_kg_phi4mini-itcum300 \
    --provider openai \
    --evaluation-model gpt-4o

# kg_phi4mini-itcum_400 with 400 datapoints
python generate_answers.py \
    --question-files ./output/aviation_sensemaking_questions.json \
    --output-file ./output/kg_phi4mini-it/answers_mistral-nemo-12b-it_kg_phi4mini-itcum_400.json \
    --kg-path ../kg/output/kg_llm/phi4mini_instruct_with_nodes_batches/cum_400/phi4mini_instruct_withprevnodes_cum_400_fixed.gml \
    --answer-model mistralai/Mistral-Nemo-Instruct-2407 \
    --provider huggingface \
    --API-provider nebius

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_mistral-nemo-12b-it_kg_phi4mini-itcum_400.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_gpt-4o_kg_phi4mini-itcum400 \
    --provider openai \
    --evaluation-model gpt-4o

# kg_phi4mini-itcum_500 with 500 datapoints
python generate_answers.py \
    --question-files ./output/aviation_sensemaking_questions.json \
    --output-file ./output/kg_phi4mini-it/answers_mistral-nemo-12b-it_kg_phi4mini-itcum_500.json \
    --kg-path ../kg/output/kg_llm/phi4mini_instruct_with_nodes_batches/cum_500/phi4mini_instruct_withprevnodes_cum_500_fixed.gml \
    --answer-model mistralai/Mistral-Nemo-Instruct-2407 \
    --provider huggingface \
    --API-provider nebius

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_phi4mini-it/answers_mistral-nemo-12b-it_kg_phi4mini-itcum_500.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_gpt-4o_kg_phi4mini-itcum500 \
    --provider openai \
    --evaluation-model gpt-4o