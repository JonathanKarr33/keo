# kg_gpt-4ocum_500 with 500 datapoints
python generate_answers.py \
    --question-files ./output/aviation_sensemaking_questions.json \
    --output-file ./output/kg_gpt-4o/answers_phi-4-14b_kg_gpt-4ocum_500.json \
    --kg-path ../kg/output/kg_llm/gpt4o_with_nodes_batches/cum_500/gpt4o_withprevnodes_cum_500_with_entity_mentions_fixed.gml \
    --answer-model microsoft/phi-4 \
    --provider huggingface \
    --API-provider nebius

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_gpt-4o/answers_phi-4-14b_kg_gpt-4ocum_500.json \
    --output-dir ./evaluation_results/answer_phi-4-14b_evaluator_gpt-4o_kg_gpt-4ocum500 \
    --provider openai \
    --evaluation-model gpt-4o
# Generate answers for the questions using vanillation, textchunk RAG, and graph RAG methods
# Note: The sample size is set to 20 for demonstration purposes; ignore for full experiment
mkdir -p ./output/kg_gpt-4o
mkdir -p ./evaluation_results
# kg_gs100
python generate_answers.py \
    --question-files ./output/aviation_sensemaking_questions.json \
    --output-file ./output/answers_mistral-nemo-12b-it_kg_gs100.json \
    --kg-path ../kg/output/knowledge_graph.gml \
    --answer-model mistralai/Mistral-Nemo-Instruct-2407 \
    --provider huggingface \
    --API-provider nebius

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/answers_mistral-nemo-12b-it_kg_gs100.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_gpt-4o_kg_gs100 \
    --provider openai \
    --evaluation-model gpt-4o

# kg_gpt-4o100 with 100 datapoints, #100
python generate_answers.py \
    --question-files ./output/aviation_sensemaking_questions.json \
    --output-file ./output/kg_gpt-4o/answers_mistral-nemo-12b-it_kg_gpt-4o100.json \
    --kg-path ../kg/output/kg_llm/gpt4o_with_nodes_batches/100/gpt4o_withprevnodes_100_with_entity_mentions_fixed.gml \
    --answer-model mistralai/Mistral-Nemo-Instruct-2407 \
    --provider huggingface \
    --API-provider nebius

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_gpt-4o/answers_mistral-nemo-12b-it_kg_gpt-4o100.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_gpt-4o_kg_gpt-4o100 \
    --provider openai \
    --evaluation-model gpt-4o

# kg_gpt-4o300 with 100 datapoints, #300
python generate_answers.py \
    --question-files ./output/aviation_sensemaking_questions.json \
    --output-file ./output/kg_gpt-4o/answers_mistral-nemo-12b-it_kg_gpt-4o300.json \
    --kg-path ../kg/output/kg_llm/gpt4o_with_nodes_batches/300/gpt4o_withprevnodes_300_with_entity_mentions_fixed.gml \
    --answer-model mistralai/Mistral-Nemo-Instruct-2407 \
    --provider huggingface \
    --API-provider nebius

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_gpt-4o/answers_mistral-nemo-12b-it_kg_gpt-4o300.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_gpt-4o_kg_gpt-4o300 \
    --provider openai \
    --evaluation-model gpt-4o

# kg_gpt-4ocum_200 with 200 datapoints
python generate_answers.py \
    --question-files ./output/aviation_sensemaking_questions.json \
    --output-file ./output/kg_gpt-4o/answers_mistral-nemo-12b-it_kg_gpt-4ocum_200.json \
    --kg-path ../kg/output/kg_llm/gpt4o_with_nodes_batches/cum_200/gpt4o_withprevnodes_cum_200_with_entity_mentions_fixed.gml \
    --answer-model mistralai/Mistral-Nemo-Instruct-2407 \
    --provider huggingface \
    --API-provider nebius

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_gpt-4o/answers_mistral-nemo-12b-it_kg_gpt-4ocum_200.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_gpt-4o_kg_gpt-4ocum200 \
    --provider openai \
    --evaluation-model gpt-4o

# kg_gpt-4ocum_300 with 300 datapoints
python generate_answers.py \
    --question-files ./output/aviation_sensemaking_questions.json \
    --output-file ./output/kg_gpt-4o/answers_mistral-nemo-12b-it_kg_gpt-4ocum_300.json \
    --kg-path ../kg/output/kg_llm/gpt4o_with_nodes_batches/cum_300/gpt4o_withprevnodes_cum_300_with_entity_mentions_fixed.gml \
    --answer-model mistralai/Mistral-Nemo-Instruct-2407 \
    --provider huggingface \
    --API-provider nebius

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_gpt-4o/answers_mistral-nemo-12b-it_kg_gpt-4ocum_300.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_gpt-4o_kg_gpt-4ocum300 \
    --provider openai \
    --evaluation-model gpt-4o

# kg_gpt-4ocum_400 with 400 datapoints
python generate_answers.py \
    --question-files ./output/aviation_sensemaking_questions.json \
    --output-file ./output/kg_gpt-4o/answers_mistral-nemo-12b-it_kg_gpt-4ocum_400.json \
    --kg-path ../kg/output/kg_llm/gpt4o_with_nodes_batches/cum_400/gpt4o_withprevnodes_cum_400_with_entity_mentions_fixed.gml \
    --answer-model mistralai/Mistral-Nemo-Instruct-2407 \
    --provider huggingface \
    --API-provider nebius

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_gpt-4o/answers_mistral-nemo-12b-it_kg_gpt-4ocum_400.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_gpt-4o_kg_gpt-4ocum400 \
    --provider openai \
    --evaluation-model gpt-4o

# kg_gpt-4ocum_500 with 500 datapoints
python generate_answers.py \
    --question-files ./output/aviation_sensemaking_questions.json \
    --output-file ./output/kg_gpt-4o/answers_mistral-nemo-12b-it_kg_gpt-4ocum_500.json \
    --kg-path ../kg/output/kg_llm/gpt4o_with_nodes_batches/cum_500/gpt4o_withprevnodes_cum_500_with_entity_mentions_fixed.gml \
    --answer-model mistralai/Mistral-Nemo-Instruct-2407 \
    --provider huggingface \
    --API-provider nebius

python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/kg_gpt-4o/answers_mistral-nemo-12b-it_kg_gpt-4ocum_500.json \
    --output-dir ./evaluation_results/answer_mistral-nemo-12b-it_evaluator_gpt-4o_kg_gpt-4ocum500 \
    --provider openai \
    --evaluation-model gpt-4o