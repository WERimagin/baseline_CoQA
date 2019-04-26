#!/bin/sh

python scripts/gen_pipeline_data.py \
 --data_file data/coqa-dev-$1.json \
 --output_file1 data/coqa.dev.pipeline.json \
 --output_file2 data/seq2seq-dev-pipeline

python rc/main.py \
--testset data/coqa.dev.pipeline.json \
--n_history 0 \
--pretrained pipeline_models --pretrained_model pipeline_models/params-interro-beam2-0425.saved \
--cuda_id $2

python scripts/gen_pipeline_for_seq2seq.py \
--data_file data/coqa.dev.pipeline.json \
--output_file pipeline_models/pipeline-seq2seq-src.txt \
--pred_file pipeline_models/predictions.json

python seq2seq/translate.py \
-model pipeline_models/seq2seq_copy_acc_84.71_ppl_2.18_e25.pt \
-src pipeline_models/pipeline-seq2seq-src.txt \
-output pipeline_models/pred.txt \
-replace_unk -dynamic_dict -gpu $2

python scripts/gen_seq2seq_output.py \
--data_file data/coqa-dev-$1.json \
--pred_file pipeline_models/pred.txt \
--output_file pipeline_models/pipeline.prediction-$1.json

<< comment

python evaluate-v1.0.py \
--data-file data/coqa-dev-interro-beam2.json \
--pred-file pipeline_models/pipeline.prediction-interro-beam2.json

comment
