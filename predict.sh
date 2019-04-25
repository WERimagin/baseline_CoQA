python scripts/gen_pipeline_data.py \
 --data_file data/coqa-dev-normal.json \
 --output_file1 data/coqa.dev.pipeline.json \
 --output_file2 data/seq2seq-dev-pipeline

python rc/main.py \
--testset data/coqa.dev.pipeline.json \
--n_history 0 \
--pretrained pipeline_models --pretrained_model pipeline_models/params-interro-modify-0423.saved \
--cuda_id 3

python scripts/gen_pipeline_for_seq2seq.py \
--data_file data/coqa.dev.pipeline.json \
--output_file pipeline_models/pipeline-seq2seq-src.txt \
--pred_file pipeline_models/predictions.json

python seq2seq/translate.py \
-model pipeline_models/seq2seq_copy_acc_85.04_ppl_2.19_e14.pt \
-src pipeline_models/pipeline-seq2seq-src.txt \
-output pipeline_models/pred.txt \
-replace_unk -dynamic_dict -gpu 3

python scripts/gen_seq2seq_output.py \
--data_file data/coqa-dev-normal.json \
--pred_file pipeline_models/pred.txt \
--output_file pipeline_models/pipeline.prediction-normal.json

python evaluate-v1.0.py \
--data-file data/coqa-dev-normal.json \
--pred-file pipeline_models/pipeline.prediction-normal.json





python3 scripts/gen_pipeline_data.py \
 --data_file data/coqa-dev-modify.json \
 --output_file1 data/coqa.dev.pipeline.json \
 --output_file2 data/seq2seq-dev-pipeline

python rc/main.py \
--testset data/coqa.dev.pipeline.json \
--n_history 0 \
--pretrained pipeline_models --pretrained_model pipeline_models/params-normal-0424.saved \
--cuda_id 3

python scripts/gen_pipeline_for_seq2seq.py \
--data_file data/coqa.dev.pipeline.json \
--output_file pipeline_models/pipeline-seq2seq-src.txt \
--pred_file pipeline_models/predictions.json

python seq2seq/translate.py \
-model pipeline_models/seq2seq_copy_acc_84.75_ppl_2.26_e21.pt \
-src pipeline_models/pipeline-seq2seq-src.txt \
-output pipeline_models/pred.txt \
-replace_unk -dynamic_dict -gpu 3

python scripts/gen_seq2seq_output.py \
--data_file data/coqa-dev-modify.json \
--pred_file pipeline_models/pred.txt \
--output_file pipeline_models/pipeline.prediction-modify.json

python evaluate-v1.0.py \
--data-file data/coqa-dev-modify.json \
--pred-file pipeline_models/pipeline.prediction-modify.json
