python3 scripts/gen_pipeline_data.py \
 --data_file data/coqa-dev-noninterro.json \
 --output_file1 data/coqa.dev.pipeline.json \
 --output_file2 data/seq2seq-dev-pipeline

python rc/main.py \
--testset data/coqa.dev.pipeline.json \
--n_history 0 \
--pretrained pipeline_models --cuda_id 3

python scripts/gen_pipeline_for_seq2seq.py \
--data_file data/coqa.dev.pipeline.json \
--output_file pipeline_models/pipeline-seq2seq-src.txt \
--pred_file pipeline_models/predictions.json

python seq2seq/translate.py \
-model pipeline_models/seq2seq_copy_acc_84.73_ppl_2.27_e40.pt \
-src pipeline_models/pipeline-seq2seq-src.txt \
-output pipeline_models/pred.txt \
-replace_unk -verbose -gpu 3

python scripts/gen_seq2seq_output.py \
--data_file data/coqa-dev-noninterro.json \
--pred_file pipeline_models/pred.txt \
--output_file pipeline_models/pipeline.prediction-noninterro.json
