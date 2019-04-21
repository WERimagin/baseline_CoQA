python3 scripts/gen_pipeline_data.py \
--data_file data/coqa-train-modify.json \
--output_file1 data/coqa.train.pipeline.json \
--output_file2 data/seq2seq-train-pipeline

python3 scripts/gen_pipeline_data.py \
 --data_file data/coqa-dev-modify.json \
 --output_file1 data/coqa.dev.pipeline.json \
 --output_file2 data/seq2seq-dev-pipeline

 python3 seq2seq/preprocess.py \
  -train_src data/seq2seq-train-pipeline-src.txt \
  -train_tgt data/seq2seq-train-pipeline-tgt.txt \
  -valid_src data/seq2seq-dev-pipeline-src.txt \
  -valid_tgt data/seq2seq-dev-pipeline-tgt.txt \
  -save_data data/seq2seq-pipeline \
  -lower -dynamic_dict -src_seq_length 10000

PYTHONPATH=seq2seq \
python seq2seq/tools/embeddings_to_torch.py \
 -emb_file_enc ~/data/glove.840B.300d.txt \
 -emb_file_dec ~/data/glove.840B.300d.txt \
 -dict_file data/seq2seq-pipeline.vocab.pt \
 -output_file data/seq2seq-pipeline.embed \
 -enc_dec enc

 PYTHONPATH=seq2seq \
 python seq2seq/tools/embeddings_to_torch.py \
  -emb_file_enc ~/data/glove.840B.300d.txt \
  -emb_file_dec ~/data/glove.840B.300d.txt \
  -dict_file data/seq2seq-pipeline.vocab.pt \
  -output_file data/seq2seq-pipeline.embed \
  -enc_dec dec
