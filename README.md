# Overcoming Catastrophic Forgetting beyond Continual Learning: Balanced Training for Neural Machine Translation
This repository contains the source code for our ACL 2022 paper Overcoming Catastrophic Forgetting beyond Continual Learning: Balanced Training for Neural Machine Translation [pdf](https://arxiv.org/abs/). This code is implemented based on the open-source toolkit [fairseq](https://github.com/pytorch/fairseq).

# Requirements
This system has been tested in the following environment.

+ Python version = 3.8
+ Pytorch version = 1.7

# Replicate the TED results
## Pre-processing
We use the tokenized TED dataset released by [VOLT](https://github.com/Jingjing-NLP/VOLT), which can be downloaded from [here](https://drive.google.com/drive/folders/1FNH7cXFYWWnUdH2LyUFFRYmaWYJJveKy) and pre-processed into subword units by [prepare-ted-bilingual.sh](https://github.com/Jingjing-NLP/VOLT/blob/master/examples/prepare-ted-bilingual.sh).

We provide the pre-processed TED En-Es dataset in this repository. First, process the data into the fairseq format.
```
TEXT=./data
python preprocess.py --source-lang en --target-lang es \
        --trainpref $TEXT/es-en.train \
        --validpref $TEXT/es-en.valid \
        --testpref $TEXT/es-en.test \
        --destdir data-bin/tedbpe10kenes \
        --nwordssrc 10240 --joined-dictionary  --workers 16
```
## Training
To train the Transformer baseline, run the following command.
```
data_dir=data-bin/tedbpe10kenes
save_dir=output/enes_base

python train.py $data_dir \
    --fp16 --dropout 0.3  --save-dir $save_dir \
    --arch transformer_wmt_en_de --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0007 --min-lr 1e-09 \
    --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --update-freq 1\
    --no-progress-bar --log-format json --log-interval 100 --save-interval-updates 1000 \
    --max-update 18000 --keep-interval-updates 10 --no-epoch-checkpoints
    
python scripts/average_checkpoints.py --inputs $save_dir \
 --num-update-checkpoints 5  --output $save_dir/average-model.pt
 ```
To train the COKD model, run the following command.
```
data_dir=data-bin/tedbpe10kenes
save_dir=output/enes_cokd

python train.py $data_dir \
    --fp16 --dropout 0.2 --kd-alpha 0.95 --num-teachers 1 --save-dir $save_dir \
    --arch transformer_wmt_en_de --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0007 --min-lr 1e-09 \
    --weight-decay 0.0 --criterion cokd_loss --label-smoothing 0.1 --max-tokens 4096 --update-freq 1\
    --no-progress-bar --log-format json --log-interval 100 --save-interval-updates 1000 \
    --max-update 18000 --keep-interval-updates 10 --no-epoch-checkpoints
    
python scripts/average_checkpoints.py --inputs $save_dir \
 --num-update-checkpoints 5  --output $save_dir/average-model.pt
 ```
The above commands assume 8 GPUs on the machine. When the number of GPUs is different, adapt --update-freq to make sure that the batch size is 32K. 
## Inference
Run the following command for inference.
```
python generate.py data-bin/tedbpe10kenes  --path output/enes_cokd/average-model.pt --gen-subset test --beam 5 --batch-size 100 --remove-bpe --lenpen 1 > out
# because fairseq's output is unordered, we need to recover its order
grep ^H out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.es
sed -r 's/(@@ )|(@@ ?$)//g' data/es-en.test.es > ref.es
perl multi-bleu.perl ref.es < pred.es
```
The expected BLEU scores are 40.86 for the Transformer baseline and 42.50 for the COKD model.

# Citation

Please cite as:

``` bibtex
@inproceedings{cokd,
  title = {Overcoming Catastrophic Forgetting beyond Continual Learning: Balanced Training for Neural Machine Translation},
  author= {Chenze Shao and
               Yang Feng},
  booktitle = {Proceedings of ACL 2022},
  year = {2022},
}
```
