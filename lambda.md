Installation
===

```bash
sudo pip3 install virtualenv

sudo apt-get install python3-dev

cd XLM
virtualenv -p /usr/bin/python3.6 venv-xlm
. venv-xlm/bin/activate

pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl

pip3 install torchvision

./install-tools.sh

git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir .
cd ..
```

Dataset
===
```bash
wget https://dl.fbaipublicfiles.com/XLM/codes_enfr
wget https://dl.fbaipublicfiles.com/XLM/vocab_enfr

./get-data-nmt.sh --src en --tgt fr --reload_codes codes_enfr --reload_vocab vocab_enfr

wget -c https://dl.fbaipublicfiles.com/XLM/mlm_enfr_1024.pth

```

Train
===
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
--exp_name unsupMT_enfr \
--dump_path ./dumped/ \
--reload_model 'mlm_enfr_1024.pth,mlm_enfr_1024.pth' \
--data_path ./data/processed/en-fr/ \
--lgs 'en-fr' \
--ae_steps 'en,fr' \
--bt_steps 'en-fr-en,fr-en-fr' \
--word_shuffle 3 \
--word_dropout 0.1 \
--word_blank 0.1 \
--lambda_ae '0:1,100000:0.1,300000:0' \
--encoder_only false \
--emb_dim 1024 \
--n_layers 6 \
--n_heads 8 \
--dropout 0.1 \
--attention_dropout 0.1 \
--gelu_activation true \
--tokens_per_batch 2000 \
--batch_size 32 \
--bptt 256 \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
--epoch_size 200000 \
--eval_bleu true \
--stopping_criterion 'valid_en-fr_mt_bleu,10' \
--validation_metrics 'valid_en-fr_mt_bleu'                    
```


Memory Requirement

| Batch Size | Memory  |
|---|---|
| tokens_per_batch=500 | 8GB |
| tokens_per_batch=1000 | 11GB  |
| tokens_per_batch=2000 | 24GB |
| tokens_per_batch=4000 | 24GB |
| tokens_per_batch=8000 | |

Throughput (words/sec) 

|   | 2060  | 2070  | 2080  |  1080 Ti | 2080 Ti | TitanRTX | Quadro RTX 6000 | V100 | Quadro RTX 8000 |
|---|---|---|---|---|---|---|---|---|---|
| tokens_per_batch=500 | OOM | 1009.9 | 1212.38 | 1182.95 | 1349.98 | 1405.14 | 1425.27 | | 1427.12 |
| tokens_per_batch=1000 | OOM | OOM | OOM | 1824.04 | 2025.35 | 2200.63 | 2183.65 | | 2071.28 |
| tokens_per_batch=2000 | OOM | OOM | OOM | OOM | OOM | 3132.97 | 3075.95 | | 2829.28 |
| tokens_per_batch=4000 | OOM | OOM | OOM | OOM | OOM | 3850.34 | 3725.16 | | 3559.68 |
| tokens_per_batch=8000 | OOM | OOM | OOM | OOM | OOM | OOM | OOM | | 3734.71 |
