#!/usr/bin/env bash
emb=/users2/yxwang/work/data/embeddings/glove/glove.6B.100d.txt.gz
lmdir=/users2/yxwang/work/data/models
dir=/users8/yqji/NeuroNLP2/data/dm_pas_psd_1000

# 可选参数
type=pas2dm
name="LS+G2G-stage2-pas2dm"
tol_epoch=0 #跳过dev集的验证
pretrain_roberta=none
pretrain_network_path= #中断之后继续训练
form=conllx
source=pas
target=dm
plus=none  # 额外词表

cp -r /users10/zllei/g2g/g2g-zhilin/test/3/pas2dm/first_stage/alphabets /users10/zllei/g2g/g2g-zhilin/test/3/pas2dm/second_stage/

train=$dir/dm_split/train.conllu
dev=$dir/dm_split/dev.conllu
test=$dir/dm_split/test.conllu #交叉验证,没有测试集
lans="en"

main=/users7/zllei/g2g/experiments/sdp_parser.py

seed=666
batch=32
evalbatch=$batch
epoch=1000
patient=20
evalevery=1


lmlr='2e-5'
lr='0.002'
lm=sroberta
lmpath=$lmdir/roberta-base
lm_config=/users7/zllei/g2g/experiments/configs/roberta-base-en-config.json

use_elmo=''
elmo_path=$lmdir/elmo
random_word=''
pretrain_word=''
freeze=''
#freeze=' --freeze'
trim=' --do_trim'
#trim=''
vocab_size=40000

opt=adamw
#sched=exponential
#decay='0.99999'
sched=step
decay='0.75'
dstep=5000
warmup=500
reset=20
beta1='0.9'
#beta2='0.999'
beta2='0.9'
eps='1e-8'
clip='5.0'
l2decay='0'
unk=0
#unk='1.0'
ndigit=''
#ndigit=' --normalize_digits'
losstype=token

posidx=3
mix=' --mix_datasets'
#form=conll

mode=train
#mode=$2

save=/users10/zllei/g2g/g2g-zhilin/test/4/pas2dm
log_file=${save}/log_${mode}_$(date "+%m%d-%H%M").txt



#source /users2/yxwang/work/env/py3.6/bin/activate
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -u $main --mode $mode \
 --config /users7/zllei/g2g/experiments/configs/biaffine_sdp.json --seed $seed \
 --num_epochs $epoch --patient_epochs $patient --batch_size $batch --eval_batch_size $evalbatch \
 --opt $opt --schedule $sched --learning_rate $lr --lr_decay $decay --decay_steps $dstep \
 --beta1 $beta1 --beta2 $beta2 --eps $eps --grad_clip $clip \
 --eval_every $evalevery --noscreen $freeze ${random_word} ${pretrain_word} \
 --loss_type $losstype --warmup_steps $warmup --reset $reset --weight_decay $l2decay --unk_replace $unk \
 --word_embedding glove --word_path $emb --char_embedding random \
 --max_vocab_size ${vocab_size} $trim $ndigit \
 --elmo_path ${elmo_path} ${use_elmo} \
 --pretrained_lm $lm --lm_path $lmpath --lm_lr $lmlr --lm_config $lm_config \
 --punctuation '.' '``' "''" ':' ',' --pos_idx $posidx \
 --format $form \
 --train $train \
 --dev $dev \
 --test $test \
 --plus $plus \
 --name $name \
 --lan_train $lans --lan_dev $lans --lan_test $lans $mix \
 --tol_epoch $tol_epoch \
 --fine_tune \
 --pretrain_network_path $pretrain_network_path \
 --model_path ${save} > $log_file
# --tol_epoch $tol_epoch
#  --fine_tune \
#--pretrain_roberta $pretrain_roberta \
#--pre_epoch \
#  --pretrain_network_path $pretrain_network_path
