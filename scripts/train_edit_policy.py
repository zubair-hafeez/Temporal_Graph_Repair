"""Train a small graph-edit policy; no target-test records are used."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tgrepair.action_features import OPERATIONS
from tgrepair.neural_policy import GraphEditPolicy
from tgrepair.provenance import snapshot_sources
from tgrepair.data import sha256


def load_collection(name):
    folder=ROOT/'data/edit_learning'/name
    manifest=json.loads((folder/'manifest.json').read_text())
    return [torch.load(folder/r['file'],map_location='cpu',weights_only=True) for r in manifest['records']],manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',required=True);p.add_argument('--validation',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--representation',choices=['temporal','static','mlp','sage'],default='temporal')
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cuda:1');p.add_argument('--max-actions',type=int,default=2048)
    args=p.parse_args();destination=ROOT/'reports'/f'{args.output}.json'
    if destination.exists():raise FileExistsError(destination)
    snapshot=snapshot_sources(ROOT);start=time.monotonic()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    # Explicit device allocation fails visibly when CUDA is not permitted.
    probe=torch.zeros(1,device=args.device)
    train,train_manifest=load_collection(args.train);validation,validation_manifest=load_collection(args.validation)
    if {r['group'] for r in train_manifest['records']} & {r['group'] for r in validation_manifest['records']}:
        raise ValueError('Execution group leakage')
    all_x=torch.cat([r['observation']['action_features'] for r in train])
    mean=all_x.mean(0);scale=all_x.std(0).clamp_min(.1)
    for row in train+validation:
        row['observation']['action_features']=(row['observation']['action_features']-mean)/scale
    model=GraphEditPolicy(representation=args.representation).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=.0001)
    report={'status':'development_neural_training','config':vars(args),'source_snapshot':snapshot,
            'train_source_snapshot':train_manifest['source_snapshot'],'validation_source_snapshot':validation_manifest['source_snapshot'],
            'parameters':sum(p.numel() for p in model.parameters()),'torch':torch.__version__,
            'optimized_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'device':torch.cuda.get_device_name(probe.device) if probe.is_cuda else 'cpu','epochs':[]}
    counts=np.zeros((3,2))
    for row in train:
        for op in range(3):
            y=row['labels'][row['observation']['operations']==op]
            counts[op]+=[len(y)-y.sum().item(),y.sum().item()]
    positive_weight=torch.tensor(np.clip(counts[:,0]/np.maximum(counts[:,1],1),1,30),dtype=torch.float32,device=args.device)
    report['training_class_counts']=counts.tolist();report['positive_weights']=positive_weight.cpu().tolist()
    best=-1.;rng=np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        model.train();losses=[]
        for i in rng.permutation(len(train)):
            row=train[i];obs=row['observation'];labels=row['labels']
            selected=np.arange(len(labels))
            if len(selected)>args.max_actions:
                positives=np.flatnonzero(labels.numpy());negatives=np.flatnonzero(labels.numpy()==0)
                # Bound positive selection too; no true action is injected into retrieval.
                positives=rng.choice(positives,min(len(positives),args.max_actions//2),replace=False)
                negatives=rng.choice(negatives,min(len(negatives),args.max_actions-len(positives)),replace=False)
                selected=np.concatenate([positives,negatives])
            data={k:v.to(args.device) for k,v in obs.items()}
            for k in ['action_features','action_nodes','operations']:data[k]=data[k][selected]
            y=labels[selected].to(args.device)
            if not len(y):continue
            optimizer.zero_grad(set_to_none=True)
            logits=model(data)
            losses_raw=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')
            weights=torch.where(y>0,positive_weight[data['operations']],1.)
            loss=(losses_raw*weights).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(loss.item())
        scheduler.step();model.eval();predictions=[];labels=[];operations=[]
        with torch.no_grad():
            for row in validation:
                data={k:v.to(args.device) for k,v in row['observation'].items()}
                predictions.append(torch.sigmoid(model(data)).cpu().numpy())
                labels.append(row['labels'].numpy());operations.append(data['operations'].cpu().numpy())
        pred,y,ops=map(np.concatenate,[predictions,labels,operations]);metrics={}
        for op,name in enumerate(OPERATIONS):
            mask=ops==op
            metrics[name]={'average_precision':float(average_precision_score(y[mask],pred[mask])) if y[mask].any() else None,
                           'roc_auc':float(roc_auc_score(y[mask],pred[mask])) if 0<y[mask].sum()<mask.sum() else None,
                           'positives':int(y[mask].sum()),'actions':int(mask.sum())}
        score=float(np.mean([m['average_precision'] for m in metrics.values() if m['average_precision'] is not None]))
        record={'epoch':epoch+1,'loss':float(np.mean(losses)),'validation':metrics,'macro_ap':score,'seconds':time.monotonic()-start}
        report['epochs'].append(record)
        if score>best:
            best=score;report['selected_epoch']=epoch+1
            checkpoint=ROOT/'checkpoints'/f'{args.output}.pt'
            torch.save({'model':model.state_dict(),'mean':mean,'scale':scale,'config':vars(args),
                        'epoch':epoch+1,'architecture':{'width':64,'layers':2,'representation':args.representation}},checkpoint)
            report['checkpoint_hash']=sha256(checkpoint)
            np.savez_compressed(ROOT/'data/edit_learning'/f'{args.output}_validation_scores.npz',probability=pred,label=y,operation=ops)
        report['seconds']=time.monotonic()-start
        destination.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(record),flush=True)
    print(json.dumps({'selected_epoch':report['selected_epoch'],'best_macro_ap':best,'seconds':report['seconds']}),flush=True)


if __name__=='__main__':main()
