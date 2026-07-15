"""Evaluation helpers for standalone Text-JEPA checkpoints."""
from __future__ import annotations
import argparse, json, os
import torch
from data import FixedBlockDataset, make_loader
from model import TextJEPAConfig, TextSpanJEPA

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--out',required=True); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); ap.add_argument('--batches',type=int,default=20); args=ap.parse_args()
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False); cfg=ckpt['config']; mcfg=TextJEPAConfig(**cfg['model'])
    model=TextSpanJEPA(mcfg); model.load_state_dict(ckpt['model']); model.to(args.device).eval()
    ds=FixedBlockDataset(cfg['val_path'], mcfg.block_size, expected_vocab_size=mcfg.vocab_size); loader=make_loader(ds,cfg.get('eval_batch_size',cfg['batch_size']),shuffle=False)
    sums={k:0.0 for k in ['loss','pred_loss','future_loss','decoder_loss','decoder_accuracy','variance_loss','covariance_loss','online_std','target_std','target_center_norm']}; n=0
    for i,b in enumerate(loader):
        if i>=args.batches: break
        out=model(b['input_ids'].to(args.device)); n+=1
        for k in sums: sums[k]+=float(out[k].detach().cpu())
    res={k:v/max(1,n) for k,v in sums.items()}; res['batches']=n
    os.makedirs(os.path.dirname(args.out) or '.',exist_ok=True)
    with open(args.out,'w') as f: json.dump(res,f,indent=2,sort_keys=True)
    print(json.dumps(res,indent=2,sort_keys=True))
if __name__=='__main__': main()
