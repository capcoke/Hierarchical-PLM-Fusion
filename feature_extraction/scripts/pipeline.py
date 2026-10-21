"""Offline adaptations of this project's ESM-2 and ProtT5 extractors."""
import argparse
import csv
import hashlib
import json
import os
import socket
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import h5py

BRANCHES={'esm2':{'ESM2_layer28':1280,'ESM2_layer33':1280},'prott5':{'Prot5':1024}}
REPLACEMENTS=str.maketrans({'U':'X','Z':'X','O':'X'})


def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def sha_text(value):return hashlib.sha256(value.encode('utf-8')).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()


def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with open(tmp,'w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def load_fasta(path,id_policy='exact'):
    records=[];pid=None;parts=[];ids=set()
    def finish():
        if pid is None:return
        sequence=''.join(parts)
        if not sequence or sequence!=sequence.upper() or not sequence.isascii() or not sequence.isalpha():
            raise ValueError(f'{pid}: FASTA must contain uppercase ungapped amino-acid letters; no silent position edits')
        if pid in ids:raise ValueError('Duplicate/colliding protein ID: '+pid)
        if '/' in pid or pid in ('.','..'):raise ValueError('Unsafe H5 key: '+pid)
        ids.add(pid)
        positions=list(range(len(sequence))) if len(sequence)<=1000 else list(range(500))+list(range(len(sequence)-500,len(sequence)))
        truncated=''.join(sequence[i] for i in positions)
        records.append({'protein_id':pid,'original_sequence':sequence,'truncated_sequence':truncated,
            'original_length':len(sequence),'input_length':len(truncated),'original_sequence_sha256':sha_text(sequence),
            'truncated_sequence_sha256':sha_text(truncated),'original_positions_0based':positions})
    with open(path,encoding='utf-8') as f:
        for line in f:
            if line.startswith('>'):
                finish();header=line[1:].strip()
                if not header:raise ValueError('Empty FASTA header')
                pid=header.split()[0];parts=[]
                if id_policy=='catalodb':pid=pid.replace('/','_').replace('.','_')
                elif id_policy=='screen':
                    sep='-' if '-' in pid else '_';pdb,chain=pid.split(sep)
                    pid=pdb.lower()+'-'+chain.upper()
            elif line.strip():
                if pid is None:raise ValueError('Sequence precedes FASTA header')
                parts.append(''.join(line.split()))
    finish()
    if not records:raise ValueError('Empty FASTA')
    return records


def branch_sequence(record,branch,profile,prott5_long_sequence_policy='pre_truncate'):
    if prott5_long_sequence_policy not in ('pre_truncate','encode_full_then_truncate'):raise ValueError('Unknown ProtT5 policy')
    seq=record['original_sequence'] if branch=='prott5' and prott5_long_sequence_policy=='encode_full_then_truncate' else record['truncated_sequence']
    return seq.translate(REPLACEMENTS) if branch=='prott5' or profile=='catalodb' else seq


def encoding_length(record,branch,args):
    return record['original_length'] if branch=='prott5' and getattr(args,'prott5_long_sequence_policy','pre_truncate')=='encode_full_then_truncate' else record['input_length']


def model_files(path,tokenizer=False):
    path=Path(path).expanduser().resolve()
    if not path.is_dir():raise FileNotFoundError(f'Offline model directory missing: {path}')
    if not tokenizer and not (path/'config.json').is_file():raise FileNotFoundError(f'Missing config.json: {path}')
    if not tokenizer and not any(path.glob('*.safetensors')) and not any(path.glob('pytorch_model*.bin')):
        raise FileNotFoundError(f'No local Hugging Face weights found: {path}; fair-esm .pt is not interchangeable')
    names=['*.json','*.txt','*.model','*.safetensors','pytorch_model*.bin']
    paths=sorted(set(p for pattern in names for p in path.glob(pattern) if p.is_file()))
    return {'directory':str(path),'files':[{'name':p.name,'sha256':sha_file(p),'bytes':p.stat().st_size} for p in paths]}


@contextmanager
def lock_directory(out,recover=False):
    out.mkdir(parents=True,exist_ok=True);lock=out/'.pipeline.lock'
    if lock.exists() and recover:
        if os.name!='posix':raise RuntimeError('Automatic stale-lock recovery is POSIX-only; on Windows verify termination and remove only this lock manually')
        old=json.loads(lock.read_text())
        if old['host']!=socket.gethostname():raise RuntimeError('Foreign-host lock: manually confirm process termination before removal')
        try:os.kill(int(old['pid']),0)
        except ProcessLookupError:lock.unlink()
        except OSError:raise RuntimeError('Cannot verify stale lock; do not run concurrently')
        else:raise RuntimeError('Lock owner is still alive; refusing recovery')
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('Output is locked. For a terminated same-host process use --recover-lock')
    with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'host':socket.gethostname(),'utc':utc()},f)
    try:yield
    finally:lock.unlink(missing_ok=True)


def shard_path(out,branch,pid):return out/'.progress'/branch/(sha_text(pid)+'.npz')


def read_shard(path,signature,record,branch):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata'].item()))
        if meta['signature']!=signature or meta['protein_id']!=record['protein_id'] or meta['original_sequence_sha256']!=record['original_sequence_sha256']:
            raise ValueError('Resume provenance mismatch: '+str(path))
        arrays={name:z[name].copy() for name in BRANCHES[branch]}
    for name,a in arrays.items():
        if a.shape!=(record['input_length'],BRANCHES[branch][name]) or not np.isfinite(a).all():raise ValueError('Corrupt shard: '+str(path))
        if a.dtype.name!=meta['dtype'] or hashlib.sha256(a.tobytes()).hexdigest()!=meta['array_sha256'][name]:raise ValueError('Shard checksum/dtype mismatch')
    return arrays,meta


def save_shard(out,branch,signature,record,arrays,source,precision):
    path=shard_path(out,branch,record['protein_id']);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('Refusing duplicate shard write')
    for name,a in arrays.items():
        if a.shape!=(record['input_length'],BRANCHES[branch][name]) or not np.isfinite(a).all():raise ValueError('Invalid extracted features: '+record['protein_id'])
    policy=source.get('prott5_long_sequence_policy','pre_truncate')
    seq=branch_sequence(record,branch,source['profile'],policy)
    meta={k:v for k,v in record.items() if k not in ('original_sequence','truncated_sequence')}
    meta.update({'signature':signature,'h5_key':record['protein_id'],'dtype':precision,'model_source':source,
        'model_input_sequence_sha256':sha_text(seq),'model_input_length':len(seq),'feature_length':record['input_length'],
        'prott5_long_sequence_policy':policy if branch=='prott5' else None,'extracted_utc':utc(),
        'array_sha256':{name:hashlib.sha256(a.tobytes()).hexdigest() for name,a in arrays.items()}})
    tmp=path.with_suffix('.tmp')
    with open(tmp,'wb') as f:
        np.savez(f,metadata=json.dumps(meta),**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def pack(out,branch,records,signature):
    metas=[]
    for r in records:_,m=read_shard(shard_path(out,branch,r['protein_id']),signature,r,branch);metas.append(m)
    for name,width in BRANCHES[branch].items():
        path=out/name/'train_per_residue_embeddings.h5';path.parent.mkdir(exist_ok=True)
        if path.exists():
            with h5py.File(path,'r') as h:
                if h.attrs.get('pipeline_signature')!=signature:raise ValueError('Refusing to overwrite unowned H5: '+str(path))
                if set(h)!=set(r['protein_id'] for r in records):raise ValueError('Published H5 keys changed')
                for r,m in zip(records,metas):
                    a=h[r['protein_id']][:]
                    if hashlib.sha256(a.tobytes()).hexdigest()!=m['array_sha256'][name]:raise ValueError('Published H5 changed')
            continue
        tmp=path.with_suffix('.tmp.h5')
        with h5py.File(tmp,'w') as h:
            h.attrs['pipeline_signature']=signature;h.attrs['created_utc']=utc()
            for r in records:
                arrays,_=read_shard(shard_path(out,branch,r['protein_id']),signature,r,branch)
                h.create_dataset(r['protein_id'],data=arrays[name])
            h.flush()
        os.replace(tmp,path)
    atomic_json(out/(branch+'_manifest.json'),{'branch':branch,'signature':signature,'records':metas})
    temp=out/(branch+'_manifest.csv.tmp')
    with open(temp,'w',encoding='utf-8',newline='') as f:
        fields=['protein_id','original_length','input_length','original_sequence_sha256','truncated_sequence_sha256',
                'model_input_sequence_sha256','model_input_length','feature_length','prott5_long_sequence_policy','feature_dimension','h5_key','dtype','model_source','extracted_utc','original_positions_0based','h5_path']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for m in metas:
            for name,width in BRANCHES[branch].items():
                row={k:m[k] for k in fields if k not in ('feature_dimension','h5_path')};row['feature_dimension']=width
                row['model_source']=json.dumps(m['model_source']);row['original_positions_0based']=json.dumps(m['original_positions_0based'])
                row['h5_path']=name+'/train_per_residue_embeddings.h5';writer.writerow(row)
    os.replace(temp,out/(branch+'_manifest.csv'))


def offline_model(branch,args):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from transformers import AutoModel,AutoTokenizer,T5EncoderModel,T5Tokenizer
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('Requested CUDA unavailable; check PyTorch/CUDA installation')
    if branch=='esm2':
        tokenizer=AutoTokenizer.from_pretrained(args.esm2_model_dir,local_files_only=True,trust_remote_code=False)
        model=AutoModel.from_pretrained(args.esm2_model_dir,local_files_only=True,trust_remote_code=False).float()
        if model.config.num_hidden_layers!=33 or model.config.hidden_size!=1280:raise ValueError('Expected ESM2 t33 650M, not 3B/15B')
    else:
        tokenizer=T5Tokenizer.from_pretrained(args.prott5_tokenizer_dir,local_files_only=True,do_lower_case=False)
        model=T5EncoderModel.from_pretrained(args.prott5_model_dir,local_files_only=True)
        model=model.half() if args.prott5_precision=='float16' else model.float()
        if model.config.d_model!=1024:raise ValueError('Expected ProtT5 encoder width1024')
    model=model.to(args.device).eval();model.requires_grad_(False)
    return tokenizer,model


def encode_batch(branch,records,tokenizer,model,args):
    import torch
    policy=getattr(args,'prott5_long_sequence_policy','pre_truncate')
    seqs=[branch_sequence(r,branch,args.profile,policy) for r in records]
    texts=seqs if branch=='esm2' else [' '.join(seq) for seq in seqs]
    enc=tokenizer(texts,return_tensors='pt',add_special_tokens=True,padding='longest')
    counts=enc['attention_mask'].sum(dim=1).tolist()
    for seq,n in zip(seqs,counts):
        if n!=len(seq)+(2 if branch=='esm2' else 1):raise ValueError('Tokenizer does not produce one token per residue')
    enc={k:v.to(args.device) for k,v in enc.items()}
    with torch.no_grad():
        if branch=='esm2':
            output=model(**enc,output_hidden_states=True)
            return [{f'ESM2_layer{layer}':output.hidden_states[layer][i,1:1+len(seq)].detach().cpu().numpy().astype(np.float32) for layer in [28,33]} for i,seq in enumerate(seqs)]
        try:output=model(enc['input_ids'],attention_mask=enc['attention_mask'])
        except RuntimeError as exc:
            if isinstance(exc,torch.cuda.OutOfMemoryError):raise
            raise RuntimeError(f'ProtT5 failed under {policy}; proteins={[r["protein_id"] for r in records]}, encoded_lengths={[len(s) for s in seqs]}; no fallback or chunking') from exc
        results=[]
        for i,(r,seq) in enumerate(zip(records,seqs)):
            array=output.last_hidden_state[i,:len(seq)].detach().cpu().numpy().astype(args.prott5_precision)
            if policy=='encode_full_then_truncate':array=array[r['original_positions_0based']]
            results.append({'Prot5':array})
        return results


def batches(records,branch,args):
    if branch=='esm2':
        for i in range(0,len(records),args.esm2_batch_size):yield records[i:i+args.esm2_batch_size]
    else:
        batch=[]
        # Preserve get_embeddings' historical double-counting of the current length.
        for i,r in enumerate(sorted(records,key=lambda r:encoding_length(r,branch,args),reverse=True)):
            batch.append(r)
            if len(batch)>=args.prott5_batch_size or sum(encoding_length(x,branch,args) for x in batch)+encoding_length(r,branch,args)>=args.prott5_max_residues or i==len(records)-1 or encoding_length(r,branch,args)>1000:
                yield batch;batch=[]


def run_branch(branch,args):
    import torch,transformers
    records=load_fasta(args.fasta,args.id_policy);out=Path(args.output).resolve()
    sources={'profile':args.profile,'model':model_files(args.esm2_model_dir if branch=='esm2' else args.prott5_model_dir)}
    if branch=='prott5':
        sources['tokenizer']=model_files(args.prott5_tokenizer_dir,True)
        sources['prott5_long_sequence_policy']=getattr(args,'prott5_long_sequence_policy','pre_truncate')
    precision='float32' if branch=='esm2' else args.prott5_precision
    config={'branch':branch,'fasta_sha256':sha_file(args.fasta),'source':sources,'profile':args.profile,'id_policy':args.id_policy,
        'truncation':'if L>1000: first500 + last500','dtype':precision,'device':args.device,
        'batch_size':args.esm2_batch_size if branch=='esm2' else args.prott5_batch_size,
        'max_residues':None if branch=='esm2' else args.prott5_max_residues,
        'torch':torch.__version__,'transformers':transformers.__version__,'script_sha256':sha_file(__file__)}
    signature=sha_text(json.dumps(config,sort_keys=True))
    with lock_directory(out,args.recover_lock):
        cp=out/(branch+'_run.json')
        if cp.exists():
            if json.loads(cp.read_text())['signature']!=signature:raise ValueError('Configuration differs from previous run; use a fresh output directory')
        else:
            for name in BRANCHES[branch]:
                if (out/name/'train_per_residue_embeddings.h5').exists():raise ValueError('Original/unowned H5 exists; choose a NEW output directory')
            atomic_json(cp,{'signature':signature,'configuration':config,'started_utc':utc(),'fasta_path':str(Path(args.fasta).resolve())})
        # Resume at original batch boundaries to avoid changing padding-dependent numerics.
        pending=[]
        for batch in batches(records,branch,args):
            present=[]
            for r in batch:
                p=shard_path(out,branch,r['protein_id'])
                if p.exists():read_shard(p,signature,r,branch);present.append(True)
                else:present.append(False)
            if not all(present):pending.append((batch,present))
        if pending:
            try:
                tokenizer,model=offline_model(branch,args)
                for batch,present in pending:
                    arrays=encode_batch(branch,batch,tokenizer,model,args)
                    for r,a,exists in zip(batch,arrays,present):
                        if not exists:save_shard(out,branch,signature,r,a,sources,precision)
                    print(f'{branch}: committed {len(batch)} original-batch records; {batch[-1]["protein_id"]}',flush=True)
                del model,tokenizer
                if torch.cuda.is_available():torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError(f'GPU memory exhausted under {sources.get("prott5_long_sequence_policy","ESM pre_truncate")}. Committed shards are safe. No policy fallback or chunking. Smaller batch settings require a NEW output directory and renewed numerical validation.') from exc
        pack(out,branch,records,signature)
        print(branch+': complete; manifests and H5 verified',flush=True)


def parser(branch='all'):
    p=argparse.ArgumentParser(description='Offline SCREEN-compatible feature extraction, no training or downloads')
    p.add_argument('--config',type=Path);p.add_argument('--fasta',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--esm2-model-dir');p.add_argument('--prott5-model-dir');p.add_argument('--prott5-tokenizer-dir')
    p.add_argument('--device');p.add_argument('--profile',choices=['screen','catalodb']);p.add_argument('--id-policy',choices=['exact','screen','catalodb'])
    p.add_argument('--esm2-batch-size',type=int);p.add_argument('--prott5-batch-size',type=int);p.add_argument('--prott5-max-residues',type=int)
    p.add_argument('--prott5-precision',choices=['float16','float32']);p.add_argument('--recover-lock',action='store_true')
    p.add_argument('--prott5-long-sequence-policy',choices=['pre_truncate','encode_full_then_truncate'])
    a=p.parse_args();config={}
    if a.config:
        import yaml
        config=yaml.safe_load(a.config.read_text(encoding='utf-8')) or {}
    defaults={'device':'cuda:0','profile':'screen','id_policy':'exact','esm2_batch_size':1,'prott5_batch_size':100,'prott5_max_residues':4000,'prott5_precision':'float16','prott5_long_sequence_policy':'pre_truncate'}
    for key in ['esm2_model_dir','prott5_model_dir','prott5_tokenizer_dir']+list(defaults):
        if getattr(a,key) is None:setattr(a,key,config.get(key,defaults.get(key)))
    needed=['esm2_model_dir'] if branch=='esm2' else ['prott5_model_dir','prott5_tokenizer_dir'] if branch=='prott5' else ['esm2_model_dir','prott5_model_dir','prott5_tokenizer_dir']
    if any(not getattr(a,k) for k in needed):p.error('Provide local model directories using --config or CLI')
    if min(a.esm2_batch_size,a.prott5_batch_size,a.prott5_max_residues)<1:p.error('Batch limits must be positive')
    if a.prott5_long_sequence_policy not in ('pre_truncate','encode_full_then_truncate'):p.error('Invalid ProtT5 policy')
    return a
