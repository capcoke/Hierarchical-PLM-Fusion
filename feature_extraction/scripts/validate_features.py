"""Validate residue identity/mapping and optionally compare unchanged SCREEN H5."""
import argparse
import json
from pathlib import Path
import numpy as np
import h5py
from pipeline import BRANCHES,load_fasta,branch_sequence,sha_text,sha_file,atomic_json


def validate(fasta,output,reference=None,id_policy='exact',profile='screen',esm_max=1e-4,esm_mean=1e-5,prot_max=.02,prot_mean=.001,prott5_long_sequence_policy=None,reference_mapping='raw'):
    records=load_fasta(fasta,id_policy);results=[];errors=[]
    for branch,names in BRANCHES.items():
        manifest_path=output/(branch+'_manifest.json')
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        metas={r['protein_id']:r for r in manifest['records']}
        if set(metas)!=set(r['protein_id'] for r in records):raise ValueError('Manifest ID set mismatch')
        for name,width in names.items():
            path=output/name/'train_per_residue_embeddings.h5'
            generated_hash=sha_file(path)
            reference_hash=sha_file(reference/name/'train_per_residue_embeddings.h5') if reference else None
            with h5py.File(path,'r') as h:
                if h.attrs.get('pipeline_signature')!=manifest['signature']:raise ValueError('H5 provenance signature mismatch')
                if set(h)!=set(metas):raise ValueError('H5 ID set mismatch')
                ref=h5py.File(reference/name/'train_per_residue_embeddings.h5','r') if reference else None
                try:
                    for r in records:
                        pid=r['protein_id'];a=h[pid][:];m=metas[pid]
                        expected_dtype='float32' if branch=='esm2' else m['dtype']
                        valid=a.shape==(r['input_length'],width) and a.dtype.name==expected_dtype and np.isfinite(a).all()
                        valid=valid and m['original_positions_0based']==r['original_positions_0based'] and m['original_sequence_sha256']==r['original_sequence_sha256'] and m['truncated_sequence_sha256']==r['truncated_sequence_sha256']
                        policy=m.get('prott5_long_sequence_policy') or 'pre_truncate'
                        seq=branch_sequence(r,branch,profile,policy)
                        valid=valid and m['model_input_sequence_sha256']==sha_text(seq) and m.get('model_input_length',len(seq))==len(seq) and m['h5_key']==pid
                        if branch=='prott5' and prott5_long_sequence_policy is not None:valid=valid and policy==prott5_long_sequence_policy
                        valid=valid and m['array_sha256'][name]==__import__('hashlib').sha256(a.tobytes()).hexdigest()
                        row={'protein_id':pid,'branch':name,'original_length':r['original_length'],'input_length':len(a),'shape':list(a.shape),'dtype':a.dtype.name,
                             'structural_mapping_finite_checksum_pass':bool(valid),'generated_h5_sha256':generated_hash}
                        if ref is not None:
                            b=ref[pid][:] if pid in ref else None
                            row['reference_raw_shape']=list(b.shape) if b is not None else None
                            row['reference_mapping']=reference_mapping
                            if b is not None and reference_mapping=='model_input' and len(b)>1000:
                                b=np.concatenate([b[:500],b[-500:]],axis=0)
                            if b is None or b.shape!=a.shape:valid=False;row['reference_error']='missing ID or mismatched shape'
                            else:
                                diff=np.abs(a.astype(np.float64)-b.astype(np.float64))
                                mx=float(diff.max());mean=float(diff.mean());limit_max=esm_max if branch=='esm2' else prot_max;limit_mean=esm_mean if branch=='esm2' else prot_mean
                                same_dtype=a.dtype==b.dtype
                                numeric=np.isfinite(b).all() and same_dtype and mx<=limit_max and mean<=limit_mean
                                valid=valid and numeric
                                row.update({'reference_dtype':b.dtype.name,'same_dtype':same_dtype,'max_absolute_error':mx,'mean_absolute_error':mean,
                                    'bitwise_equal':bool(np.array_equal(a,b)),'max_error_limit':limit_max,'mean_error_limit':limit_mean,
                                    'numerical_tolerance_pass':bool(numeric),'reference_h5_sha256':reference_hash})
                        row['pass']=bool(valid);results.append(row)
                        if not valid:errors.append(name+'/'+pid)
                finally:
                    if ref is not None:ref.close()
    return {'status':'FAIL' if errors else 'NUMERICAL_TOLERANCE_PASS' if reference else 'STRUCTURAL_VALIDATION_ONLY',
        'numerical_validation':'COMPARED_EXISTING_H5' if reference else 'PENDING_REFERENCE_COMPARISON',
        'reference_scope':'selected protein IDs only; no reinterpretation of whole historical data',
        'tolerances_declared_before_comparison':{'esm_max':esm_max,'esm_mean':esm_mean,'prot_max':prot_max,'prot_mean':prot_mean},
        'records':results,'errors':errors,'all_bitwise_equal':bool(reference and all(r.get('bitwise_equal',False) for r in results))}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fasta',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference-root',type=Path)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--profile',choices=['screen','catalodb'],default='screen');p.add_argument('--id-policy',choices=['exact','screen','catalodb'],default='exact')
    p.add_argument('--reference-mapping',choices=['raw','model_input'],default='raw')
    p.add_argument('--prott5-long-sequence-policy',choices=['pre_truncate','encode_full_then_truncate'])
    a=p.parse_args();result=validate(a.fasta,a.output,a.reference_root,a.id_policy,a.profile,prott5_long_sequence_policy=a.prott5_long_sequence_policy,reference_mapping=a.reference_mapping);atomic_json(a.report,result)
    print(json.dumps({'status':result['status'],'all_bitwise_equal':result['all_bitwise_equal'],'errors':result['errors']}))
    raise SystemExit(1 if result['errors'] else 0)
