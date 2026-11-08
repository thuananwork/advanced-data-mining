from pathlib import Path
import numpy as np, json, gzip

def write_localpfn_tabzilla(root, X, y, chosen, split, dataset_name='breast_cancer_shared'):
    out=Path(root)/'datasets_tabzilla'/dataset_name
    out.mkdir(parents=True,exist_ok=True)
    metadata={'name':dataset_name,'target_type':'classification',
              'cat_idx':[],'cat_dims':[],'num_instances':int(len(y)),
              'num_features':int(X.shape[1]),'num_classes':int(len(np.unique(y)))}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    def save(name,arr):
        with gzip.open(out/name,'wb') as f:np.save(f,arr,allow_pickle=True)
    save('X.npy.gz',np.asarray(X,dtype=np.float32))
    save('y.npy.gz',np.asarray(y,dtype=np.int64))
    s={'train':np.asarray(chosen,dtype=int),'val':np.asarray(split['val'],dtype=int),
       'test':np.asarray(split['test'],dtype=int)}
    save('split_indeces.npy.gz',np.asarray([s],dtype=object)) # original repo spelling
    return out
