import hashlib,json
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from casmi_retrieval import prepare_peaks,identity

test=pd.read_parquet('test.parquet')
submission=pd.read_csv('submission.csv').set_index('molecule_id')
top={mid:identity(s.split(';')[0])[0] for mid,s in submission.smiles.items()}
def signature(mz,it,pmz,adduct):
    mz,it=prepare_peaks(mz,it,pmz)
    # Match the full processed peak representation; no chemical labels involved.
    payload=np.round(mz,6).astype('<f8').tobytes()+np.round(it,8).astype('<f8').tobytes()
    return adduct,hashlib.sha256(payload).hexdigest()
queries=defaultdict(list)
for r in test.itertuples():
    queries[signature(r.ms2_mzs,r.ms2_normalized_intensities,r.precursor_mz,r.adduct)].append((r.molecule_id,r.spectrum_id))
mzs={a:np.sort(g.precursor_mz.unique()) for a,g in test.groupby('adduct')}
evidence=defaultdict(set)
cols=['ingest_lib','normalized_smiles','adduct','precursor_mz','ms2_mzs','ms2_normalized_intensities']
for batch in pq.ParquetFile('train.parquet').iter_batches(batch_size=65536,columns=cols):
    df=batch.to_pandas()
    for adduct,group in df.groupby('adduct'):
        if adduct not in mzs:continue
        pm=group.precursor_mz.to_numpy();known=mzs[adduct]
        pos=np.clip(np.searchsorted(known,pm),0,len(known)-1)
        prev=np.clip(pos-1,0,len(known)-1)
        keep=np.minimum(abs(known[pos]-pm),abs(known[prev]-pm))<=0.005
        for r in group.loc[keep].itertuples():
            sig=signature(r.ms2_mzs,r.ms2_normalized_intensities,r.precursor_mz,r.adduct)
            if sig in queries:
                ident=identity(r.normalized_smiles)
                if ident:
                    for mid,sid in queries[sig]:evidence[(mid,sid)].add((r.ingest_lib,ident[0]))
rows=[]
for (mid,sid),entries in evidence.items():
    keys={key for _,key in entries}
    rows.append({'molecule_id':mid,'spectrum_id':sid,'training_sources':';'.join(sorted({s for s,_ in entries})),
                 'matching_structure_count':len(keys),'top_prediction_matches':top[mid] in keys,
                 'training_keys':';'.join(sorted(keys))})
audit=pd.DataFrame(rows);audit.to_csv('overlap_audit.csv',index=False)
report={'matched_processed_spectra':len(audit),'total_test_spectra':len(test),
        'molecules_with_matches':audit.molecule_id.nunique(),'total_test_molecules':test.molecule_id.nunique(),
        'matched_spectra_top_prediction_consistent':int(audit.top_prediction_matches.sum()),
        'ambiguous_spectra':int((audit.matching_structure_count>1).sum()),
        'source_counts':audit.training_sources.value_counts().to_dict(),
        'matching_definition':'Same adduct; precursor within 0.005 Da; processed m/z rounded to 6 decimals and normalized square-root intensity to 8 decimals identical.'}
open('overlap_report.json','w').write(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
