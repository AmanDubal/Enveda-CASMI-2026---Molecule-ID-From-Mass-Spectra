"""Small CPU regression checks; not neural training or accuracy validation."""
import types
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import nbformat

torch.set_num_threads(2)
torch.manual_seed(42)
MAX_SMILES_TOKENS = 16
tokenizer = types.SimpleNamespace(token_to_id=lambda _: 3)
nb = nbformat.read('casmi_corrected_denovo.ipynb', as_version=4)
exec(next(c.source for c in nb.cells if c.cell_type=='code' and c.source.startswith('# define module classes')))
dec = SmilesDecoder(embed_dim=16,vocab_size=12,n_layers=1,n_heads=2,
                    pad_token_id=0,bos_token_id=1,eos_token_id=2,dropout=0.)
dec.eval()
# Nonzero head exposes causal leakage rather than trivially comparing zero logits.
nn.init.normal_(dec.lm_head.weight,std=.1)
memory=torch.randn(2,4,16)
mask=torch.ones(2,4,dtype=torch.long)
a=torch.tensor([[1,4,5,6],[1,7,8,9]])
b=a.clone();b[:,-1]=10
out_a=dec(a,memory,mask)['logits']
out_b=dec(b,memory,mask)['logits']
assert torch.allclose(out_a[:,:3],out_b[:,:3],atol=1e-6), 'Future-token leakage'
target=torch.tensor([[4,5,6,2],[7,8,9,2]])
loss=dec(a,memory,mask,structure_tokens=target)['loss']
loss.backward()
assert torch.isfinite(loss)
assert dec.wpe.weight.grad is not None

# Verify that generation presents the full growing prefix and accumulates scores.
lengths=[]
def forced_forward(self,idx,encoder_outputs,encoder_attention_mask,structure_tokens=None):
    lengths.append(idx.shape[1])
    logits=torch.full((len(idx),idx.shape[1],12),-100.)
    logits[:,-1,4 if idx.shape[1]<3 else 2]=100.
    return {'logits':logits}
dec.forward=types.MethodType(forced_forward,dec)
tokens,scores=dec.generate(memory,mask,n_samples=3,max_new_tokens=8)
assert lengths==[1,2,3],lengths
assert tokens.shape==(2,3,4)
assert (tokens[:,:,-1]==2).all() and torch.isfinite(scores).all()

# Exercise the notebook's actual teacher() against a capture decoder.
class LightningStub(nn.Module):
    def save_hyperparameters(self,*args,**kwargs):pass
L=types.SimpleNamespace(LightningModule=LightningStub)
exec(next(c.source for c in nb.cells if c.cell_type=='code' and c.source.startswith('def ranked_candidates')))
params={'peak_embedder':{'d_model':16,'dropout':0.},
        'spectrum_encoder':{'embed_dim':16,'n_heads':2,'n_layers':1,'dropout':0.},
        'smiles_decoder':{'embed_dim':16,'vocab_size':12,'n_layers':1,'n_heads':2,
                          'pad_token_id':0,'bos_token_id':1,'eos_token_id':2,'dropout':0.},
        'optimizer':{'lr':1e-3}}
model=DeNovoLightningModel(params)
seen={}
class Capture(nn.Module):
    def forward(self,idx,encoder_outputs,encoder_attention_mask,structure_tokens=None):
        seen['input']=idx;seen['target']=structure_tokens
        return {'loss':torch.tensor(0.)}
model.smiles_decoder=Capture()
all_tokens=torch.tensor([[1,4,5,2,0],[1,6,7,8,2]])
model.teacher({'structure_tokens':all_tokens,'mzs':torch.tensor([[100.,50.,0.],[200.,75.,25.]]),
               'intensities':torch.tensor([[2.,1.,0.],[2.,1.,.5]]),
               'attention_mask':torch.tensor([[1,1,0],[1,1,1]])})
assert torch.equal(seen['input'],all_tokens[:,:-1])
assert torch.equal(seen['target'],all_tokens[:,1:])
print('PASS: causal masking, finite loss/backpropagation, positional gradients, full-prefix generation, next-token teacher targets')
