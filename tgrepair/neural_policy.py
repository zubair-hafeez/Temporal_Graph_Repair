"""Compact temporal graph policy that scores typed, reversible graph edits."""
import torch
from torch import nn

from .action_features import ACTION_DIM,NODE_DIM


class GraphEditPolicy(nn.Module):
    def __init__(self,width=64,layers=2,representation='temporal'):
        super().__init__()
        if representation not in {'temporal','static','mlp'}:raise ValueError(representation)
        self.representation=representation
        self.node_input=nn.Linear(NODE_DIM,width)
        self.type_embedding=nn.Embedding(17,width,padding_idx=0)
        self.activity_embedding=nn.Embedding(129,width,padding_idx=0)
        self.relation_embedding=nn.Embedding(4,16)
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(width+19,width),nn.ReLU(),nn.Linear(width,width)) for _ in range(layers)])
        self.gates=nn.ModuleList([nn.Sequential(nn.Linear(2*width+19,width//2),nn.ReLU(),nn.Linear(width//2,1),nn.Sigmoid()) for _ in range(layers)])
        self.updates=nn.ModuleList([nn.Sequential(nn.Linear(2*width,width),nn.ReLU(),nn.Dropout(.1)) for _ in range(layers)])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(3*width+ACTION_DIM,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,1)) for _ in range(3)])

    def forward(self,data):
        h=torch.relu(self.node_input(data['node_features'])+self.type_embedding(data['node_types'])+self.activity_embedding(data['activity_types']))
        edges=data['edges'];features=data['edge_features']
        if self.representation=='static':
            keep=features[:,0]<2;edges=edges[:,keep];features=features[keep]
        if self.representation!='mlp' and edges.shape[1]:
            source,target=edges
            relation=torch.cat([self.relation_embedding(features[:,0].long()),features[:,1:]],dim=1)
            for message,gate,update,norm in zip(self.messages,self.gates,self.updates,self.norms):
                weight=gate(torch.cat([h[source],h[target],relation],dim=1))
                values=message(torch.cat([h[source],relation],dim=1))*weight
                aggregate=torch.zeros_like(h).index_add_(0,target,values)
                denominator=torch.zeros((len(h),1),device=h.device).index_add_(0,target,weight).clamp_min(1)
                h=norm(h+update(torch.cat([h,aggregate/denominator],dim=1)))
        source,target=data['action_nodes'].T
        pair=torch.cat([h[source],h[target],h[source]*h[target],data['action_features']],dim=1)
        result=torch.zeros(len(pair),device=h.device)
        for operation,head in enumerate(self.heads):
            mask=data['operations']==operation
            if mask.any():result[mask]=head(pair[mask]).squeeze(1)
        return result


def observation_tensors(observation,device='cpu',indices=None):
    data={name:torch.as_tensor(getattr(observation,name),device=device) for name in
          ['node_features','node_types','activity_types','edges','edge_features','action_nodes','action_features','operations']}
    if indices is not None:
        indices=torch.as_tensor(indices,device=device)
        for key in ['action_nodes','action_features','operations']:data[key]=data[key][indices]
    return data
