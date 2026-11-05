"""Original transfer architectures with graph data passed explicitly.

No experiment runs at import time. Common initialization and layer order are
preserved so the extracted models can be checked against the original runs.
"""
import torch
from torch import nn

HIDDEN = 32

def group_mean(x,index,size):
    total=x.new_zeros((size,x.shape[1])).index_add_(0,index,x)
    count=torch.bincount(index,minlength=size).to(x.dtype).clamp_min(1).unsqueeze(1)
    return total/count


class HierarchicalRentalModel(nn.Module):
    """Same transforms in each arm; pooling and trainable area IDs are toggled."""
    def __init__(self,kind,bundle,graph):
        super().__init__()
        node_dim = bundle["nodes"].shape[1]
        record_dim = bundle["parts"][0]["record"].shape[1]
        BLOCK_SUB, SUB_AREA = graph["block_sub"], graph["sub_area"]
        BLOCK_AREA = SUB_AREA[BLOCK_SUB]
        NS, NA = graph["n_subzones"], graph["n_areas"]
        self.n_subzones, self.n_areas = NS, NA
        self.kind=kind
        self.pool=kind.startswith('Hierarchy')
        self.learn_area=kind.endswith('area_embeddings')
        self.projection=nn.Linear(node_dim,HIDDEN)
        self.sub_up=nn.ModuleList([nn.Linear(HIDDEN,HIDDEN) for _ in range(2)])
        self.area_up=nn.ModuleList([nn.Linear(HIDDEN,HIDDEN) for _ in range(2)])
        self.sub_self=nn.ModuleList([nn.Linear(HIDDEN,HIDDEN) for _ in range(2)])
        self.area_down=nn.ModuleList([nn.Linear(HIDDEN,HIDDEN,bias=False) for _ in range(2)])
        self.sub_down=nn.ModuleList([nn.Linear(HIDDEN,HIDDEN,bias=False) for _ in range(2)])
        self.block_self=nn.ModuleList([nn.Linear(HIDDEN,HIDDEN) for _ in range(2)])
        self.gates=nn.ModuleList([nn.Linear(2*HIDDEN,HIDDEN) for _ in range(2)])
        for gate in self.gates:nn.init.constant_(gate.bias,-2.)
        self.head=nn.Sequential(nn.Linear(HIDDEN+record_dim,HIDDEN),nn.ReLU(),nn.Dropout(.1),nn.Linear(HIDDEN,1))
        self.drop=nn.Dropout(.1)
        # Zero initialization and no random embedding initializer keep common
        # parameter initialization and dropout streams aligned between arms.
        self.sub_embedding=nn.Parameter(torch.zeros(NS,HIDDEN),requires_grad=self.learn_area)
        self.area_embedding=nn.Parameter(torch.zeros(NA,HIDDEN),requires_grad=self.learn_area)
        self.register_buffer('block_sub',BLOCK_SUB)
        self.register_buffer('sub_area',SUB_AREA)
        self.register_buffer('block_area',BLOCK_AREA)

    def forward(self,bundle,part):
        nodes = bundle["nodes"]
        z=self.drop(torch.relu(self.projection(nodes)))
        for depth in range(2):
            if self.pool:
                # Explicit bottom-up pass: blocks -> subzones -> planning areas.
                sub=group_mean(z,self.block_sub,self.n_subzones)
                sub=torch.relu(self.sub_up[depth](sub)+self.sub_embedding)
                area=group_mean(sub,self.sub_area,self.n_areas)
                area=torch.relu(self.area_up[depth](area)+self.area_embedding)
                # Top-down pass completes the hierarchy within this round.
                sub=torch.relu(self.sub_self[depth](sub)+self.area_down[depth](area[self.sub_area]))
                context=self.sub_down[depth](sub[self.block_sub])
            else:
                # Each block has its own virtual group states. The exact same
                # transforms/area ID embeddings are active, with no pooling
                # or message transfer from other blocks.
                sub=torch.relu(self.sub_up[depth](z)+self.sub_embedding[self.block_sub])
                area=torch.relu(self.area_up[depth](sub)+self.area_embedding[self.block_area])
                sub=torch.relu(self.sub_self[depth](sub)+self.area_down[depth](area))
                context=self.sub_down[depth](sub)
            gate=torch.sigmoid(self.gates[depth](torch.cat([z,context],dim=1)))
            z=self.drop(torch.relu(self.block_self[depth](z)+gate*context))
        return self.head(torch.cat([z[part['ids']],part['record']],dim=1)).squeeze(-1)


class StationModel(nn.Module):
    """Pool first, propagate on the small graph, then read back to each block.

    All arms have identical active parameters and initialization. The local MLP
    applies the same transforms to a private virtual station for each block.
    """
    def __init__(self, kind, bundle, graph, hidden=32):
        super().__init__()
        self.graph = graph
        self.kind = kind
        self.projection = nn.Linear(bundle['nodes'].shape[1], hidden)
        self.station_projection = nn.Linear(bundle['station'].shape[1], hidden)
        self.up = nn.Linear(hidden, hidden)
        self.roots = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        self.messages = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(3)])
        self.station_gates = nn.ModuleList([nn.Linear(2*hidden, hidden) for _ in range(3)])
        self.down = nn.Linear(hidden, hidden, bias=False)
        self.local = nn.Linear(hidden, hidden)
        self.gate = nn.Linear(2*hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden+bundle['parts'][0]['record'].shape[1], hidden),
                                  nn.ReLU(), nn.Dropout(.1), nn.Linear(hidden, 1))
        self.drop = nn.Dropout(.1)
        for gate in [*self.station_gates, self.gate]:
            nn.init.constant_(gate.bias, -2.)

    def forward(self, bundle, part):
        z = self.drop(torch.relu(self.projection(bundle['nodes'])))
        local = self.kind == 'MLP_station'
        if local:
            pooled = z
            descriptors = bundle['station'][self.graph["membership"]]
        else:
            pooled = z.new_zeros((self.graph["n_stations"], z.shape[1])).index_add_(0, self.graph["membership"], z)/self.graph["counts"]
            descriptors = bundle['station']
        station = torch.relu(self.up(pooled) + self.station_projection(descriptors))
        for layer, (root, message, gate) in enumerate(zip(self.roots, self.messages, self.station_gates)):
            propagate = self.kind in self.graph["adjacency"] and (self.kind != 'Rail_1' or layer == 0)
            incoming = message(self.graph["adjacency"][self.kind] @ station if propagate else station)
            amount = torch.sigmoid(gate(torch.cat([station, incoming], 1)))
            station = torch.relu(root(station)+amount*incoming)
        context = self.down(station if local else station[self.graph["membership"]])
        amount = torch.sigmoid(self.gate(torch.cat([z, context], 1)))
        z = self.drop(torch.relu(self.local(z)+amount*context))
        return self.head(torch.cat([z[part['ids']], part['record']], 1)).squeeze(-1)
