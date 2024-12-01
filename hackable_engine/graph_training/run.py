import torch as th

from hackable_engine.graph_training.models.gcn import GCN

device = th.device('xpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = th.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = th.nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
