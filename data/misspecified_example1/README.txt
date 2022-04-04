Generated using generate_nn_hls_problems.ipynbProblem Properties
num contexts: 5000
num actions: 10
dim: 20
==============================
net0.pth
------------------------------
MLLinearNetwork(
  (layers): ModuleList(
    (0): Linear(in_features=20, out_features=60, bias=True)
    (1): Tanh()
    (2): Linear(in_features=60, out_features=10, bias=True)
    (3): Tanh()
  )
  (fc2): Linear(in_features=10, out_features=1, bias=False)
)
Following properties are computed wrt the network predictions
HLS rank: 10 / 10
is HLS: True
HLS min eig: 0.21417641653139144
is CMB: True

==============================
net1.pth
------------------------------
MLLinearNetwork(
  (layers): ModuleList(
    (0): Linear(in_features=20, out_features=60, bias=True)
    (1): Tanh()
    (2): Linear(in_features=60, out_features=10, bias=True)
    (3): Tanh()
  )
  (fc2): Linear(in_features=10, out_features=1, bias=False)
)
Following properties are computed wrt the network predictions
HLS rank: 10 / 10
is HLS: True
HLS min eig: 0.3729509800119442
is CMB: True

==============================
net2.pth
------------------------------
MLLinearNetwork(
  (layers): ModuleList(
    (0): Linear(in_features=20, out_features=300, bias=True)
    (1): Tanh()
  )
  (fc2): Linear(in_features=300, out_features=1, bias=False)
)
Following properties are computed wrt the network predictions
HLS rank: 300 / 300
is HLS: True
HLS min eig: 0.039743458476127855
is CMB: True

==============================
net3.pth
------------------------------
MLLinearNetwork(
  (layers): ModuleList(
    (0): Linear(in_features=20, out_features=10, bias=True)
    (1): Tanh()
    (2): Linear(in_features=10, out_features=10, bias=True)
    (3): Tanh()
  )
  (fc2): Linear(in_features=10, out_features=1, bias=False)
)
Following properties are computed wrt the network predictions
HLS rank: 10 / 10
is HLS: True
HLS min eig: 0.19327760868642435
is CMB: True

