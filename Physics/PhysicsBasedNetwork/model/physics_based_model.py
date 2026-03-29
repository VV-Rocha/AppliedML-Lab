import sys
sys.path.append("../solver/")
from euler_scheme import _Solver

import torch
from torch import nn

class _PhysicalParameters:
    def _physical_model(self,):
        self.gamma = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.k = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.k2 = nn.Parameter(torch.tensor(.05), requires_grad=True)
        
        self.psi0 = nn.Parameter(torch.tensor([.5, .5]), requires_grad=True)

    def init_physical_model(self,):
        self._physical_model()

class _SolverConfig:
    def __init__(
        self,
        tmax,
        Nt,
        dtype,
        device,
        verbose,
        *args,
        **kwargs,
    ):
        self.tmax = tmax
        self.Nt = Nt
        
        if dtype == "float64":
            self.dtype = torch.float64
        elif dtype == "float32":
            self.dtype = torch.float32
            
        self.device = device
        self.verbose = verbose
            
        super().__init__(*args, **kwargs)    

class _SolverInit(_SolverConfig):
    def _time_grid(self,):
        self.t = torch.linspace(0., self.tmax, self.Nt)
        self.dt = self.t[1]-self.t[0]
        
    def _states(self,):
        self.states = torch.zeros((self.Nt-1, 2))  # [Nt-1, x/v]  THE INITIAL STATE IS TRAINED FOR SO IT WILL NOT BE INCLUDED IN THIS states ARRAY
    
    def _init_dtypes(self,):
        torch.set_default_dtype(self.dtype)
        
    def _init_device(self,):
        torch.set_default_device(self.device)
        
    def init_solver(self,):
        self._init_dtypes()
        self._init_device()
        self._time_grid()
        self._states()

class _Solver(_SolverInit):
    def _solve(self, i):
        states = []
        current_state = self.psi0
        for _ in range(self.Nt - 1):
            current_state = current_state @ self.matrix
            states.append(current_state)

        self.states = torch.stack(states, dim=0)
                    
    def solve(self,):
        states = []
        current_state = self.psi0
        for _ in range(self.Nt - 1):
            current_state = (current_state @ self.matrix) + self.k2 * torch.tensor([0., current_state[1]**2])
            states.append(current_state)

        self.states = torch.stack(states, dim=0)

class PhysicsModel(_Solver, _PhysicalParameters, nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.init()
    
    def init(self,):
        self.init_solver()
        self.init_physical_model()
        
    def update_matrix(self):
        one = torch.tensor(1.0, dtype=self.dtype, device=self.device)

        self.matrix = torch.stack([
            torch.stack([one, -self.k * self.dt]),
            torch.stack([self.dt, one - self.gamma * self.dt])
        ])
        
    def forward(self,):
        self.solve()
        return self.states