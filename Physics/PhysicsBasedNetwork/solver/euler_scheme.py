import torch

class _PhysicalConfig:
    def __init__(
        self,
        gamma,
        k,
        k2,
        *args,
        **kwargs,
    ):
        self.gamma = gamma
        self.k = k
        self.k2 = k2
        
        super().__init__(*args, **kwargs)

class _InitialConfig:
    def __init__(
        self,
        x0,
        v0,
        *args,
        **kwargs,
    ):
        self.x0 = x0
        self.v0 = v0
        
        super().__init__(*args, **kwargs)

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
        self.states = torch.zeros((self.Nt, 2))  # [Nt, x/v]
        
        # set initial conditions
        self.states[0,0] = self.x0
        self.states[0,1] = self.v0
    
    def _init_dtypes(self,):
        torch.set_default_dtype(self.dtype)
        
    def _init_device(self,):
        torch.set_default_device(self.device)
        
    def _matrix(self,):
        self.matrix = torch.tensor([[1, -self.k*self.dt], [self.dt, 1. - self.gamma*self.dt]])
        
    def init_solver(self,):
        self._init_dtypes()
        self._init_device()
        self._time_grid()
        self._states()
        self._matrix()
    
class _Solver:
    def _solve(self, i):
        self.states[i] = torch.matmul(self.states[i-1], self.matrix) + self.k2 * torch.tensor([0., self.states[i-1, 1]**2])

    def init(self,):
        self.init_solver()

    def solve(self,):
        for i in range(1, self.Nt):
            self._solve(i)
            
            if self.verbose==1:
                print(f"{i+1}/{self.Nt}")
    
class EulerSolver(_Solver, _SolverInit, _InitialConfig, _PhysicalConfig):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        self.init()