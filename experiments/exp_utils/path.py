import os

class Path():
    """Class for handling Path strings relating to ChampSim.
    
    Inherited by:
        ResultsPath: Paths for results paths (.txt)
        TracePath: Paths for traces (.trace.gz)
    """
    def __init__(self, path: str):
        self.path = path
        self.full_trace = \
            Path._get_full_trace(path)  # Trace and simpoint together
        self.trace, self.simpoint = \
            Path._get_trace(path)       # Trace and simpoint separately
        
    @staticmethod
    def _get_full_trace(path):
        """Helper function to get the full trace
        from a path string."""
        return os.path.basename(path).split('-')[0].replace('.trace', '')
    
    @staticmethod
    def _get_trace(path):
        """Helper function to get the trace + simpoint
        from a path string."""
        tokens = Path._get_full_trace(path).split('_')
        if len(tokens) == 1: # GAP           : e.g. bfs
            return tokens[0], None
        if len(tokens) == 2: # SPEC '06 / 17 : e.g. astar_313B, 603.bwaves_s
            return tokens[0], tokens[1]
        if len(tokens) == 3: # Cloudsuite    : e.g. cassandra_phase0_core0
            return tokens[0] + '_' + tokens[2], tokens[1] # Name_Core, Simpoint
        
        
        
class TracePath(Path):
    """Class for handling trace path strings (.trace.gz)
    """
    def __init__(self, path: str):
        assert path.endswith('.trace.xz') \
               or path.endswith('.trace.gz') \
               or path.endswith('.trace'), \
               ('Wrong file extension for TracePath, should be'
                '.trace or .trace.(g|x)z')
        super.__init__(path)
        
        


class ResultsPath(Path):
    """Class for handling ChampSim results path strings (.txt)
    
    TODO: Build from actual output, instead of file name?"""
    
    def __init__(self, path: str):
        assert path.endswith('.txt'), \
            'Wrong file extension for ResultsPath, should be .txt'
        super().__init__(path)
        
        # Prefetchers
        self.l1_prefetcher, self.l2_prefetcher, self.llc_prefetcher = \
            ResultsPath._get_prefetcher_names(path) # Prefetcher names (l1, l2, llc)
        self.l2_prefetcher_degree, self.llc_prefetcher_degree = \
            ResultsPath._get_prefetcher_degrees(path) # Prefetcher degrees (l2, llc)
        
        # Pythia
        self.pythia_level_threshold = \
            ResultsPath._get_pythia_level_threshold(path)
        
        self.pythia_features = \
            ResultsPath._get_pythia_features(path)
        
        # Champsim
        self.champsim_seed = \
            ResultsPath._get_champsim_seed(path)

        
    @staticmethod
    def _get_prefetcher_names(path):
        """Get the prefetcher(s) names.
        """
        # (l1_prefethcher, l2_prefetcher, llc_prefetcher)
        p = os.path.basename(path).split('-')[2:5]
        for i in range(len(p)):
            if p[i] not in ['no', 'multi_pc_trace']:
                p[i] = (p[i].replace('spp_dev2', 'sppdev2')
                            .replace('scooby_double', 'scoobydouble')
                            .split('_')[0])
                p[i] = (p[i].replace(',', '_')
                            .replace('sppdev2', 'spp_dev2')
                            .replace('scoobydouble', 'scooby_double'))
        
        return (*p,)
    
    @staticmethod
    def _get_prefetcher_degrees(path):
        """Get the prefethcer(s) degrees.
        """
        # (l2_prefetcher, llc_prefetcher)
        p = os.path.basename(path).split('-')[3:5]
        d = [None for pref in p]
        for i in range(len(p)):
            if p[i] not in ['no', 'multi_pc_trace']:
                d[i] = (p[i].replace('spp_dev2', 'sppdev2')
                            .replace('scooby_double', 'scoobydouble')
                            .split('_')[1])
                d[i] = tuple((None if d == 'na' else int(d)) 
                             for d in d[i].split(','))
    
        return (*d,)
    
    @staticmethod
    def _get_pythia_level_threshold(path):
        """Get the level threshold for Pythia,
        if one of the prefetchers is using it.
        """
        if not any(p == 'scooby' for p in ResultsPath._get_prefetcher_names(path)):
            return None

        path = os.path.basename(path)
        if 'threshold' in path:
            return float(path[path.index('threshold'):].split('_')[1].split('.')[0])

        return None
    
    @staticmethod
    def _get_pythia_features(path):
        """Get the featuers for Pythia,
        if it is being used.
        """
        if not any(p == 'scooby' for p in ResultsPath._get_prefetcher_names(path)):
            return None

        path = os.path.basename(path)
        if 'features' in path:
            path_ = (path[path.index('features'):]
                         .split('_')[1]
                         .replace('seed','')
                         .split('.')[0])
            return tuple(int(f) for f in path_.split(','))
        
    @staticmethod
    def _get_champsim_seed(path):
        """Get the ChampSim seed used.
        """
        path = os.path.basename(path)
        if 'seed' in path:
            path_ = path[path.index('seed'):].split('_')[1].split('.')[0]
            return int(path_)
        return None
    
    def prefetchers_match(self, other: 'ResultsPath'):
        """Return whether the prefetchers used match the other
        ChampSim path.
        """
        return (self.l1_prefetcher == other.l1_prefetcher
                and self.l2_prefetcher == other.l2_prefetcher
                and self.llc_prefetcher == other.llc_prefetcher)
    
    
class PCStatsPath(ResultsPath):
    """Class for handling PC stats. Currently identical
    to ResultsPath.
    """
    def __init__(self, path: str):
        assert path.endswith('.csv'), \
            'Wrong file extension for PCStatsPath, should be .csv'
        super().__init__(path)