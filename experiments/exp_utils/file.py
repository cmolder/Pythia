import os
import glob
from collections import defaultdict
from typing import Optional, Tuple


class ChampsimFile():
    """Base class for handling files relating to ChampSim.
    
    Inherited by:
        ChampsimResultsFile: Paths for results paths (.txt)
            ChampsimStatsFile: Paths for per-PC/address statistics 
                               (_(l1d|l2c|llc).txt)
        ChampsimTraceFile: Paths for traces (.trace.gz)
    """
    def __init__(self, path: str):
        self.path = path
        self.full_trace = ChampsimFile._get_full_trace(
            path)  # Trace+simpoint together
        self.trace, self.simpoint = ChampsimFile._get_trace(
            path)  # Trace+simpoint separately

    @staticmethod
    def _get_full_trace(path: str) -> str:
        """Helper function to get the full trace
        from a path string."""
        return (os.path.basename(path).split('-')[0].replace('.trace', ''))

    @staticmethod
    def _get_trace(path: str) -> str:
        """Helper function to get the trace + simpoint
        from a path string."""
        tokens = ChampsimFile._get_full_trace(path).split('_')
        if len(tokens) == 1:  # GAP, e.g. bfs
            return tokens[0], None
        if len(tokens) == 2:  # SPEC 06/17, e.g. astar_313B, 603.bwaves_s
            return tokens[0], tokens[1]
        if len(tokens) == 3:  # Cloudsuite, e.g. cassandra_phase0_core0
            # Trace: name_core; Simpoint: phase
            return tokens[0] + '_' + tokens[2], tokens[1]


class ChampsimTraceFile(ChampsimFile):
    """Class for handling trace path strings (.trace.gz)
    """
    def __init__(self, path: str):
        assert path.endswith('.trace.xz') \
               or path.endswith('.trace.gz') \
               or path.endswith('.trace'), \
               ('Wrong file extension for ChampsimTraceFile, should be'
                '.trace or .trace.(g|x)z')
        super.__init__(path)


class ChampsimResultsFile(ChampsimFile):
    """Class for handling ChampSim result files (.txt)
    
    TODO: Build from actual output, instead of file name?
    """
    def __init__(self, path: str):
        assert path.endswith('.txt'), \
            'Wrong file extension for ResultsPath, should be .txt'
        super().__init__(path)

        # Prefetchers
        # Prefetcher names (l1, l2, llc)
        self.l1_prefetcher, self.l2_prefetcher, self.llc_prefetcher = \
            ChampsimResultsFile._get_prefetcher_names(path)
        # Prefetcher degrees (l2, llc)
        self.l2_prefetcher_degree, self.llc_prefetcher_degree = \
            ChampsimResultsFile._get_prefetcher_degrees(path)

        # Pythia
        self.pythia_level_threshold = \
            ChampsimResultsFile._get_pythia_level_threshold(path)
        self.pythia_features = \
            ChampsimResultsFile._get_pythia_features(path)

        # Champsim
        self.champsim_seed = \
            ChampsimResultsFile._get_champsim_seed(path)

    def read(self) -> Optional[dict]:
        """Read and parse ChampSim results file, returning
        a dictionary of data.
        """
        expected_keys = ('ipc', 'dram_bw_epochs', 'L1D_useful', 'L2C_useful',
                         'LLC_useful', 'L1D_useless', 'L2C_useless',
                         'LLC_useless', 'L1D_issued_prefetches',
                         'L2C_issued_prefetches', 'LLC_issued_prefetches',
                         'L1D_total_miss', 'L2C_total_miss', 'LLC_total_miss',
                         'L1D_load_miss', 'L2C_load_miss', 'LLC_load_miss',
                         'L1D_rfo_miss', 'L2C_rfo_miss', 'LLC_rfo_miss',
                         'seed', 'kilo_inst')
        data = {}

        # Optional data (not checked in expected_keys)
        pythia_dyn_level = False
        data['pythia_features'] = []
        data['pythia_level_threshold'] = None
        data['pythia_low_conf_prefetches'] = None
        data['pythia_high_conf_prefetches'] = None
        cpu = 0  # TODO : Implement multi-core support
        with open(self.path, 'r') as f:
            for line in f:
                if 'ChampSim seed:' in line:
                    data['seed'] = int(line.split()[3])
                if 'Finished CPU' in line:
                    data['ipc'] = float(line.split()[9])
                    data['kilo_inst'] = int(line.split()[4]) / 1000
                if 'DRAM_bw_pochs' in line:
                    data['dram_bw_epochs'] = int(line.split()[1])
                if 'scooby_enable_dyn_level 1' in line:
                    pythia_dyn_level = True
                if 'scooby_dyn_level_threshold' in line and pythia_dyn_level:
                    data['pythia_level_threshold'] = float(line.split()[1])
                if 'scooby_low_conf_pref' in line:
                    data['pythia_low_conf_prefetches'] = int(line.split()[1])
                if 'scooby_high_conf_pref' in line:
                    data['pythia_high_conf_prefetches'] = int(line.split()[1])
                if 'le_featurewise_active_features' in line:
                    data['pythia_features'] = \
                        tuple((line.replace(',','').split())[1:])

                # Per-core, cache-level statistics
                if f'Core_{cpu}' not in line:
                    continue
                line = line.strip()

                for level in ['L1D', 'L2C', 'LLC']:
                    if f'{level}_load_miss' in line:
                        data[f'{level}_load_miss'] = int(line.split()[1])
                    elif f'{level}_RFO_miss' in line:
                        data[f'{level}_rfo_miss'] = int(line.split()[1])
                    elif f'{level}_total_miss' in line:
                        data[f'{level}_total_miss'] = int(line.split()[1])
                    elif f'{level}_prefetch_useful' in line:
                        data[f'{level}_useful'] = int(line.split()[1])
                    elif f'{level}_prefetch_useless' in line:
                        data[f'{level}_useless'] = int(line.split()[1])
                    elif f'{level}_prefetch_issued' in line:
                        data[f'{level}_issued_prefetches'] = \
                            int(line.split()[1])

        if not all(key in data for key in expected_keys):
            return None
        return data

    def prefetchers_match(self, other: 'ChampsimResultsFile') -> bool:
        """Return whether the prefetchers used match the other
        ChampSim path.
        """
        return (self.l1_prefetcher == other.l1_prefetcher
                and self.l2_prefetcher == other.l2_prefetcher
                and self.llc_prefetcher == other.llc_prefetcher)

    def get_prefetcher_at_level(self, level: str) -> Optional[str]:
        """Get the prefetcher(s) at 
        the specified cache level (l1/l1d, l2/l2c, llc)
        """
        level = level.lower()
        if 'l1' in level:  # l1 / l1d
            return self.l1_prefetcher
        elif 'l2' in level:  # l2 / l2c
            return self.l2_prefetcher
        elif 'llc' in level:
            return self.llc_prefetcher
        return None

    def get_prefetcher_degree_at_level(self, level: str) -> \
        Tuple[Optional[int], ...]:
        """Get the prefetcher degree(s) at 
        the specified cache level (l1/l1d, l2/l2c, llc)
        """
        level = level.lower()
        if 'l1' in level:  # l1 / l1d
            return (None, )  # Not implemented for L1
        elif 'l2' in level:  # l2 / l2c
            return self.l2_prefetcher_degree
        elif 'llc' in level:
            return self.llc_prefetcher_degree
        return None

    def get_all_prefetchers(self) -> Tuple[str, str, str]:
        """Get a tuple of the prefetchers at all levels.
        """
        return (self.l1_prefetcher, self.l2_prefetcher, self.llc_prefetcher)

    def get_all_prefetcher_degrees(self) -> \
        Tuple[Optional[int], Optional[int]]:
        """Get a tuple of the prefetcher degrees at all levels (l2, llc)
        """
        return (self.l2_prefetcher_degree, self.llc_prefetcher_degree)

    def get_all_variants(self) -> Tuple:
        """Get the key for the variant properties (degree, Pythia 
        variables, etc.), mainly for indexing inside a Directory.
        """
        variants = {
            'l2_prefetcher_degree': self.l2_prefetcher_degree,
            'llc_prefetcher_degree': self.llc_prefetcher_degree,
            'pythia_level_threshold': self.pythia_level_threshold,
            'pythia_features': self.pythia_features,
        }
        return tuple(sorted(variants.items()))

    @staticmethod
    def get_baseline_prefetchers() -> Tuple[str, str, str]:
        """Get a tuple of all prefetchers for the baseline.
        """
        return ('no', 'no', 'no')

    @staticmethod
    def get_baseline_variants() -> Tuple:
        """Get the key for the variants for the baseline. The
        seed is a variant, but must be specified explicitly.
        """
        variants = {
            'l2_prefetcher_degree': (None, ),
            'llc_prefetcher_degree': (None, ),
            'pythia_level_threshold': None,
            'pythia_features': None,
        }
        return tuple(sorted(variants.items()))

    @staticmethod
    def _get_prefetcher_names(path: str) -> Tuple[str, ...]:
        """Get the prefetcher(s) names.
        """
        # (l1_prefethcher, l2_prefetcher, llc_prefetcher)
        p = os.path.basename(path).split('-')[2:5]
        for i in range(len(p)):
            if p[i] not in ['no', 'multi_pc_trace']:
                p[i] = (p[i].replace('spp_dev2', 'sppdev2').replace(
                    'scooby_double', 'scoobydouble').split('_')[0])
                p[i] = (p[i].replace(',', '_').replace('sppdev2',
                                                       'spp_dev2').replace(
                                                           'scoobydouble',
                                                           'scooby_double'))
        return (*p, )

    @staticmethod
    def _get_prefetcher_degrees(path: str) \
        -> Tuple[Tuple[Optional[int], ...], ...]:
        """Get the prefethcer(s) degrees.
        """
        # (l2_prefetcher, llc_prefetcher)
        p = os.path.basename(path).split('-')[3:5]
        d = [None for pref in p]
        for i in range(len(p)):
            if p[i] not in ['no', 'multi_pc_trace']:
                d[i] = (p[i].replace('spp_dev2', 'sppdev2').replace(
                    'scooby_double', 'scoobydouble').split('_')[1])
                d[i] = tuple(
                    (None if d == 'na' else int(d)) for d in d[i].split(','))
        return (*d, )

    @staticmethod
    def _get_pythia_level_threshold(path: str) -> Optional[float]:
        """Get the level threshold for Pythia,
        if one of the prefetchers is using it.
        """
        if not any(p == 'scooby'
                   for p in ChampsimResultsFile._get_prefetcher_names(path)):
            return None

        path = os.path.basename(path)
        if 'threshold' in path:
            return float(
                path[path.index('threshold'):].split('_')[1].split('.')[0])
        return None

    @staticmethod
    def _get_pythia_features(path: str) -> Optional[Tuple[int, ...]]:
        """Get the featuers for Pythia,
        if it is being used.
        """
        if not any(p == 'scooby'
                   for p in ChampsimResultsFile._get_prefetcher_names(path)):
            return None

        path = os.path.basename(path)
        if 'features' in path:
            path_ = (path[path.index('features'):].split('_')[1].replace(
                'seed', '').split('.')[0])
            return tuple(int(f) for f in path_.split(','))

    @staticmethod
    def _get_champsim_seed(path: str) -> Optional[int]:
        """Get the ChampSim seed used.
        """
        path = os.path.basename(path)
        if 'seed' in path:
            path_ = path[path.index('seed'):].split('_')[1].split('.')[0]
            return int(path_)
        return None


class ChampsimStatsFile(ChampsimResultsFile):
    """Class for handling a per-PC/address statistics file.
    The optional parameter <index> can track what is being
    tracked (PCs or addresses).
    """
    def __init__(self, path: str, index: str = 'pc'):
        assert (path.endswith('_l1d.txt') or
                path.endswith('_l2c.txt') or
                path.endswith('_llc.txt')), \
                ('Wrong file extension for ChampsimStatsFile, ',
                 'should be _(l1d|l2c|llc).txt')
        super().__init__(path)

        # l1d, l2c, llc, inferred from path
        self.level = path.split('_')[-1].split('.')[0]
        self.index = index

    def read(self) -> dict:
        """Read and parse ChampSim statistics file, returning
        a dictionary of data.
        """
        stats = defaultdict(dict)
        with open(self.path, 'r') as f:
            for line in f:
                index = line.split()[0]
                stats[index]['useful'] = int(line.split()[1])
                stats[index]['useless'] = int(line.split()[2])
        return stats


class ChampsimDirectory():
    """Base class for handling directories relating to ChampSim.
    
    Inherited by:
        ChampsimResultsDirectory: Directories for results paths (.txt)
            ChampsimStatsDirectory: Directories for per-PC/address 
                                    statistics (_(l1d|l2c|llc).txt)
    """
    def __init__(self, path: str):
        self.paths = []
        self.files = []
        self.baselines: dict = {}  # Points to the baseline ChampsimFile
        # for (full_trace, seed)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> dict:
        return self.files[idx]

    def __iter__(self):
        return iter(self.files)

    def _gather_files(self) -> None:
        raise NotImplementedError(
            'Base class ChampsimDirectory does not implement _gather_files')


class ChampsimResultsDirectory(ChampsimDirectory):
    """A directory of ChampSim result files (ChampsimResultFile).
    
    Organizes files into a hierarchical dictionary:
        trace -> seed -> prefetchers -> variants (degree, Pythia 
        variables, etc.)
    """
    def __init__(self, path: str):
        super().__init__(path)
        self.paths = glob.glob(os.path.join(path, '*.txt'))
        self._gather_files()

    def get_baseline(self, trace: str, seed: int) -> ChampsimResultsFile:
        return self.baselines[(trace, seed)]

    def _gather_files(self) -> None:
        for path in self.paths:
            file = ChampsimResultsFile(path)
            self.files.append(file)

            if (file.get_all_prefetchers() ==
                    ChampsimResultsFile.get_baseline_prefetchers()):
                self.baselines[(file.full_trace, file.champsim_seed)] = file


class ChampsimStatsDirectory(ChampsimResultsDirectory):
    """A directory of ChampSim per-PC address/stats files 
    (ChampsimStatsFile).
    
    Organizes files into a hierarchical dictionary:
        trace -> seed -> prefetchers -> variants (degree, Pythia
        variables, etc.)
    """
    def __init__(self, path: str, level: str = 'llc'):
        super().__init__(path)
        self.paths = glob.glob(os.path.join(path, f'*_{self.level}.txt'))
        self.level = level
        self._gather_files()
