"""Utility classes and functions for working with ChampSim files.

Author: Carson Molder
"""

import os
import glob
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple


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
        # Trace + simpoint together
        self.full_trace = ChampsimFile.__get_full_trace(path)
        # Trace + simpoint separately
        self.trace, self.simpoint = ChampsimFile.__get_trace(path)

    @staticmethod
    def __get_full_trace(path: str) -> str:
        """Helper function to get the full trace
        from a path string."""

        # Universally parses SPEC '06, '17, GAP.
        return (os.path.basename(path)
                .split('-')[0]
                .replace('.xz', '')
                .replace('.gz', '')
                .replace('.trace', ''))

    @staticmethod
    def __get_trace(path: str) -> str:
        """Helper function to get the trace + simpoint
        from a path string."""
        tokens = ChampsimFile.__get_full_trace(path).split('_')
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
        assert (path.endswith('.trace')
                or path.endswith('.gz')
                or path.endswith('.xz')), (
            'Wrong file extension for ChampsimTraceFile, should be'
            '.trace or .(g|x)z')
        super().__init__(path)


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
            ChampsimResultsFile.__get_prefetcher_names(path)
        # Prefetcher degrees (l2, llc)
        self.l1_prefetcher_degree, self.l2_prefetcher_degree, self.llc_prefetcher_degree = \
            ChampsimResultsFile.__get_prefetcher_degrees(path)

        # Pythia
        self.pythia_level_threshold = \
            ChampsimResultsFile.__get_pythia_level_threshold(path)
        self.pythia_features = \
            ChampsimResultsFile.__get_pythia_features(path)

        # Champsim
        self.champsim_seed = \
            ChampsimResultsFile.__get_champsim_seed(path)

    def read(self) -> Optional[Dict[str, Any]]:
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
        data['pythia_features'] = None
        data['pythia_pooling'] = None
        data['pythia_level_threshold'] = None
        data['pythia_low_conf_prefetches'] = 0
        data['pythia_high_conf_prefetches'] = 0
        data['pythia_action_called'] = 0
        cpu = 0  # TODO : Implement multi-core support
        with open(self.path, 'r') as path_f:
            for line in path_f:
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
                        tuple((line.replace(',', '').split())[1:])
                if 'le_featurewise_pooling_type' in line:
                    pythia_pooling = int(line.split()[1])
                    if pythia_pooling == 1:
                        data['pythia_pooling'] = 'sum'
                    else:
                        data['pythia_pooling'] = 'max'
                if 'learning_engine_featurewise.action.called' in line:
                    data['pythia_action_called'] = int(line.split()[1])

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
            # FIXME: Where is pythia_pooling?
            #'pythia_pooling': self.pythia_pooling,
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
    def __merge_names(prefetcher_name):
        """TODO: Docstring
        """
        return (prefetcher_name.replace('bop_orig', 'boporig')
                               .replace('spp_dev2', 'sppdev2')
                               .replace('scooby_double', 'scoobydouble')
                               .replace('pc_trace', 'pctrace')
                               .replace('from_file', 'fromfile'))

    @staticmethod
    def __unmerge_names(prefetcher_name):
        """TODO: Docstring
        """
        return (prefetcher_name.replace(',', '_')
                               .replace('boporig', 'bop_orig')
                               .replace('sppdev2', 'spp_dev2')
                               .replace('scoobydouble', 'scooby_double')
                               .replace('pctrace', 'pc_trace')
                               .replace('fromfile', 'from_file'))

    @staticmethod
    def __get_prefetcher_names(path: str) -> Tuple[str, str, str]:
        """Get the prefetcher(s) names.

        TODO: Docstring
        """
        # (l1_prefethcher, l2_prefetcher, llc_prefetcher)
        path = os.path.splitext(os.path.basename(path))[0]





        if 'l1dpf' in path:
            l1d_prefetcher = path[path.index('l1dpf'):].split('-')[0]
            # Remove extra underscores
            l1d_prefetcher = ChampsimResultsFile.__merge_names(l1d_prefetcher)
            # Split prefetchers and degrees
            l1d_prefetcher = l1d_prefetcher.split('_')[-2]
            # Re-add extra underscores
            l1d_prefetcher = ChampsimResultsFile.__unmerge_names(
                l1d_prefetcher)
        else:
            l1d_prefetcher = 'no'

        if 'l2pf' in path:
            l2_prefetcher = path[path.index('l2pf'):].split('-')[0]
            l2_prefetcher = ChampsimResultsFile.__merge_names(l2_prefetcher)
            l2_prefetcher = l2_prefetcher.split('_')[-2]
            l2_prefetcher = ChampsimResultsFile.__unmerge_names(l2_prefetcher)
        else:
            l2_prefetcher = 'no'

        if 'llcpf' in path:
            llc_prefetcher = path[path.index('llcpf'):].split('-')[0]
            llc_prefetcher = ChampsimResultsFile.__merge_names(llc_prefetcher)
            llc_prefetcher = llc_prefetcher.split('_')[-2]
            llc_prefetcher = ChampsimResultsFile.__unmerge_names(
                llc_prefetcher)
        else:
            llc_prefetcher = 'no'

        return l1d_prefetcher, l2_prefetcher, llc_prefetcher

    @staticmethod
    def __get_prefetcher_degrees(path: str) \
            -> Tuple[Tuple[Optional[int], ...], ...]:
        """Get the prefethcer(s) degrees.

        TODO: Docstring
        """
        # (l1_prefetcher, l2_prefetcher, llc_prefetcher)
        path = os.path.splitext(os.path.basename(path))[0]

        if 'l1dpf' in path:
            l1d_prefetcher = path[path.index('l1dpf'):].split('-')[0]
            # Remove extra underscores
            l1d_prefetcher = ChampsimResultsFile.__merge_names(l1d_prefetcher)
            # Split prefetchers and degrees
            l1d_degree = l1d_prefetcher.split('_')[-1]
            # Convert degree into a tuple
            l1d_degree = tuple(
                (None if d == '0' else int(d))
                for d in l1d_degree.split(','))
        else:
            l1d_degree = (None,)

        if 'l2pf' in path:
            l2_prefetcher = path[path.index('l2pf'):].split('-')[0]
            l2_prefetcher = ChampsimResultsFile.__merge_names(l2_prefetcher)
            l2_degree = l2_prefetcher.split('_')[-1]
            l2_degree = tuple(
                (None if d == '0' else int(d))
                for d in l2_degree.split(','))
        else:
            l2_degree = (None,)

        if 'llcpf' in path:
            llc_prefetcher = path[path.index('llcpf'):].split('-')[0]
            llc_prefetcher = ChampsimResultsFile.__merge_names(llc_prefetcher)
            llc_degree = llc_prefetcher.split('_')[-1]
            llc_degree = tuple(
                (None if d == '0' else int(d))
                for d in llc_degree.split(','))
        else:
            llc_degree = (None,)

        return l1d_degree, l2_degree, llc_degree

    @staticmethod
    def __get_pythia_level_threshold(path: str) -> Optional[float]:
        """Get the level threshold for Pythia,
        if one of the prefetchers is using it.
        """
        path = os.path.splitext(os.path.basename(path))[0]

        if not any(p == 'scooby'
                   for p in ChampsimResultsFile.__get_prefetcher_names(path)):
            return None

        if 'pythia_threshold' in path:
            pythia_threshold = path[path.index('pythia_threshold'):].split('-')[0]
            pythia_threshold = pythia_threshold.split('_')[-1]
            return float(pythia_threshold.split('.')[0])
        return None

    @staticmethod
    def __get_pythia_features(path: str) -> Optional[Tuple[int, ...]]:
        """Get the featuers for Pythia,
        if it is being used.
        """
        path = os.path.splitext(os.path.basename(path))[0]

        if not any(p == 'scooby'
                   for p in ChampsimResultsFile.__get_prefetcher_names(path)):
            return None

        if 'pythia_features' in path:
            pythia_features = path[path.index('pythia_features'):].split('-')[0]
            pythia_features = pythia_features.split('_')[-1]
            return tuple(int(f) for f in pythia_features.split(','))

    @staticmethod
    def __get_champsim_seed(path: str) -> Optional[int]:
        """Get the ChampSim seed used.
        """
        path = os.path.splitext(os.path.basename(path))[0]
        if 'seed' in path:
            seed = path[path.index('seed'):].split('-')[0]
            seed = seed.split('_')[-1]
            return int(seed)
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
        with open(self.path, 'r') as path_f:
            for line in path_f:
                index = line.split()[0]
                stats[index]['useful'] = int(line.split()[1])
                stats[index]['useless'] = int(line.split()[2])
        return stats


class WeightFile():
    """Class for handling weights files for SimPoints (weights.txt)

    Format of each line:
    <trace_phase> <weight normalized to 1.0>
    """

    def __init__(self, path: str):
        self.weights = defaultdict(dict)
        with open(path) as path_f:
            for line in path_f:
                if line == '\n':
                    continue
                full_trace, weight = line.split()
                dummy = ChampsimFile(full_trace)  # TODO: Remove dummy "file"
                self.weights[dummy.trace][dummy.simpoint] = float(weight)

    def get_trace_weights(self, trace: str) -> Dict[str, float]:
        """TODO: Docstring
        """
        return self.weights[trace]

    def get_simpoint_weight(self, trace: str, simpoint: str) -> float:
        """TODO: Docstring
        """
        return self.weights[trace][simpoint]

    def get_traces(self):
        """TODO: Docstring
        """
        return self.weights.keys()


class ChampsimDirectory():
    """Base class for handling directories relating to ChampSim.

    Inherited by:
        ChampsimResultsDirectory: Directories for results paths (.txt)
            ChampsimStatsDirectory: Directories for per-PC/address
                                    statistics (_(l1d|l2c|llc).txt)
    """

    def __init__(self):
        self.paths = []
        self.files = []

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> dict:
        return self.files[idx]

    def __iter__(self):
        return iter(self.files)

    def __gather_files(self) -> None:
        raise NotImplementedError(
            'Base class ChampsimDirectory does not implement _gather_files')


class ChampsimTraceDirectory(ChampsimDirectory):
    """A directory of ChampSim trace files (ChampsimTraceFile).

    Organizes files into a flat dictionary:
        trace
    """

    def __init__(self, path: str):
        super().__init__()
        self.paths = glob.glob(os.path.join(path, '*.[g|x]z'))
        self.__gather_files()

    def __gather_files(self) -> None:
        for path in self.paths:
            file = ChampsimTraceFile(path)
            self.files.append(file)


class ChampsimResultsDirectory(ChampsimDirectory):
    """A directory of ChampSim result files (ChampsimResultFile).

    Organizes files into a hierarchical dictionary:
        trace -> seed -> prefetchers -> variants (degree, Pythia
        variables, etc.)
    """

    def __init__(self, path: str):
        super().__init__()
        self.paths = glob.glob(os.path.join(path, '*.txt'))
        self.baselines: dict = {}  # Points to the baseline ChampsimFile
        # for (full_trace, seed)
        self.__gather_files()

    def get_baseline(self, trace: str, seed: int) -> ChampsimResultsFile:
        """TODO: Docstring
        """
        return self.baselines[(trace, seed)]

    def __gather_files(self) -> None:
        """TODO: Docstring
        """
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
        self.level = level
        self.paths = glob.glob(os.path.join(path, f'*_{self.level}.txt'))
        self.__gather_files()
