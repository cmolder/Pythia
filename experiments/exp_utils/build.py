"""
Utility functions for building versions of Pythia fork of ChampSim.

Author: Carson Molder
"""

# Try not to import anything outside Python default libraries.
import os
import shutil
from exp_utils import defaults


def change_llc_sets(cacheh_path: str, num_cpus: int, num_sets: int) -> None:
    """Replace the number of LLC sets in the ChampSim LLC definition."""
    print(f'Changing LLC sets in inc/cache.h to NUM_CPUS*{num_sets} '
          f'(effectively {num_cpus} * {num_sets}, '
          f'{num_cpus*num_sets*16 / 1024} KB)...')

    replacement = ''
    with open(cacheh_path, 'rt') as f:
        for line in f:
            if 'LLC_SET' in line:
                line = f'#define LLC_SET NUM_CPUS*{num_sets}\n'
            replacement += line

    with open(cacheh_path, 'wt') as f:
        print(replacement, file=f)


def build_binary(branch_pred: str, l1d_pref: str, l2c_pref: str, llc_pref: str,
                 llc_repl: str, num_cpus: int) -> None:
    """System call to build the binary."""
    cmd = (f'./build_champsim.sh {branch_pred} {l1d_pref} '
           f'{l2c_pref} {llc_pref} {llc_repl} {num_cpus}')
    print(f'Calling "{cmd}"...')
    os.system(cmd)


def backup_file(path: str) -> None:
    """Back up a file."""
    if os.path.exists(path):
        print(f'Backing up {path}...')
        shutil.copyfile(path, path + '.bak')


def restore_file(path: str) -> None:
    """Restore a file from backup."""
    if os.path.exists(path + '.bak'):
        print(f'Restoring {path} from backup...')
        shutil.copyfile(path + '.bak', path)
        os.remove(path + '.bak')


def move_file(old_path: str, new_path: str) -> None:
    """Move a file from <old_path> to <new_path>"""
    if os.path.exists(old_path):
        print(f'Moving {old_path} to {new_path}...')
        shutil.move(old_path, new_path)
    else:
        print(f'[DEBUG] {old_path} does not exist, cannot move to {new_path}.')


def build_config(num_cpus: int,
                 branch_pred: str = 'perceptron',
                 l1d_pref: str = 'no',
                 l2c_pref: str = 'no',
                 llc_pref: str = 'no',
                 llc_repl: str = 'ship',
                 llc_num_sets: int = 2048) -> None:
    """Build a configuration of ChampSim.
    
    Parameters:
        llc_pref_fn: string
            The LLC prefetch function to build.
        
        num_cpus: int
            The number of cores to configure the binary to run.
        
        llc_num_sets: int (optional)
            The number of LLC sets to configure the binary to have.
    """
    print(f'=== Building ChampSim binary, {num_cpus} '
          f'core{"s" if num_cpus > 1 else ""}, {llc_num_sets} LLC sets ===')

    # Backup files
    backup_file('./inc/cache.h')  # Backup original cache.h file
    old_binary = defaults.binary_base.format(branch_pred=branch_pred,
                                             l1d_pref=l1d_pref,
                                             l2c_pref=l2c_pref,
                                             llc_pref=llc_pref,
                                             llc_repl=llc_repl,
                                             n_cores=num_cpus)
    new_binary = (old_binary +
                  defaults.llc_sets_suffix.format(llc_n_sets=llc_num_sets))

    # Backup original binary (if one clashes with ChampSim's output)
    backup_file(old_binary)

    # Modify files and build
    # Change cache.h file to accomodate desired number of sets
    change_llc_sets('./inc/cache.h', num_cpus, llc_num_sets)
    # Build ChampSim with modified cache.h
    build_binary(branch_pred, l1d_pref, l2c_pref, llc_pref, llc_repl, num_cpus)
    move_file(old_binary, new_binary)  # Rename new binary to reflect changes.
    os.chmod(new_binary, 0o755)  # Make binary executable

    # Restore backups
    restore_file('./inc/cache.h')  # Restore original cache.h file.
    restore_file(old_binary)  # Restore original binary (if one exists)
