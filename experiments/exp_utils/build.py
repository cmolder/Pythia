"""
Utility functions for building versions of Pythia fork of ChampSim.

Author: Carson Molder
"""

# Try not to import anything outside Python default libraries.
import os
import shutil
from exp_utils import defaults



def change_llc_sets(cacheh_path, num_cpus, num_sets):
    """Replace the number of LLC sets in the ChampSim LLC definition."""
    print(f'Changing LLC sets in inc/cache.h to NUM_CPUS*{num_sets} (effectively {num_cpus} * {num_sets}, {num_cpus*num_sets*16 / 1024} KB)...')

    replacement = ''
    with open(cacheh_path, 'rt') as f:
        for line in f:
            if 'LLC_SET' in line:
                line = f'#define LLC_SET NUM_CPUS*{num_sets}\n'
            replacement += line

    with open(cacheh_path, 'wt') as f:
        print(replacement, file=f)


def build_binary(branch_pred, l1d_pref, l2c_pref, llc_pref, llc_repl, num_cpus):
    """System call to build the binary."""
    cmd = f'./build_champsim.sh {branch_pred} {l1d_pref} {l2c_pref} {llc_pref} {llc_repl} {num_cpus}'
    print(f'Calling "{cmd}"...')
    os.system(cmd)


def backup_file(path):
    """Back up a file."""
    if os.path.exists(path):
        print(f'Backing up {path}...')
        shutil.copyfile(path, path + '.bak')


def restore_file(path):
    """Restore a file from backup."""
    if os.path.exists(path + '.bak'):
        print(f'Restoring {path} from backup...')
        shutil.copyfile(path + '.bak', path)
        os.remove(path + '.bak')


def move_file(old_path, new_path):
    """Move a file from <old_path> to <new_path>"""
    if os.path.exists(old_path):
        print(f'Moving {old_path} to {new_path}...')
        shutil.move(old_path, new_path)
    else:
        print(f'[DEBUG] {old_path} does not exist, cannot move to {new_path}.')


def build_config(num_cpus, 
                 branch_pred='perceptron',
                 l1d_pref='no', l2c_pref='no', llc_pref='no',
                 llc_repl='ship',
                 llc_num_sets=2048):
    """Build a configuration of ChampSim.
    
    Parameters:
        llc_pref_fn: string
            The LLC prefetch function to build.
        
        num_cpus: int
            The number of cores to configure the binary to run.
        
        llc_num_sets: int (optional)
            The number of LLC sets to configure the binary to have.
    """
    print(f'=== Building ChampSim binary, {num_cpus} core{"s" if num_cpus > 1 else ""}, {llc_num_sets} LLC sets ===')

    # Backup files
    backup_file('./inc/cache.h') # Backup original cache.h file
    old_binary = defaults.binary_base.format(
        branch_pred=branch_pred,
        l1d_pref=l1d_pref,
        l2c_pref=l2c_pref,
        llc_pref=llc_pref,
        llc_repl=llc_repl,
        n_cores=num_cpus
    )
    new_binary = old_binary + defaults.llc_sets_suffix.format(llc_n_sets=llc_num_sets)
    backup_file(old_binary)      # Backup original binary (if one clashes with ChampSim's output)

    # Modify files and build
    change_llc_sets('./inc/cache.h', num_cpus, llc_num_sets) # Change cache.h file to accomodate desired number of sets
    build_binary(                                            # Build ChampSim with modified cache.h
        branch_pred,
        l1d_pref, l2c_pref, llc_pref, 
        llc_repl, num_cpus
    )  
    move_file(old_binary, new_binary) # Rename new binary to reflect changes.
    os.chmod(new_binary, 0o755)       # Make binary executable

    # Restore backups
    restore_file('./inc/cache.h') # Restore original cache.h file.
    restore_file(old_binary)      # Restore original binary (if one exists)
    