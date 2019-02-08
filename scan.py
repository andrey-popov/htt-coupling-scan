#!/usr/bin/env python

"""Scans over coupling modifier for various signal hypotheses.

The scan is adaptive.  A thread pool is utilized.
"""

from __future__ import print_function, division
import argparse
from contextlib import closing
import itertools
import json
from operator import itemgetter
import os
import Queue
import subprocess
import sys
import threading
from uuid import uuid4

import numpy as np

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True


class CLs(object):
    """Auxiliar class to describe observed and expected CLs.
    
    This is a simple wrapper around a list of six CLs values.
    """
    
    __slots__ = ['values']
    
    
    def __init__(self, values):
        """Initialize from raw list.
        
        Arguments:
            values:  List of six CLs values, first observed one, then
                five expected values in the order from -2 sigma to
                +2 sigma.
        """
        
        self.values = values
    
    
    @classmethod
    def from_obsexp(cls, obs, exp):
        
        return cls([obs] + exp)
    
    
    def __getitem__(self, index):
        """Access elements of underlying raw list."""
        
        return self.values[index]
    
    
    def obs(self):
        """Return observed CLs."""
        
        return self.values[0]
    
    
    def exp(self, index):
        """Return expected CLs.
        
        Arguments:
            index:  Integer number from -2 to +2 that identifies the
                quantile of the expected distribution of the test
                statistic.
        
        Return value:
            Expected CLs value for the given index.
        """
        
        return self.values[index + 3]


class FailedFitError(Exception):
    """An exception to indicate fit failure."""
    
    pass


class Calc(object):
    """Class to compute CLs.
    
    Internally calls combine and reads its output.
    """
    
    def __init__(self, card, label=''):
        """Initialize from a combine data card.
        
        Arguments:
            card:  Path to combine data card.
            label:  Optional label that will be added to the name of
                ROOT file with workspace.
        
        Creates a file with workspace for the given data card.  User
        should call method close to delete this file after the
        computation is over.
        """
        
        # Construct the workspace
        self.workspace_path = 'workspace_{}{}.root'.format(label, uuid4().hex)
        
        with open(os.devnull, 'w') as devnull:
            subprocess.check_call(
                [
                    'text2workspace.py', card, '-P',
                    'CombineHarvester.CombineTools.InterferencePlusFixed:interferencePlusFixed',
                    '-o', self.workspace_path
                ],
                stdout=devnull
            )
    
    
    def __call__(self, g):
        """Compute CLs values for given g.
        
        Raise a FailedFitError if the computation fails.
        
        Arguments:
            g:  Value of the coupling modifier.
        
        Return value:
            An instance of class CLs with results for the given g.
        """
        
        # Compute CLs for the given value of the coupling.  Suppress the
        # warning about multiple POI in the model.
        uid = uuid4().hex
        
        with open(os.devnull, 'w') as devnull:
            command = [
                'combine', '-M', 'AsymptoticLimits', '-d', self.workspace_path,
                '--setParameters', 'g={}'.format(g), '--freezeParameters', 'g',
                '--singlePoint', '1', '-n', uid, '--X-rtd', 'MINIMIZER_analytic',
                '--picky', '--cminPreFit', '1'
            ]
            proc = subprocess.Popen(command, stdout=devnull, stderr=subprocess.PIPE)
            
            for line in proc.stderr:
                if "ModelConfig 'ModelConfig' defines more than one parameter of interest" in line:
                    continue
                
                sys.stderr.write(line)
            
            if proc.wait() != 0:
                raise subprocess.CalledProcessError(proc.returncode, command)
        
        
        # Extract CLs values from ROOT file created by combine
        results = []
        
        resfile_name = 'higgsCombine{}.AsymptoticLimits.mH120.root'.format(uid)
        resfile = ROOT.TFile(resfile_name)
        tree = resfile.Get('limit')
        
        if tree.GetEntries() != 6:
            os.remove(resfile_name)
            raise FailedFitError
        
        for entry in tree:
            results.append((entry.quantileExpected, entry.limit))
        
        resfile.Close()
        os.remove(resfile_name)
        
        
        results.sort(key=itemgetter(0))
        return CLs([r[1] for r in results])
    
    
    def close(self):
        """Delete the file with workspace."""
        
        os.remove(self.workspace_path)



class Scanner(object):
    """Class to perform a scan over coupling modifier.
    
    The scan can be performed over a predetermined set of values of g or
    in an adaptive manner.  All CLs values computed are added to
    internal list cls, which contains tuples of values of g and CLs
    objects.  The list is sorted by g.
    
    The scan is done using a thread pool of size specified at the
    initialization.
    """
    
    def __init__(self, card, num_threads=1):
        """Initialize from combine data card.
        
        Arguments:
            card:  Path to combine data card.
            num_threads:  Number of threads to be used in the scan.
        """
        
        self.calc = Calc(card)
        self.cls = []
        
        self.num_threads = num_threads
    
    
    def close(self):
        """Perform cleanup."""
        
        self.calc.close()
    
    
    def scan(self, g_values):
        """Scan over given values of the coupling modifier.
        
        Add all computed CLs values to the internal list.
        
        Arguments:
            g_values:  Iterable with values of g to be used.
        
        Return value:
            List of tuples of values of g and corresponding CLs objects
            obtained in the current run.  The list is sorted by g.
        
        Run in multiple threads according to the thread multiplicity
        provided at initialization.
        """
        
        g_queue = Queue.Queue()
        
        for g in g_values:
            g_queue.put(g)
        
        threads = []
        cls = []
        
        for i in range(self.num_threads):
            t = threading.Thread(target=self._scan_worker, args=(g_queue, cls))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        cls.sort(key=itemgetter(0))
        
        self.cls.extend(cls)
        self.cls.sort(key=itemgetter(0))
        
        return cls
    
    
    def scan_auto(self, g_range=(0., 3.), step_coarse=0.1, step_fine=0.01, level=0.05):
        """Run an adaptive scan over coupling modifier.
        
        Scan over the given range using a coarse step.  Identify regions
        in which any CLs value (observed or any expected alike)
        approaches the given level.  Run another scan with a finer grid
        in these regions.
        
        Arguments:
            g_range:  Range for the coupling modifier.
            step_coarse, step_fine:  Steps for the grid to be used in
                the coarse and fine scans.
            level:  Level for CLs that defines the regions of interest.
        
        Return value:
            List of all CLs values computed over the lifetime of this
            object.  Follows the same format as the return value of
            method scan.
        """
        
        # Perform a coarse scan
        self.scan(np.arange(
            g_range[0], g_range[1] + step_coarse / 2, step_coarse
        ))
        
        
        # Find regions of interest, where CLs (observed or any of the
        # five versions of the expected one) approaches the given level.
        # The regions are represented with indices of bins of the coarse
        # grid.
        roi_indices = set()
        
        for cls_type in range(6):
            for i in range(len(self.cls) - 1):
                cls_cur = self.cls[i][1][cls_type]
                cls_next = self.cls[i + 1][1][cls_type]
                
                # Add the current bin if (CLs - level) changes sign
                # inside it
                if (cls_cur > level) != (cls_next > level):
                    roi_indices.add(i)
                
                # When cls_cur approximately equals level, add both bins
                # that touch the point.  This is done to account for
                # potential numerical inaccuracy in CLs.
                if i > 0:
                    cls_prev = self.cls[i - 1][1][cls_type]
                    scale = min(abs(cls_next - cls_cur), abs(cls_prev - cls_cur))
                    
                    if abs(cls_cur - level) < scale / 10:
                        roi_indices.add(i - 1)
                        roi_indices.add(i)
        
        
        # Create a fine grid within the identified regions of interest.
        # Skip bin edges as they have already been included in the
        # coarse grid.
        g_values = []
        
        for i in roi_indices:
            g_values.extend(list(np.arange(
                self.cls[i][0] + step_fine, self.cls[i + 1][0] - step_fine / 2, step_fine
            )))
        
        self.scan(g_values)
        
        return self.cls
    
    
    def _scan_worker(self, g_queue, cls):
        """Worker to perform the scan in a thread pool."""
        
        while True:
            try:
                g = g_queue.get_nowait()
            except Queue.Empty:
                break
            
            try:
                cls.append((g, self.calc(g)))
            except FailedFitError:
                pass


def scan_worker(hypotheses_queue, results):
    """Worker for thread pool to run over signal hypotheses."""
    
    while True:
        try:
            label, card = hypotheses_queue.get_nowait()
        except Queue.Empty:
            break
        
        with closing(Scanner(card, num_threads=1)) as scanner:
            cls = scanner.scan_auto()
            results[label] = [[e[0]] + e[1].values for e in cls]



if __name__ == '__main__':
    
    arg_parser = argparse.ArgumentParser(epilog=__doc__)
    arg_parser.add_argument('cards', help='Directory with combine data cards')
    arg_parser.add_argument(
        '-j', '--jobs', type=int, default=16,
        help='Number of cards to process in parallel'
    )
    arg_parser.add_argument(
        '-o', '--output', default='scan.json',
        help='Name for output JSON file'
    )
    args = arg_parser.parse_args()
    
    ROOT.gROOT.SetBatch(True)
    
    
    hypotheses = Queue.Queue()
    
    for cp, mass, width in itertools.product(
        ['A', 'H'], ['400', '500', '600', '750'], ['2p5', '5', '10', '25', '50']
    ):
        hypotheses.put((
            '{}-m{}-relW{}'.format(cp, mass, width),
            os.path.join(args.cards, '{}ToTT-m{}-relW{}.txt'.format(cp, mass, width))
        ))
    
    
    results = {}
    threads = []
    
    for i in range(args.jobs):
        t = threading.Thread(target=scan_worker, args=(hypotheses, results))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    
    with open(args.output, 'w') as f:
        json.dump(results, f)
