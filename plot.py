#!/usr/bin/env python

"""Plots observed and expected exclusion regions in (m, g) plane."""

import argparse
from bisect import bisect_left
from collections import namedtuple
from copy import copy
import itertools
import json
import math
from operator import itemgetter
import os

import numpy as np

import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt
from matplotlib.offsetbox import TextArea


class ExclusionBuilder:
    """Class to construct exclusion regions."""
    
    _Crossing = namedtuple('Crossing', ['g', 'up'])
    _Crossing.__doc__ = """Value of g and sign of derivative of CLs(g).

    An auxiliary class to aggreate value of g at which CL(g) crosses a
    threshold and the direction of the crossing, i.e. the sign of the
    derivative of CLs(g).  Attribute up is expected to be set to True
    if the derivative is positive and to False otherwise.
    """


    class _PointWithDerivatives:
        """Auxiliary class describing a point with derivatives.
        
        Used in method _clean_series.  Contains a coordinate and
        (finite) derivatives of a function to the left and to the right
        of that point.
        """
        
        __slots__ = ['x', 'der_left', 'der_right']
        
        def __init__(self, x, der_left, der_right):
            """Initialize from x coordinate and two derivatives."""
            
            self.x = x
            self.der_left = der_left
            self.der_right = der_right
        
        
        @property
        def der_scale(self):
            """Return characteristic scale of derivatives."""
            
            return min(abs(self.der_left), abs(self.der_right))
        
        
        @property
        def is_spike(self):
            """Check if derivative changes sign at this point."""
            
            return (self.der_left > 0.) != (self.der_right > 0.)

    
    def __init__(self, mass_scans):
        """Initialize from scan results for various mass points.
        
        Arguments:
            mass_scans:  List of pairs of values of the mass (converted
                into a float internally) and results of corresponding
                scans.  Results of a scan must be represented with a
                list of lists that contain value of g, observed CLs
                value and five expected ones, in the order from -2 to +2
                sigma.  The results must be sorted in g.
        """
        
        self.masses = np.array([float(s[0]) for s in mass_scans])
        self.scans = np.array([np.asarray(s[1]) for s in mass_scans])
        
        # Make sure the scans are sorted in mass
        sorted_order = self.masses.argsort()
        self.masses = self.masses[sorted_order]
        self.scans = self.scans[sorted_order]
        
        # Full probed range for the coupling
        self.g_range = (
            min(scan[0, 0] for scan in self.scans),
            max(scan[-1, 0] for scan in self.scans)
        )
    
    
    def expected(self, num_sigma):
        """Construct contour(s) for expected exclusion.
        
        The contours correspond to CLs = 0.05.
        
        Arguments:
            num_sigma:  Integer number from -2 to +2 that determines
                the quantile of the expected distribution to be used.
        
        Return value:
            LineCollection with the contour(s).
        """
        
        lines = []
        
        
        # Scan over mass points
        cls_index = num_sigma + 4
        crossings_prev = self._find_crossings(self.scans[0][:, 0], self.scans[0][:, cls_index])
        
        for imass in range(1, len(self.masses)):
            
            # Find crossings for the current mass point and construct a
            # matching to crossings for the previous mass point
            crossings_cur = self._find_crossings(
                self.scans[imass][:, 0], self.scans[imass][:, cls_index]
            )
            matches, unmatched_prev, unmatched_cur = self._match_crossings(
                crossings_prev, crossings_cur
            )
            
            
            # Build segments of the contour CLs = 0.05 from the matched
            # pairs of crossings
            segments = []
            
            for match in matches:
                segments.append((
                    (self.masses[imass - 1], crossings_prev[match[0]].g),
                    (self.masses[imass], crossings_cur[match[1]].g)
                ))
            
            
            # Add additional segments from unmatched crossings
            mass_midpoint = (self.masses[imass - 1] + self.masses[imass]) / 2
            
            for mass, crossings, unmatched in [
                (self.masses[imass - 1], crossings_prev, unmatched_prev),
                (self.masses[imass], crossings_cur, unmatched_cur)
            ]:
                # Find consequitive pairs of up- and down-crossings.
                # They are matched with each other.
                unmatched_list = sorted(list(unmatched))
                
                for ii in range(len(unmatched_list) - 1):
                    i1 = unmatched_list[ii]
                    i2 = unmatched_list[ii + 1]
                    
                    if i2 - i1 == 1 and crossings[i1].up != crossings[i2].up:
                        g_midpoint = (crossings[i1].g + crossings[i2].g) / 2
                        segments.extend([
                            ((mass, crossings[i1].g), (mass_midpoint, g_midpoint)),
                            ((mass_midpoint, g_midpoint), (mass, crossings[i2].g))
                        ])
                        
                        unmatched.remove(i1)
                        unmatched.remove(i2)
                
                
                # There can be unmatched crossings at the boundaries of
                # the g range if the contour enters the region of the
                # scan from outside.  Connect them to the respective
                # boundaries.  A tricky point, however, is when one of
                # the lists of crossings is empty and the other only
                # contains a single crossing.  It should still be
                # connected to a boundary, but it is not clear, which
                # one.  As a heuristic, give a preference to the upper
                # boundary.
                ilast = len(crossings) - 1
                
                if ilast in unmatched:
                    segments.append(
                        ((mass, crossings[ilast].g), (mass_midpoint, self.g_range[1])),
                    )
                    unmatched.remove(ilast)
                
                if 0 in unmatched:
                    segments.append(
                        ((mass_midpoint, self.g_range[0]), (mass, crossings[0].g))
                    )
                    unmatched.remove(0)
                
                
                # Cannot handle other unmatched crossings
                if unmatched:
                    raise RuntimeError(
                        'Cannot handle following unmatched crossings for mass {:g} and {:+d} '
                        'sigma expected CLs: {}.'.format(
                            mass, num_sigma, [crossings[i] for i in sorted(list(unmatched))]
                        )
                    )
            
            
            # Attempt to attach newly created line segments to existing
            # lines
            for segment in segments:
                attached = False
                
                for line in lines:
                    if line[-1] == segment[0]:
                        line.append(segment[1])
                        attached = True
                        break
                
                if not attached:
                    # Start a new line
                    lines.append([segment[0], segment[1]])
            
            
            crossings_prev = crossings_cur
        
        return mpl.collections.LineCollection(lines)
    
    
    def observed(self):
        """Construct observed exclusion region(s).
        
        The region (potentially, a set of disjoint regions) corresponds
        to CLs < 0.05.
        
        Return value:
            A list of polygons that describe the region(s).
        """
        
        polygons = []
        
        # Scan over mass points
        crossings_prev = self._find_crossings(self.scans[0][:, 0], self.scans[0][:, 1])
        
        for imass in range(1, len(self.masses)):
            
            # Find crossings for the current mass point and construct a
            # matching to crossings for the previous mass point
            crossings_cur = self._find_crossings(self.scans[imass][:, 0], self.scans[imass][:, 1])
            matches, unmatched_prev, unmatched_cur = self._match_crossings(
                crossings_prev, crossings_cur
            )
            
            # Nothing to do if there are not crossings
            if not crossings_prev and not crossings_cur:
                continue
            
            
            # Identify vertices of the polygons that describe the
            # excluded region.  Their coordinates are stored in a list,
            # and in the following the vertices are identify by their
            # indices in this list.  The coordinates are pairs
            # (mass_logic, g), where mass_logic is the logical mass
            # coordinate.  It is set to -1 for the previous mass point,
            # +1 for the current one, and 0 for the midpoint.
            # Also find line segments that connect the vertices.  They
            # are repesented by pairs of indices of vertices that the
            # line segments connect.  The direction of the line segments
            # is chosen according to clockwise traversal, i.e. the
            # internal area of the polygon is always to the right of a
            # line segment.  Vertical line segments (along the g axis)
            # are implied and not added explicitly.
            vertices = []
            segments = []
            
            
            # First include vertices and segments from matched crossings
            for match in matches:
                n = len(vertices)
                vertices.extend([
                    (-1, crossings_prev[match[0]].g),
                    (1, crossings_cur[match[1]].g)
                ])
                
                segment = [n, n + 1]
                
                if crossings_prev[match[0]].up == False:
                    # This is a down-crossing, therefore the excluded
                    # region is to the right of the line
                    segment.reverse()
                
                segments.append(segment)
            
            
            # Then process unmatched crossings.  They will result in
            # vertices lying at the mass midpoint (mass_logic = 0).  But
            # segments always start and end at mass_logic = +-1.
            veto_boundaries = set()
            
            for mass_logic, crossings, unmatched in [
                (-1, crossings_prev, unmatched_prev),
                (1, crossings_cur, unmatched_cur)
            ]:
                # Find consequitive pairs of up- and down-crossings.
                # They are matched with each other.
                unmatched_ordered = sorted(list(unmatched))
                
                for ii in range(len(unmatched_ordered) - 1):
                    i1 = unmatched_ordered[ii]
                    i2 = unmatched_ordered[ii + 1]
                    
                    if i2 - i1 != 1 or crossings[i1].up == crossings[i2].up:
                        continue
                    
                    n = len(vertices)
                    g1 = crossings[i1].g
                    g2 = crossings[i2].g
                    
                    vertices.extend([
                        (mass_logic, g1), (0, (g1 + g2) / 2), (mass_logic, g2)
                    ])
                    segment = [n, n + 1, n + 2]
                    
                    # Inverse the direction of the segment if
                    #   NOT (crossing at g1 is up) XOR (mass_logic is 1)
                    # Expression obtained by analyzing full
                    # combinatorics.
                    if (crossings[i1].up == True) == (mass_logic == 1):
                        segment.reverse()
                    
                    segments.append(segment)
                    
                    unmatched.remove(i1)
                    unmatched.remove(i2)
                
                
                # There can be unmatched crossings at the boundaries of
                # the g range if the contour enters the region of the
                # scan from outside.  Connect them to mass midpoints at
                # the respective boundaries.  A tricky point, however,
                # is when one of the lists of crossings is empty and the
                # other only contains a single crossing.  It should
                # still be connected to a boundary, but it is not clear
                # which one.  As a heuristic, give a preference to the
                # upper boundary.
                for boundary in ['gmax', 'gmin']:
                    if boundary == 'gmax':
                        iedge = len(crossings) - 1
                        g_value = self.g_range[1]
                    else:
                        iedge = 0
                        g_value = self.g_range[0]
                    
                    if iedge not in unmatched:
                        continue
                    
                    # Put a vertex at the mass midpoint of the
                    # respective boundary
                    n = len(vertices)
                    vertices.extend([
                        (mass_logic, crossings[iedge][0]),
                        (0, g_value)
                    ])
                    
                    # Add the third vertex so that the full segment
                    # ends at one of the mass boundaries.  This is the
                    # same boundary as for the current unmatched
                    # crossing (i.e. given by mass_logic) if
                    #  (crossing is up) XOR (bounday is upper)
                    # The direction of the resulting three-vertex
                    # segment needs to be inverted if
                    #  NOT (crossing is up) XOR (mass_logic is 1)
                    # Both expressions by analyzing the full
                    # combinators.
                    if (crossings[iedge].up == True) != (boundary == 'gmax'):
                        vertices.append((mass_logic, g_value))
                    else:
                        vertices.append((-mass_logic, g_value))
                    
                    segment = [n, n + 1, n + 2]
                    
                    if (crossings[iedge].up == True) == (mass_logic == 1):
                        segment.reverse()
                    
                    segments.append(segment)
                    
                    veto_boundaries.add(boundary)
                    unmatched.remove(iedge)
                
                
                # Cannot handle other unmatched crossings
                if unmatched:
                    raise RuntimeError(
                        'Cannot handle following unmatched crossings for observed CLs for '
                        'mass {:g}: {}.'.format(
                            self.masses[imass], [crossings[i] for i in sorted(list(unmatched))]
                        )
                    )
            
            
            # If the excluded region touches the full boundary of the
            # g range, add coresponding horizontal lines to the list of
            # segments.  Note that this is only possible if both lists
            # of crossings are not empty.
            if crossings_prev and crossings_cur:
                if 'gmin' not in veto_boundaries:
                    if crossings_prev[0].up == True and crossings_cur[0].up == True:
                        n = len(vertices)
                        vertices.extend([
                            (1, self.g_range[0]), (-1, self.g_range[0])
                        ])
                        segments.append([n, n + 1])
                
                if 'gmax' not in veto_boundaries:
                    if crossings_prev[-1].up == False and crossings_cur[-1].up == False:
                        n = len(vertices)
                        vertices.extend([
                            (-1, self.g_range[1]), (1, self.g_range[1])
                        ])
                        segments.append([n, n + 1])
            
            
            # Among all vertices, find those that are located at the
            # mass boundaries (i.e. mass_logic != 0) and sort them in
            # the clockwise direction, starting from the lower left
            # corner.  They are referred to as endpoints.
            endpoints = [
                i for i in range(len(vertices)) if vertices[i][0] != 0
            ]
            endpoints.sort(key=lambda i: (vertices[i][0], -vertices[i][0] * vertices[i][1]))
            
            
            mass_map = {
                -1: self.masses[imass - 1],
                1: self.masses[imass],
                0: (self.masses[imass - 1] + self.masses[imass]) / 2
            }
            
            # Construct the polygons.  Start from an end point.  If
            # there is a segment starting from this end point, include
            # it into the polygon.  Add the next (unused) end point
            # (i.e. a vertical segment).  Repeat until the algorithm
            # returns to the starting end point.  Start the next
            # polygon from the next unused end point.
            segments.sort(key=itemgetter(0))
            segment_heads = [segment[0] for segment in segments]
            
            cur_endpoint = endpoints[0]
            masked_endpoints = set()
            polygon_vertices = []
            
            while True:
                polygon_vertices.append(cur_endpoint)
                masked_endpoints.add(cur_endpoint)
                
                
                # Try to find a segment that starts from the current
                # endpoint
                isegment = bisect_left(segment_heads, cur_endpoint)
                
                if isegment < len(segments) and segment_heads[isegment] == cur_endpoint:
                    segment = segments[isegment]
                    
                    for ivertex in segment[1:]:
                        polygon_vertices.append(ivertex)
                    
                    # A segment always ends with an endpoint
                    cur_endpoint = segment[-1]
                    masked_endpoints.add(cur_endpoint)
                    
                    # If the polygon has been closed, find physical
                    # coordinates of its vertices and add it to the list
                    # of constructed polygons
                    if cur_endpoint == polygon_vertices[0]:
                        polygon = []
                        
                        for ivertex in polygon_vertices[:-1]:
                            polygon.append(
                                (mass_map[vertices[ivertex][0]], vertices[ivertex][1])
                            )
                        
                        polygons.append(polygon)
                        polygon_vertices = []
                
                
                # Check if there are any unused endpoints.  Otherwise
                # all polygons have been constructed.
                if len(masked_endpoints) == len(endpoints):
                    break
                
                
                # Continue the polygon along the g axis to the next
                # unused endpoint
                i_next_endpoint = endpoints.index(cur_endpoint) + 1
                
                while endpoints[i_next_endpoint] in masked_endpoints:
                    i_next_endpoint += 1
                
                cur_endpoint = endpoints[i_next_endpoint]
            
            
            crossings_prev = crossings_cur
        
        return [mpl.patches.Polygon(polygon) for polygon in polygons]
    
    
    @staticmethod
    def _clean_series(x, y, y_range=(-math.inf, math.inf), der_cutoff=3., knn=5):
        """Clean y(x) series of spikes and constrain it to a y range.
        
        Remove points for which y coordinate falls outside of the given
        range.  Find and remove spikes, which are defined as points at
        which the (finite) derivative changes sign and is large in
        value.  The latter condition is evaluated by comparing the scale
        of the derivative with the median one computed among given
        number of nearest neighbours; their ratio should be above the
        given cutoff value.
        
        Arguments:
            x, y:  Arrays with x and y coordinates.  Must be sorted in
                x coordinate.
            y_range:  Allowed range for y values.
            der_cutoff:  Minimal ratio of the scale of the derivative to
                the local median one to consider the point a spike.
            knn:  Number of nearest neighbours to use in the computation
                of the median scale of the derivative.
        
        Return value:
            Tuple of filtered arrays x and y.
        
        This method is used by _find_crossings to clean CLs(g).
        """
        
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Identify points that lay outside of the y range
        out_of_range = np.logical_or(y < y_range[0], y > y_range[1])
        
        # See if any points survive the selection
        if len(y) - np.sum(out_of_range) == 0:
            return [], []
        
        
        # Look for spikes
        spikes = np.zeros(len(x), dtype=bool)
        
        if der_cutoff > 0:
            
            # Construct a list of points with left and right
            # derivatives
            points = []
            derivative = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
            
            for i in range(1, len(x) - 1):
                points.append((
                    i,
                    ExclusionBuilder._PointWithDerivatives(
                        x[i], derivative[i - 1], derivative[i]
                    )
                ))
            
            
            # Loop over identified points
            for i in range(len(points)):
                cur_index, cur_point = points[i]
                
                # Find up to knn closest neighbours on each side.  The
                # current point is included in the list of neighbours.
                neighbours = []
                
                for j in range(max(i - knn, 0), min(i + knn + 1, len(points))):
                    neighbours.append(points[j][1])
                
                neighbours.sort(key=lambda p: abs(p.x - cur_point.x))
                
                
                # Compute median scale of the derivative for knn nearest
                # neighbours
                der_scales = [n.der_scale for n in neighbours]
                median_der_scale = np.median(der_scales[:knn])
                
                
                # Check if there is a spike above the threshold at the
                # current point
                if cur_point.is_spike and cur_point.der_scale > der_cutoff * median_der_scale:
                    spikes[cur_index] = True
            
            
            # A sanity check: make sure that the fraction of detected
            # points with spikes with respect to all points in the
            # allowed range is small
            num_spikes = np.sum(np.logical_and(spikes, np.logical_not(out_of_range)))
            num_probed =  len(x) - np.sum(out_of_range)
            
            if num_spikes / num_probed > 0.05:
                raise RuntimeError(
                    'Cleaning algorithm has rejected {} / {} points as spikes.'.format(
                        num_spikes, num_probed
                    )
                )
        
        
        # Return cleaned series
        masked = np.logical_or(out_of_range, spikes)
        return x[np.logical_not(masked)], y[np.logical_not(masked)]
    
    
    @staticmethod
    def _find_crossings(x, y, level=0.05):
        """Solve y(x) = level for x.
        
        Find all solutions.  Use a linear approximation to interpolate
        between the given points.
        
        Arguments:
            x, y:  Arrays with x and y coordinates.  Must be sorted in
                x coordinate.
            level:  Desired value for y(x).
        
        Return value:
            List of objects of type Crossing.
        """
        
        x, y = ExclusionBuilder._clean_series(x, y, y_range=(level / 2, level * 2))
        crossings = []
        
        for i in range(len(y) - 1):
            if (y[i] > level) != (y[i + 1] > level) or y[i] == level:
                # Function crosses the given level within the current
                # segment.  Find the crossing point using linear
                # approximation.
                x_cross = x[i] + (x[i + 1] - x[i]) / (y[i + 1] - y[i]) * (level - y[i])
                
                # Determine the sign of the derivative
                upcrossing = ((y[i + 1] > y[i]) == (x[i + 1] > x[i]))
                
                crossings.append(ExclusionBuilder._Crossing(x_cross, upcrossing))
        
        return crossings
    
    
    @staticmethod
    def _match_crossings(crossings1, crossings2):
        """Perform matching between two lists of crossings.
        
        Construct one-to-one correspondence between two lists of
        crossings.  Only crossings of the same type (up or down) can be
        paired.  If the lists contain different numbers of crossings (of
        the same type), there will be unmatched ones.
        
        Arguments:
            crossing1, crossing2:  Two lists of crossings in the format
                returned by method _find_crossings.
        
        Return value:
            Tuple consisting of three elements: list of pairs of indices
            of matched crossings and a set of indices of unmatched
            crossings for each input list.
        """
        
        matches = []
        
        # Consider up- and down-crossings independently
        for up in [True, False]:
            
            # Find which of the lists contains fewer crossings of given
            # type and define aliases accordingly
            c_short = crossings1
            c_long = crossings2
            
            if sum(1 for c in c_short if c.up == up) > sum(1 for c in c_long if c.up == up):
                c_short, c_long = c_long, c_short
                swapped = True
            else:
                swapped = False
            
            
            # Loop over crossings of given type in the shorter list and
            # match them to closest crossings of the same type in the
            # longer list.  Same crossing in the longer list can only
            # be matched to one crossing in the shorter list.
            used_crossings_long = set()
            
            for i in range(len(c_short)):
                if c_short[i].up != up:
                    continue
                
                smallest_dist = math.inf
                j_closest = -1
                
                for j in range(len(c_long)):
                    if c_long[j].up != up or j in used_crossings_long:
                        continue
                    
                    dist = abs(c_long[j][0] - c_short[i][0])
                    
                    if dist < smallest_dist:
                        smallest_dist = dist
                        j_closest = j
                
                if j_closest == -1:
                    # This should not happen unless there is a bug
                    raise RuntimeError('Failed to find a match.')
                
                matches.append((i, j_closest) if not swapped else (j_closest, i))
                used_crossings_long.add(j_closest)
        
        matches.sort(key=itemgetter(0))
        
        
        # Identify unmatched crossings
        unmatched1 = set(range(len(crossings1)))
        unmatched2 = set(range(len(crossings2)))
        
        for match in matches:
            unmatched1.remove(match[0])
            unmatched2.remove(match[1])
        
        return matches, unmatched1, unmatched2


def max_g(cp, mass, rel_width):
    """Compute maximal allowed value of the coupling scale factor.
    
    Computed value corresponds to a 100% branching ratio of H->tt.
    
    Arguments:
        cp:  CP state, 'A' or 'H'.
        mass:  Mass of the Higgs boson, GeV.
        width:  Relative width of the Higgs boson.
    """
    
    gF = 1.1663787e-5  # GeV^(-2)
    mt = 172.5  # GeV
    
    if mass <= 2 * mt:
        return 0.
    
    w = 3 * gF * mt ** 2 * mass / (4 * math.pi * math.sqrt(2))
    beta = math.sqrt(1 - (2 * mt / mass) ** 2)
    
    if cp == 'A':
        width_g1 = w * beta
    elif cp == 'H':
        width_g1 = w * beta ** 3
    else:
        raise RuntimeError('Cannot recognize CP state "{}".'.format(cp))
    
    return math.sqrt(rel_width * mass / width_g1)
                


if __name__ == '__main__':
    
    arg_parser = argparse.ArgumentParser(epilog=__doc__)
    arg_parser.add_argument('input', help='JSON file with results of the scan')
    arg_parser.add_argument(
        '-r', '--raw', action='store_true',
        help='Request producing raw plots of CLs(g)'
    )
    arg_parser.add_argument(
        '-p', '--preliminary', action='store_true',
        help='Include label "Preliminary"'
    )
    arg_parser.add_argument(
        '-l', '--lumi', default='35.9 fb$^{-1}$ (13 TeV)',
        help='luminosity label'
    )
    arg_parser.add_argument(
        '-o', '--output', default='fig',
        help='Output directory'
    )
    args = arg_parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    
    mpl.rc('figure', figsize=(6.0, 4.8))
    
    mpl.rc('xtick', top=True, direction='in')
    mpl.rc('ytick', right=True, direction='in')
    mpl.rc(['xtick.minor', 'ytick.minor'], visible=True)
    
    mpl.rc('lines', linewidth=0.8, markersize=2.)
    mpl.rc('errorbar', capsize=1.)
    mpl.rc('hatch', linewidth=0.8)
    
    mpl.rc('axes.formatter', limits=[-3, 4], use_mathtext=True)
    mpl.rc('axes', labelsize='large')
    
    
    with open(args.input) as f:
        scans = json.load(f)
    
    mass_labels = ['400', '500', '600', '750']
    width_labels = ['2p5', '5', '10', '25', '50']
    
    
    if args.raw:
        # Plot results of CLs scans
        for (cp, cp_label), mass_label, width_label in itertools.product(
            [('A', 'CP-odd'), ('H', 'CP-even')], mass_labels, width_labels
        ):
            scan = np.asarray(scans['{}-m{}-relW{}'.format(cp, mass_label, width_label)])
            
            fig = plt.figure()
            fig.patch.set_alpha(0.)
            axes = fig.add_subplot(111)
            
            axes.plot(scan[:, 0], scan[:, 1], color='black', marker='o', label='Observed')
            
            for iexp, label_exp in [
                (2, '$-2\\sigma$ exp.'), (3, '$-1\\sigma$ exp.'), (4, 'Median exp.'),
                (5, '$+1\\sigma$ exp.'), (6, '$+2\\sigma$ exp.')
            ]:
                axes.plot(scan[:, 0], scan[:, iexp], marker='o', label=label_exp)
            
            axes.axhline(0.05, color='red', ls='dashed', lw=0.8)
            
            axes.margins(x=0.)
            axes.set_yscale('log')
            axes.set_ylim(0.01, 1.)
            
            axes.set_xlabel('$g$')
            axes.set_ylabel('$CL_s$')
            axes.legend()
            axes.text(
                0., 1.002, '{}, $m = {}$ GeV, $\\Gamma / m = {}$%'.format(
                    cp_label, mass_label, width_label.replace('p', '.')
                ),
                ha='left', va='bottom', transform=axes.transAxes
            )
            
            fig.savefig(os.path.join(
                args.output, 'raw_{}-m{}-relW{}.pdf'.format(cp, mass_label, width_label)
            ))
            plt.close(fig)
    
    
    # Produce main plots
    for cp, width_label in itertools.product(['A', 'H'], width_labels):
        width = float(width_label.replace('p', '.'))  # Per cent
        
        builder = ExclusionBuilder([
            (mass_label, scans['{}-m{}-relW{}'.format(cp, mass_label, width_label)])
            for mass_label in mass_labels
        ])
        
        fig = plt.figure()
        fig.patch.set_alpha(0.)
        axes = fig.add_subplot(111)
        
        for polygon in builder.observed():
            polygon.set_color('deepskyblue')
            polygon.set_linewidth(0)
            axes.add_artist(polygon)
        
        exp_contours = {}
        
        for num_sigma, style in [
            (0, 'solid'), (-1, 'dashed'), (1, 'dashed'), (-2, 'dotted'), (2, 'dotted')
        ]:
            lines = builder.expected(num_sigma)
            lines.set_color('black')
            lines.set_linestyle(style)
            
            exp_contours[num_sigma] = lines  # Save to build legend
            axes.add_artist(lines)
        
        axes.set_xlim(builder.masses[0], builder.masses[-1])
        axes.set_ylim(*builder.g_range)
        
        axes.set_xlabel('$m_\\mathrm{{{}}}$ [GeV]'.format(cp))
        axes.set_ylabel('Coupling scale factor')
        
        
        # Mark unphysical values of the coupling
        mass_grid = np.linspace(builder.masses[0], builder.masses[-1], num=100)
        maxg_values = np.empty_like(mass_grid)
        
        for i in range(len(mass_grid)):
            maxg_values[i] = max_g(cp, mass_grid[i], width / 100.)
        
        unphys_region = axes.fill(
            list(mass_grid) + [builder.masses[-1], builder.masses[0]],
            list(maxg_values) + [builder.g_range[1]] * 2,
            color='gray', alpha=0.5, lw=0, zorder=1.5
        )
        
        
        # Create the legend
        axes.legend(
            title='$\\mathrm{{{}}} \\to \\mathrm{{t\\bar t}}$, $\\Gamma / m = {:g}$%' \
                '\n95% CL exclusion'.format(cp, width),
            handles=[
                polygon, exp_contours[0], exp_contours[1], exp_contours[2],
                mpl.patches.Patch(
                    color=unphys_region[0].get_facecolor(), alpha=unphys_region[0].get_alpha(),
                    lw=0.
                )
            ],
            labels=[
                'Observed', 'Median exp.', '$\\pm 1\\sigma$ exp.', '$\\pm 2\\sigma$ exp.',
                r'$\Gamma_\mathrm{t\bar t} > \Gamma_\mathrm{tot}$'
            ],
            loc='lower right' if width > 15 else 'upper left'
        )
        
        
        # CMS, luminosity, and channel labels
        if not args.preliminary:
            axes.text(
                0., 1., 'CMS',
                size='x-large', weight='bold',
                ha='left', va='bottom', transform=axes.transAxes
            )
        else:
            textBoxes = [
                TextArea('CMS', textprops={'size': 'x-large', 'weight': 'bold'}),
                TextArea(' ', textprops={'size': 'large'}),
                TextArea('Preliminary', textprops={'size': 'large', 'style': 'italic'})
            ]
            cmsLabelPacker = mpl.offsetbox.HPacker(children=textBoxes, pad=0., sep=0.)
            cmsLabelBox = mpl.offsetbox.AnchoredOffsetbox(
                child=cmsLabelPacker, frameon=False,
                loc=3, # Means 'lower left'. Strings not supported here.
                bbox_to_anchor=(0., 1.), bbox_transform=axes.transAxes,
                pad=0., borderpad=0.
            )
            axes.add_artist(cmsLabelBox)
        
        axes.text(
            1., 1., args.lumi,
            ha='right', va='bottom', transform=axes.transAxes
        )
        
        axes.text(
            0.35 if args.preliminary else 0.15, 1., r'$\ell + \mathrm{jets}$ channel',
            ha='left', va='bottom', transform=axes.transAxes
        )
        
        
        fig.savefig(os.path.join(args.output, '{}-relW{}.pdf'.format(cp, width_label)))
        plt.close(fig)
