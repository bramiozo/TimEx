# algorithm to extract "gnome paths" for univariate series:
# 1. set the starting point for each line at y=0, t=0 (optionally standardize or just off-set)
# 2. extract the monotonic cubic splines for the series from step 1.
# 3. integrate the paths through (y^',t) with a kernel and with a grid-intersection sum
#
# grid-intersection sum, with count-weighting per time-interval
# this gives a relative presence per grid-cell
# we can draw a path through the maxima. Can we combine the presence-matrix
# with a 2d Markov decision matrix?
#
