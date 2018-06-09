import pstats
import cProfile

import pfr_mdf

cProfile.runctx("pfr_mdf.run_solver(300)",
                globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
