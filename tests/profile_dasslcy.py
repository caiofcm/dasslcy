import pstats
import cProfile

import dasslcy_simple

cProfile.runctx("dasslcy_simple.run_scenario3_mod()",
                globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
