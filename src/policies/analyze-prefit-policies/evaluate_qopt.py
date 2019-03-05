import os
import sys
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', '..', '..')
sys.path.append(pkg_dir)

import argparse
import src.policies.diagnostics.fitted_sarsa as fs

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--L", type=int)
  parser.add_argument("--test", type=str)
  parser.add_argument("--iterations", type=int)
  parser.add_argument("--refit", type=str)
  args = parser.parse_args()

  test = (args.test == 'True')
  refit = (args.refit == 'True')
  fs.evaluate_qopt(args.L, horizons=(10, 50), refit=refit, test=test, iterations=args.iterations)
