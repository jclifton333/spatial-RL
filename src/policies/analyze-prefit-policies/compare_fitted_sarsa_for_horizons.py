import os
import sys
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', '..', '..')
sys.path.append(pkg_dir)

import argparse
import src.policies.fitted_sarsa as fs

# def compute_mse_for_multiple_horizons(fname):
#   data = yaml.load(open(fname, 'r'))
#
#   for horizon in data.keys():
#     q0_hat = np.array(data[horizon]['qhat0_vals'])
#     q1_hat = np.array(data[horizon]['qhat1_vals'])
#     q0_true = np.array(data[horizon]['q0_true_vals'])
#     q1_true = np.array(data[horizon]['q1_true_vals'])
#
#     print('t: {} q0 mse: {}'.format(horizon, np.mean((q0_hat - q0_true)**2)))
#     print('t: {} q1 mse: {}'.format(horizon, np.mean((q1_hat - q1_true)**2)))
#
#   return


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--L", type=int)
  parser.add_argument("--test", type=str)
  parser.add_argument("--iterations", type=int)
  parser.add_argument("--refit", type=str)
  args = parser.parse_args()

  test = (args.test == 'True')
  refit = (args.refit == 'True')
  fs.compare_at_multiple_horizons(args.L, refit=refit, test=test, iterations=args.iterations)


