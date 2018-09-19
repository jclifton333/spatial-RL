"""
From RSS paper, pdf pg. 13.

Decision rule construction (suppressing dependnce on time t)
  Can treat c locations
  E is space of parameters \eta
  R(s, a; eta) is a vector of priority scores (one per location), assuming locations j where
    a_j = 1 are going to be treated
  for nonnegative integers m, define
    U_l(s, a; eta, m) = 1 if R_l(s, a; eta) >= R_(m)(s, a; eta) ( i.e. mth order stat)
                        0 o.w.
  take k <= c
  define d^(1)(s; eta) to be the binary vector that selects floor(c / k) highest-priority locs
    Let w^(1) denote d^(1)(s; eta)
  Recursively, for j=2,...,k
    w^(j) = d^(j)(s; eta)
    delta_j = floor(j*c / k) - floor((j-1)*c/k)
    d^(j) = U(s, w^(j-1); eta, delta_j) + w^(j-1)
  Final decision rule is d^(k)(s; eta)

  C^T(d; beta, theta) is expected cumulative value under decision rule d, against model parametrized by theta, beta,
  up to horizon T.

  Need to take argmax of C^T(d; \hat{beta}, \hat{theta}) over class of policies described above.

Inputs
  T
  S
  eta_0
  f(y | s, a, beta)
  g(s | s, a, theta)
  { alpha_j }_j>=1
  { zeta_j }_j>=1
  tol > 0

set k = 1, \tilde{S} = S
draw Z^k from unif(-1, 1)^d

while alpha_k >= tol
  from m = 1, ..., T-1
    set A^t+m = d(S^t+m, eta^k + zeta_k Z^k)
    draw S^t+m+1 ~ g(s^t+m+1 | s^t+m, a^t+m; theta)
    draw Y^t+m ~ f(y^t+m | s^t+m, a^t+m; beta)
    set \tilde{A}^t+m
    draw \tilde{Y}^t+m...
    draw \tilde{S}^t+m+1...
  set eta^k+1 = G_E [ eta^k + alpha_k / (2 zeta_k) (Z^k 1^T_L (Y^t+T-1 - \tilde{Y}^t+T-1))]
  set k = k + 1
output eta_k

where G_E(x) is the projection of x onto the parameter space E (where eta lives)
"""