# Two limits of the same flow

The Schrödinger bridge is a noisy transport process constrained by its starting and ending densities. Here it runs from a positive density $`\rho_0`$ to uniform density $1$. Following its net mass velocity, called the **current velocity**, gives a deterministic cartogram. Densities have mean one with respect to area. The domains used here are the unit sphere, of geodesic diameter $`\pi`$, and the reflecting rectangle $`[0,2\pi]\times[0,\pi]`$, of diameter $`\pi\sqrt5`$.

```text
gamma -> 0                 finite gamma                    gamma -> infinity
transport map  <------ current-velocity bridge flow ------> diffusion map
conditional quadratic rate                                eigenvalue clock
```

Write $`P_s=e^{s\Delta}`$ for heat evolution over time $s$, with $`s=\gamma t/2`$ and $`S=\gamma/2`$. The bridge potentials, positive functions that enforce the two endpoint densities, solve

```math
\widehat\phi_0 P_S\phi_1=\rho_0,\qquad \phi_1 P_S\widehat\phi_0=1.
```

At heat time $s$, $`\widehat\phi_s=P_s\widehat\phi_0`$, $`\phi_s=P_{S-s}\phi_1`$ and $`\rho_s=\phi_s\widehat\phi_s`$. The current velocity per unit heat time and the diffusion velocity are

```math
V_s=\nabla\log\phi_s-\nabla\log\widehat\phi_s,
\qquad U_\sigma=-\nabla\log(P_\sigma\rho_0).
```

The bridge map integrates $`v_t=(\gamma/2)V_{\gamma t/2}`$ over $[0,1]$. The Gastner-Newman map integrates $`U_\sigma`$ over $`[0,\infty)`$. A time change affects comparison of velocities, not the endpoint of a given flow path.

The noise convention throughout is $`\partial_t u=(\gamma/2)\Delta u`$ on bridge time $`t\in[0,1]`$.

## Transport end: a conditional quadratic rate

Set $`\phi_t=\sqrt{\rho_t}\exp(\psi_t/\gamma)`$ and $`\widehat\phi_t=\sqrt{\rho_t}\exp(-\psi_t/\gamma)`$. The equations become

```math
\partial_t\rho+\nabla\cdot(\rho\nabla\psi)=0,
\qquad
\partial_t\psi+\tfrac12|\nabla\psi|^2
=-\tfrac{\gamma^2}{2}\frac{\Delta\sqrt\rho}{\sqrt\rho}.
```

At zero noise this is the fluid formulation of optimal transport. Noise enters through $`\gamma^2`$.

**Hypothesis (H).** The bridge variables $`(\rho^\gamma,\psi^\gamma)`$ have uniformly bounded $`C^k([0,1]\times M)`$ norms, for $`k\ge4`$, and $`\rho^\gamma\ge c>0`$, uniformly for $`0<\gamma\le\gamma_0`$.

Under (H), the perturbative derivation gives

```math
\sup_x d(T_\gamma(x),T_0(x))\le C\gamma^2.
```

Subtracting the zero-noise optimality system leaves a forcing of order $`\gamma^2`$. The argument uses a bounded inverse for the linearised optimality system at the smooth, nondegenerate transport interpolation. Controlling the velocity difference and applying the flow stability estimate then gives the map rate. This stability step is part of the unrefereed derivation; even dependence on $`\gamma`$ alone is not a regularity argument. The formal expansion is $`T_\gamma=T_0+\gamma^2T_2+O(\gamma^4)`$ in local coordinates.

The numerical checks use smooth manufactured densities: test inputs constructed from known transport maps so that errors can be measured directly. They do not establish (H) for arbitrary population data. On the plane, $`T_0=x-\nabla\psi`$ and $`\rho_0=\det(I-D^2\psi)`$ for cosine potentials. On the sphere, $`T_0=\exp_x(-\nabla u)`$ and the source density is its Jacobian. Here $`\exp_x`$ follows a great-circle displacement from $x$. The [stored reference checks](../results/transport_end_sphere/mccann_check.json) evaluate positivity and transport optimality on finite point sets; they are not a substitute for a global theorem.

The static barycentric projection instead sends each source point to the average of its possible destinations under the bridge's endpoint coupling. Its planar interior term $`(\gamma/2)\nabla\log\rho_0`$ is already published by Mulcahy and Pal; $`\nabla\log\rho_0`$ is the density's score, or gradient of its logarithm. At reflecting walls, a boundary layer prevents a global first-order maximum-error claim for that static map. On the sphere, a **Fréchet barycentre** minimises the average squared geodesic distance to those destinations. Its stored fitted exponent is near one, but the planar coefficient is not assumed to carry over unchanged. The contribution examined here is the current-velocity flow and its endpoint limits.

## Diffusion end: why the natural clock fails

At the endpoint, $`\phi_S\widehat\phi_S=1`$ exactly, hence

```math
V_S=-2\nabla\log\widehat\phi_S.
```

For large noise, $`\widehat\phi_S`$ approaches $`P_S\rho_0`$ to the required relative order, giving $`V_S\sim2U_S`$. The coefficient two in the potential identity is exact. Equality to twice the diffusion velocity is an asymptotic statement, not an identity at arbitrary finite noise. Consequently, with $`\sigma=s`$,

```math
\frac{\|V_S-U_S\|_2}{\|U_S\|_2}\longrightarrow1.
```

The failure is concentrated near the end. Agreement at a fixed early heat time does not imply uniform agreement along the bridge.

## The eigenvalue clock

Heat flow smooths each spatial mode at an exponential rate given by an eigenvalue of $`-\Delta`$. Let $`\lambda_1`$ be the lowest positive eigenvalue, the slowest nonconstant decay rate, and $`\lambda_*`$ the next **distinct** eigenvalue. Assume **(G)**: $`\Pi_1(\rho_0-1)\ne0`$, meaning the starting density contains some of the slowest-decaying modes. For smooth positive data bounded below, define

```math
e^{-\lambda_1\sigma(s)}
=e^{-\lambda_1s}-e^{-\lambda_1(\gamma-s)}+e^{-\lambda_1\gamma}.
```

This increasing clock maps $`[0,\gamma/2]`$ to $`[0,\gamma]`$. If $`x=e^{-\lambda_1s}`$ and $`y=e^{-\lambda_1(\gamma-s)}`$, then

```math
\sigma'(s)=\frac{x+y}{x-y+xy}.
```

It matches the velocity amplitude of the lowest eigenspace. Compare the velocities using

```math
E_\gamma(s)=\frac{\|V_s/\sigma'(s)-U_{\sigma(s)}\|_{L^2(dA)}}{\|U_{\sigma(s)}\|_{L^2(dA)}}.
```

The unrefereed derivation gives the uniform-in-time bound

```math
\sup_{0\le s\le\gamma/2}E_\gamma(s)\le C e^{-\kappa\gamma},
\qquad \kappa=\tfrac12\min(\lambda_1,\lambda_*-\lambda_1).
```

| Domain | $`\lambda_1`$ | $`\lambda_*`$ | Predicted $`\kappa`$ |
|---|---:|---:|---:|
| Unit sphere | 2 | 6 | 1 |
| Reflecting rectangle | 1/4 | 1 | 1/8 |

The dominant correction on these domains comes from self-interaction of the lowest mode, not just from the next eigenvalue. Writing $`q=e^{-\lambda_1\gamma/2}`$ and $`\ell=\Pi_1(\rho_0-1)`$, the terminal expansion gives

```math
E_\gamma(S)\sim\frac{q}{2}\frac{\|\nabla(\ell^2)\|_2}{\|\nabla\ell\|_2}.
```

In particular, $`\sigma(S)=\gamma`$ while $`\sigma'(S)=2/q`$. The clock doubles the terminal heat-time horizon; its terminal derivative is not two. The separate factor two above concerns the bridge's natural-clock velocity.

**Formal claim A3.** A second-order comparison over admissible clocks suggests that the exponential rate cannot be improved, only its constant. With $`G=\|\nabla(\ell^2)\|_2/\|\nabla\ell\|_2`$, the proposed lower bound is $qG/(2+G)$, against $qG/2$ for this clock. This is a formal argument, not an established optimality result.

If (G) fails, the domain's $`\lambda_1`$ need not work. The included symmetric plane example uses the lowest active eigenvalue instead. Its projected and unprojected runs also show how rounding can excite a slower mode absent from the exact density. On a discrete grid, the relevant spectrum is that of the chosen heat operator; satisfying a discrete continuity equation does not establish a continuum transport rate at fixed grid spacing.

## References

- Gastner and Newman, *Diffusion-based method for producing density-equalizing maps*, PNAS 101 (2004). [Paper](https://doi.org/10.1073/pnas.0400280101).
- Bogdan, *The optimal-transport cartogram: world population as a Brenier map* (2026). [Preprint](https://arxiv.org/abs/2609.14089), [public code](https://github.com/philippbogdan/historical-cartogram). Context for the transport construction; the bridge measurements are included in this repository.
- Garrett Mulcahy and Soumik Pal, *Diffusion Approximations to Schrödinger Bridges and the Convergence of Entropic Potentials* (submitted 18 September 2026). [Preprint](https://arxiv.org/abs/2609.22595). Credit for the planar score term, not a claim of originality here.
