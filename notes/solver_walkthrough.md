# Walkthrough of `code/src/solver.jl`

This note walks through the backward-induction solver for the 2-period
regional Igami model. The solver lives in `code/src/solver.jl` and is
called from `solve_initial(s0, p)` / `solve_2period(p)`.

## 1. What the solver computes

For a given **initial state** $s_0$ and parameters $p$ (including the
per-region agglomeration vector $\gamma \in \mathbb{R}^R$), the solver
returns a `StateCCPs` struct containing the per-region conditional choice
probabilities at $s_0$:

| Field    | Meaning                                                   |
|----------|-----------------------------------------------------------|
| `p_so[r]`| P(stay | old firm in region $r$ decides)                  |
| `p_io[r]`| P(innovate | old firm in region $r$ decides)              |
| `p_sb[r]`| marginal P(stay | "both" firm in region $r$ decides)      |
| `p_sn[r]`| marginal P(stay | "new" firm in region $r$ decides)       |
| `p_ep[r]`| marginal P(enter | PE in region $r$ decides)              |

Because moves are sequential within each stage (see §3), later-region CCPs
depend on the realised history of earlier-region moves. The reported
numbers are **marginalised** over the equilibrium distribution of those
histories — computed by a forward traversal at the end.

## 2. The model in one paragraph

Firms choose actions in a **fixed sequence** within each period:
1. **Old** incumbents decide stay / innovate / exit.
2. **Both**-type incumbents (old firms that innovated in earlier stages,
   plus freshly locked-in innovators from this period's old stage) decide
   stay / exit.
3. **New**-type firms decide stay / exit.
4. **Potential entrants** (PE) decide enter / not.

Decisions are private information with i.i.d. EVT1 shocks, so every
firm's within-stage problem is a single-agent discrete choice with
closed-form logit CCPs. Flow profits come from a two-market Cournot
equilibrium (`cournot_profits_regional`, see `cournot_derivation.md`).

The **regional twist**: the cost of new-tech differs across regions because
of local agglomeration, $c_{n,r} = c_{n,0} - \gamma_r(n_b[r] + n_n[r])$, so
the Cournot FOC system is asymmetric and different (type, region) slots
earn different per-firm profits.

## 3. The key simplification: sequential region moves within stages

At each of the four stages, we further order the three regions
$r = 1 \to 2 \to 3$. When firms in region $r$ decide, they **observe** the
realised outcome of regions $1, \dots, r-1$ and **integrate** over the
strategies of regions $r+1, \dots, R$ (which themselves depend on the
extended history).

This is the regional analogue of Igami's original trick: in the
single-region model he orders *types* (old → both → new → pe) precisely so
that each later type's problem is single-agent given the realised play of
earlier types. Here we add one more level of sequencing, across regions
within a stage. The payoff is structural:

**Without sequencing**, each stage is a joint fixed point over the vector
$(p_1, p_2, p_3)$ of regional mixing probabilities — because $u_{stay,r}$
depends on all three of them via the continuation value.

**With sequencing**, each stage becomes three *scalar* fixed points, one
per region, solved in reverse order $r = R \to 1$ with CCPs keyed by the
realised prefix of earlier-region outcomes. Region $R$'s FP takes the full
history as given (no more sub-stages to integrate over). Region $R-1$
plugs in the cached region-$R$ response. And so on.

One visible cost of the ordering: it isn't economically neutral.
Region 1 commits without observing anyone; region 3 best-responds to the
realised history. This shows up as a small first-mover asymmetry in the
baseline ($\le 10^{-3}$ in our calibration) but doesn't affect any
counterfactual comparison qualitatively.

## 4. Top-down structure of the solver file

The file is organised strictly bottom-up, so Julia parses dependencies in
order:

```
1.  EV struct, ZERO_EV, tuple helpers
2.  compute_terminal_values(states, p)          [period-1 Cournot]
3.  SolveContext, SolveCaches
4.  solve_pe_region     ↔  ev_after_pe_region   [stage 4, global cache]
5.  solve_new_region    ↔  ev_after_new_region  [stage 3]
6.  solve_both_region   ↔  ev_after_both_region [stage 2]
7.  solve_old_region    ↔  ev_after_old_region  [stage 1]
8.  traverse_old! → traverse_both! → traverse_new! → traverse_pe!
9.  solve_state, solve_2period, solve_initial
```

Each stage has two partner functions:

- **`solve_<stage>_region(s, r, ...)`** — returns the CCP(s) for a firm in
  region $r$ at sub-state $s$. Runs a scalar fixed point over the common
  mixing probability of that region's peers.
- **`ev_after_<stage>_region(s, r, ...)`** — returns the expected
  continuation value $\mathrm{EV}$ *after* regions $r, r+1, \dots, R$ of
  the current stage have finished playing (and all downstream stages too).
  Used as the continuation plugged into earlier sub-stages' FP.

Stages hand off to each other at the base case `r > R`: e.g.
`ev_after_old_region(s, R+1, ...)` calls `ev_after_both_region(s, 1, ...)`,
which eventually hands off to `ev_after_new_region(s, 1, ...)` then
`ev_after_pe_region(s, 1, ...)` then `V1[s]` at the bottom of the
recursion.

## 5. Sub-state semantics

The solver uses the `State` struct (four `NTuple{R,Int}` fields
`n_o, n_b, n_n, n_pe`) to represent both the "true" state and the
intermediate states mid-way through a stage. The mapping:

- **At OLD region $r$:** regions $1..r-1$ already played old-stage, so
  their `n_o[r']` has shrunk to survivors $k_{so}[r']$ and `n_b[r']` has
  grown by $k_{io}[r']$ (newly locked-in innovators). Regions $r..R$ are
  still at their original counts.
- **At BOTH region $r$:** similar, but the stage-2 survivors/exiters
  update `n_b[r']` for $r' < r$. The **deciders** at region $r$ are
  `ctx.s_orig.n_b[r]`, *not* current `s.n_b[r]` — because the latter also
  contains locked-in innovators from the OLD stage, who don't choose
  again at BOTH.
- **At NEW region $r$:** n_n updates track stage-3 survivors. Deciders
  are current `s.n_n[r]` which equals `s_orig.n_n[r]` (NEW hasn't touched
  them yet).
- **At PE region $r$:** `s.n_pe[r']` shrinks and `s.n_n[r']` grows for
  $r' < r$ as earlier regions' PE firms entered.

So the same `State` type represents states at every sub-stage; what
differs is which counts are "locked in from earlier sub-stages" vs. "about
to play."

## 6. The stage solvers, one at a time

Take `solve_new_region` as the prototype (lines ~140–170):

```julia
function solve_new_region(s, r, ctx, V1, p, C; ...)
    key = (s, r)
    haskey(C.new_ccp, key) && return C.new_ccp[key]     # cache lookup
    n = s.n_n[r]
    n == 0 && (C.new_ccp[key] = 0.0; return 0.0)

    p_r = 0.5
    for _ in 1:max_iter                                 # scalar FP
        u_stay = ctx.flow_n[r]                          # instantaneous payoff
        for v in 0:(n - 1)                              # sum over peer survivors
            lp = log_binomial_prob(n - 1, v, p_r)
            lp == -Inf && continue
            prob = exp(lp)
            s_next = State(s.n_o, s.n_b, set_i(s.n_n, r, v + 1), s.n_pe)
            ev = ev_after_new_region(s_next, r + 1, ctx, V1, p, C)
            u_stay += p.beta * prob * ev.new[r]
        end
        new_p = logit2(u_stay, 0.0, p.sigma)
        diff = abs(new_p - p_r); p_r = new_p
        diff < tol && break
    end
    C.new_ccp[key] = p_r
    return p_r
end
```

Reading it top to bottom:

1. **Cache lookup.** If we've already solved this sub-state, return.
2. **Trivial case.** If region $r$ has no new-type firms, the CCP is
   irrelevant; store zero.
3. **Scalar FP.** Initialise $p_r = 0.5$. Each iteration:
   - Start with the instantaneous payoff from staying, `ctx.flow_n[r]`
     (this is $\pi_n[r]$ at $s_{\text{orig}}$, computed once at the top of
     `solve_state`).
   - Sum over the number of peers who stay, drawn from
     $\text{Binomial}(n-1, p_r)$. For each realisation:
     - Build the next sub-state: own + $v$ peers stay, so `n_n[r]` becomes
       $v+1$.
     - Look up the continuation via `ev_after_new_region(s_next, r+1)` —
       which is cached after its first call, so subsequent FP iterations
       are fast.
     - Pick the `ev.new[r]` component because the firm we're reasoning
       about ends up as a new-type firm in region $r$.
   - Logit update: $p_r^{\text{new}} = \mathrm{logit}(u_{stay}, 0;
     \sigma)$, against outside option 0 (exit = zero continuation).
   - Check convergence. Done in ~10 iterations.
4. **Store and return.**

The other three stage solvers are structurally identical with these swaps:

- **PE** (`solve_pe_region`): own-action is "enter" (not "stay"), payoff is
  $-\phi$ (entry cost), continuation is still `ev.new[r]` because a PE that
  enters becomes a new firm. Handoff base case: `ev_after_pe_region(s, R+1)
  = V1[s]` (the bottom of the recursion).
- **BOTH** (`solve_both_region`): deciders are `ctx.s_orig.n_b[r]`, not
  `s.n_b[r]` (see §5). Continuation is `ev.both[r]`.
- **OLD** (`solve_old_region`): 3-way choice, so the FP is over the pair
  $(p_s, p_i)$ instead of a scalar. Peers distribute multinomially over
  $(k_{so}, k_{io}, k_{eo})$. Two continuations per realisation: `ev.old[r]`
  for stay, `ev.both[r]` for innovate (own moves to the "both" slot). The
  logit update normalises over 3 actions including the outside option.

## 7. The `ev_after_*_region` functions

These are what you plug in for the continuation value when *you* are not
one of the deciders at the current sub-stage — e.g., when an old-stage
decider wants to know "what's the expected state after all downstream
stages play out?"

Prototype (`ev_after_new_region`, lines ~175–205):

```julia
function ev_after_new_region(s, r, ctx, V1, p, C)
    r > R && return ev_after_pe_region(s, 1, V1, p, C)   # handoff
    key = (s, r)
    haskey(C.ev_new, key) && return C.ev_new[key]

    n = s.n_n[r]
    n == 0 && return (C.ev_new[key] = ev_after_new_region(s, r+1, ...))

    p_r = solve_new_region(s, r, ctx, V1, p, C)           # solve region r first
    old = zeros(R); both = zeros(R); new_ = zeros(R)
    for v in 0:n                                          # ← note: n peers, not n-1
        lp = log_binomial_prob(n, v, p_r)
        lp == -Inf && continue
        prob = exp(lp)
        s_next = State(s.n_o, s.n_b, set_i(s.n_n, r, v), s.n_pe)
        ev = ev_after_new_region(s_next, r + 1, ctx, V1, p, C)
        for k in 1:R
            old[k]  += prob * ev.old[k]
            both[k] += prob * ev.both[k]
            new_[k] += prob * ev.new[k]
        end
    end
    result = EV(Tuple(old), Tuple(both), Tuple(new_))
    C.ev_new[key] = result
    return result
end
```

The structural differences from `solve_new_region`:

- **The observer isn't among the deciders**, so the peer count is `n`, not
  `n-1`. All $n$ region-$r$ firms distribute over stay/exit.
- **Own action doesn't contribute** anything to the state update — we're
  integrating over the equilibrium play of *someone else's* stage.
- **Returns a full `EV` struct**, not a scalar. The caller picks out
  whichever component it needs (`ev.old[r]`, `ev.both[r]`, `ev.new[r]`).
- **Recurses forward** into the next region of the same stage via
  `r + 1`, accumulating state updates. When `r > R`, hands off to the next
  stage's `r=1` function.

The `n == 0` short-circuit saves work: if region $r$ has no firms of the
current type, there's nothing to integrate, so we just skip straight to
region $r+1$.

## 8. The base case: `ev_after_pe_region` at `r > R`

This is where the whole recursion bottoms out (line ~104):

```julia
function ev_after_pe_region(s, r, V1, p, C)
    r > R && return get(V1, s, ZERO_EV)        # ← base case
    # ...
end
```

After regions $1, \dots, R$ of the PE stage have all resolved, the state
$s$ is the final first-period state, and `V1[s]` gives the per-firm
period-1 Cournot profits by type and region — exactly what
`compute_terminal_values` filled in at the top of `solve_state`. The
`get(V1, s, ZERO_EV)` fallback handles unreachable states (shouldn't fire
if the initial total is $\le N_{\max}$, but it's a safety net).

## 9. Caches: global vs. per-`s_orig`

The `SolveCaches` struct bundles eight `Dict`s. Two are **global** (shared
across initial states) and six are **per-`s_orig`** (rebuilt inside
`solve_state`):

| Cache        | Key              | Scope             | Why |
|--------------|------------------|-------------------|-----|
| `pe_ccp`     | `(State, r)`     | **global**        | PE entry problem depends only on `(s, r, V1, p)`, not on flow profits at $s_\text{orig}$ |
| `ev_pe`      | `(State, r)`     | **global**        | Same |
| `new_ccp`    | `(State, r)`     | per `s_orig`      | Uses `ctx.flow_n[r]` from `s_orig` |
| `ev_new`     | `(State, r)`     | per `s_orig`      | Integrates over new-stage CCPs, which are `s_orig`-dependent |
| `both_ccp`   | `(State, r)`     | per `s_orig`      | Uses `ctx.flow_b[r]` |
| `ev_both`    | `(State, r)`     | per `s_orig`      | Same |
| `old_ccp`    | `(State, r)`     | per `s_orig`      | Uses `ctx.flow_o[r]` and `p.kappa` |
| `ev_old`     | `(State, r)`     | per `s_orig`      | Same |

The caches turn the recursive tree search into top-down dynamic
programming. Without them the solver would be exponential in the tree
depth (12 sub-stages × branching) and effectively never finish; with
them, each `(stage, r, State)` key is computed at most once.

Why is it safe to share the PE caches globally? Because the PE decider's
payoff is $-\phi + \beta V_1[s_{\text{next}}].\text{new}[r]$ — no flow
profit term, and $V_1$ itself is a pure function of the period-1 state.
The other stages all have `u_stay = \text{flow} + \beta \cdot \mathrm{EV}$
where `flow` comes from $s_{\text{orig}}$'s Cournot outcome, so mixing
their caches across initial states would return wrong answers.

## 10. Forward traversal for marginal CCPs

Under sequential moves, the CCP at `(stage, r)` is a function of the
realised history — different earlier-region outcomes give different
region-$r$ mixing probabilities. But the reported `StateCCPs` has one
number per region. What do we report?

The **expected** CCP, averaged over the equilibrium distribution over
histories. The `traverse_*!` family (lines ~350–450) does this by walking
the sub-stage tree forward from $s_0$:

```
traverse_old!(s0, 1, weight=1.0)
  → at each sub-stage (stage, r):
      solve that region's CCP at the current sub-state
      add `weight · p_r` to the marginal accumulator
      branch over the (multi/bi)nomial realisations
      for each branch, recurse with weight × branch-probability
  → at r > R, hand off to the next stage's traverse function
```

By the time `traverse_pe!` terminates, each `mso[r], mio[r], msb[r],
msn[r], mep[r]` accumulator holds the expected mixing probability at
region $r$ of the corresponding stage. These are packed into the
`StateCCPs` returned by `solve_state`.

Crucially, these functions **reuse the same cache** as the main solvers —
`solve_new_region(s, r, ...)` inside `traverse_new!` is a cache hit (the
function has already been called during the backward-induction phase when
computing `ev_after_*_region`), so the traversal is cheap: it's just
walking the tree and reading off already-computed CCPs.

## 11. `solve_state`: putting it all together

The top-level for a single initial state (lines ~460–495):

```julia
function solve_state(s_orig, V1, pe_ccp_cache, ev_after_pe_cache, p)
    # 1. Compute s_orig flow profits once
    cn = c_n_vec(s_orig, p)
    pi_o, pi_b, pi_n = cournot_profits_regional(
        s_orig.n_o, s_orig.n_b, s_orig.n_n, p.c_o, cn, p)

    # 2. Bundle context + caches
    ctx = SolveContext(s_orig, pi_o, pi_b, pi_n)
    C = SolveCaches(pe_ccp_cache, ev_after_pe_cache,
                    Dict(...), Dict(...), Dict(...),
                    Dict(...), Dict(...), Dict(...))

    # 3. Forward traversal, accumulating marginal CCPs
    mso, mio, msb, msn, mep = (zeros(R) for _ in 1:5)
    traverse_old!(s_orig, 1, 1.0, ctx, V1, p, C, mso, mio, msb, msn, mep)

    return StateCCPs(Tuple(mso), Tuple(mio), Tuple(msb), Tuple(msn), Tuple(mep))
end
```

Three steps:

1. **Evaluate flow profits at $s_\text{orig}$** using
   `cournot_profits_regional` (derived in `cournot_derivation.md`). These
   become the `ctx.flow_*` per-region NTuples that every sub-stage solver
   reads off for its `u_stay`.
2. **Build the cache bundle.** The PE caches come in from outside (so
   multiple `solve_state` calls share them); the other six are fresh.
3. **Run the forward traversal**, which internally invokes every
   `solve_<stage>_region` (populating the caches lazily as it goes) and
   collects marginalised CCPs.

That's the whole solver. The Cournot step is $O(R^3)$ (linear solve over a
$4R \times 4R$ system); the traversal and backward induction together
cost roughly the number of reachable sub-states times a small constant,
which works out to ~20ms per `solve_state` call in our baseline
calibration.

## 12. `solve_initial` and `solve_2period`

These are thin wrappers:

- **`solve_initial(s0, p)`** — build the full state space, compute `V1`
  once, create the PE caches, then call `solve_state` at $s_0$ only.
  Returns `(V1, ccps_at_s0)`.
- **`solve_2period(p)`** — same setup, but loops over every state in the
  enumerated state space and calls `solve_state` for each, returning
  `(V1, Dict{State, StateCCPs})`. The per-`s_orig` caches get rebuilt
  each iteration; the PE caches stay populated across the loop, so later
  calls are faster. Mainly used for estimation sweeps where you need CCPs
  at every state.

## 13. Where to look in the code for common questions

| Question | Where to look |
|----------|---------------|
| What are the stages and their order? | Module docstring, top of `solver.jl` |
| How is a sub-state represented? | §5 above + `State` struct in `state_space.jl` |
| Why is there a fixed point *at all* within a region? | `solve_new_region` peer sum — firms' beliefs about peer mixing must be consistent, hence FP over scalar $p_r$ |
| Where do flow profits come from? | `cournot_profits_regional` call inside `solve_state` |
| Why is the PE cache global but the others per-`s_orig`? | §9 |
| Why do the `ev_after_*_region` functions use `n` peers but `solve_*_region` uses `n-1`? | The observer is *not* one of the deciders in the `ev_after` case; in the `solve` case, the "own firm" is one of the $n$, so $n-1$ are peers |
| Why integrate over regions in reverse order? | Region $R$ has no future sub-stages at its stage, so its FP doesn't need a continuation from future regions — resolve it first, cache it, and earlier regions just plug in the cached answer |
| Where does the first-mover asymmetry come from? | The ordering $r = 1 \to 2 \to 3$ isn't neutral; see §3 |
