# Derivation of the regional Cournot FOC system

This note derives the linear system built in `cournot_profits_regional`
(`code/src/cournot.jl`) starting from the firms' profit functions.

## 1. Demand system

Two markets (old-gen, new-gen) with cross-substitution. Let $Q_o$ and $Q_n$
denote *total* global quantities in each market:
$$
P_o = A - b\,Q_o - s\,Q_n, \qquad
P_n = A - b\,Q_n - s\,Q_o,
$$
where $b = B/M$ and $s = \rho B/M \in [0, b)$. Competition is global, so
$P_o, P_n$ are common to all firms regardless of region.

Decompose the totals into the four product types, summed across regions
$r' = 1,\dots,R$:
$$
Q_o = \sum_{r'} \bigl(n_o[r']\,q_{oo}[r'] + n_b[r']\,q_{bo}[r']\bigr),
\qquad
Q_n = \sum_{r'} \bigl(n_b[r']\,q_{bn}[r'] + n_n[r']\,q_{nn}[r']\bigr).
$$
Within each (type, region) slot all firms are symmetric, so we index
quantities by the slot rather than the individual firm.

## 2. Profit functions

Let $c_o$ be the (common) old-gen marginal cost and $c_n[r]$ the new-gen
marginal cost in region $r$ (regional agglomeration enters here). A
representative firm in each slot has profit:

**Old-type firm in region $r$** (sells old-gen only):
$$
\pi_o[r] = (P_o - c_o)\,q_{oo}[r].
$$

**New-type firm in region $r$** (sells new-gen only):
$$
\pi_n[r] = (P_n - c_n[r])\,q_{nn}[r].
$$

**"Both"-type firm in region $r$** (sells both generations, internalises
cannibalisation):
$$
\pi_b[r] = (P_o - c_o)\,q_{bo}[r] + (P_n - c_n[r])\,q_{bn}[r].
$$

## 3. First-order conditions

Each firm chooses its quantities taking *other* firms' quantities as given
(Cournot), but internalises its own effect on prices. The key move is to
recognise that when differentiating $P_o = A - b Q_o - s Q_n$ with respect
to the firm's own quantity, only its *own* contribution to $Q_o, Q_n$
matters.

### 3a. Old-type firm, FOC for $q_{oo}[r]$

$$
\frac{\partial \pi_o[r]}{\partial q_{oo}[r]}
  = P_o - c_o + q_{oo}[r]\,\frac{\partial P_o}{\partial q_{oo}[r]}
  = P_o - c_o - b\,q_{oo}[r] = 0.
$$

Substitute the demand for $P_o$:
$$
A - b\,Q_o - s\,Q_n - c_o = b\,q_{oo}[r].
$$

Expand $Q_o, Q_n$ and move everything to the LHS:
$$
A - c_o
= b\,q_{oo}[r]
  + b\!\sum_{r'}\!n_o[r']\,q_{oo}[r']
  + b\!\sum_{r'}\!n_b[r']\,q_{bo}[r']
  + s\!\sum_{r'}\!n_b[r']\,q_{bn}[r']
  + s\!\sum_{r'}\!n_n[r']\,q_{nn}[r'].
$$

The first two terms combine because the own-region sum already contains
one copy of the firm's own slot:
$$
\boxed{\,A - c_o
= b\,(n_o[r]{+}1)\,q_{oo}[r]
  + b\!\sum_{r' \neq r}\!n_o[r']\,q_{oo}[r']
  + b\!\sum_{r'}\!n_b[r']\,q_{bo}[r']
  + s\!\sum_{r'}\!n_b[r']\,q_{bn}[r']
  + s\!\sum_{r'}\!n_n[r']\,q_{nn}[r'].\,}
$$

This is exactly the `:oo` block in the code (lines 217–232 of
`cournot.jl`): a rhs of $A - c_o$, own-slot coefficient $b(n_o[r]+1)$, and
then cross-slot coefficients that pick $b$ for same-market (old-gen)
competitors and $s$ for cross-market (new-gen) competitors.

### 3b. New-type firm, FOC for $q_{nn}[r]$

Symmetric to 3a, swapping old-gen and new-gen roles. The rhs uses
$c_n[r]$ (regional cost):
$$
A - c_n[r]
= b\,(n_n[r]{+}1)\,q_{nn}[r]
  + b\!\sum_{r' \neq r}\!n_n[r']\,q_{nn}[r']
  + b\!\sum_{r'}\!n_b[r']\,q_{bn}[r']
  + s\!\sum_{r'}\!n_b[r']\,q_{bo}[r']
  + s\!\sum_{r'}\!n_o[r']\,q_{oo}[r'].
$$

This is the `:nn` block (lines 278–294).

### 3c. "Both"-type firm, FOC for $q_{bo}[r]$

The crucial case. Because a "both" firm chooses *both* $q_{bo}[r]$ and
$q_{bn}[r]$, its own new-gen production pulls its own old-gen price down
via the cross-term $s$ — this is **cannibalisation**.

$$
\frac{\partial \pi_b[r]}{\partial q_{bo}[r]}
= P_o - c_o
  + q_{bo}[r]\,\frac{\partial P_o}{\partial q_{bo}[r]}
  + q_{bn}[r]\,\frac{\partial P_n}{\partial q_{bo}[r]} = 0.
$$

The cross-term $q_{bn}[r]\cdot\partial P_n/\partial q_{bo}[r]$ is what a
single-product firm *would not* internalise. Evaluating:
$$
\frac{\partial P_o}{\partial q_{bo}[r]} = -b, \qquad
\frac{\partial P_n}{\partial q_{bo}[r]} = -s,
$$
so
$$
P_o - c_o - b\,q_{bo}[r] - s\,q_{bn}[r] = 0.
$$

Substitute $P_o$ and expand:
$$
A - c_o
= b\,q_{bo}[r] + s\,q_{bn}[r]
  + b\!\sum_{r'}\!n_o[r']\,q_{oo}[r']
  + b\!\sum_{r'}\!n_b[r']\,q_{bo}[r']
  + s\!\sum_{r'}\!n_b[r']\,q_{bn}[r']
  + s\!\sum_{r'}\!n_n[r']\,q_{nn}[r'].
$$

Now collect own-slot terms. The $b\,q_{bo}[r]$ combines with one copy from
$b\,n_b[r]\,q_{bo}[r]$ in the sum, and the $s\,q_{bn}[r]$ combines with one
copy from $s\,n_b[r]\,q_{bn}[r]$:

$$
\boxed{\,A - c_o
= b\,(n_b[r]{+}1)\,q_{bo}[r]
  + s\,(n_b[r]{+}1)\,q_{bn}[r]
  + b\!\sum_{r' \neq r}\!n_b[r']\,q_{bo}[r']
  + b\!\sum_{r'}\!n_o[r']\,q_{oo}[r']
  + s\!\sum_{r' \neq r}\!n_b[r']\,q_{bn}[r']
  + s\!\sum_{r'}\!n_n[r']\,q_{nn}[r'].\,}
$$

The **two $+1$'s** on own-region slots are the signature of
cannibalisation: one on the own-market own-slot (standard Cournot markup)
and one on the *cross*-market own-slot (the extra term that makes "both"
firms reluctant to push new-gen). This matches the `:bo` block of the code
(lines 234–255), including the comment about the `(n_b[r]+1)` cross
coefficient.

### 3d. "Both"-type firm, FOC for $q_{bn}[r]$

Symmetric to 3c with markets swapped and $c_n[r]$ on the rhs:
$$
A - c_n[r]
= b\,(n_b[r]{+}1)\,q_{bn}[r]
  + s\,(n_b[r]{+}1)\,q_{bo}[r]
  + b\!\sum_{r' \neq r}\!n_b[r']\,q_{bn}[r']
  + b\!\sum_{r'}\!n_n[r']\,q_{nn}[r']
  + s\!\sum_{r' \neq r}\!n_b[r']\,q_{bo}[r']
  + s\!\sum_{r'}\!n_o[r']\,q_{oo}[r'].
$$

This is the `:bn` block (lines 257–276).

## 4. Assembled linear system

Stacking 3a–3d across all active $(type, r)$ slots gives a linear system
of up to $4R$ equations in up to $4R$ unknowns:
$$
\mathbf{M}\,\mathbf{q} = \mathbf{r},
$$
where:

- $\mathbf{q}$ collects the active $q_{\cdot}[r]$ variables;
- $\mathbf{r}$ has entries $A - c_o$ for $:oo$ and $:bo$ rows, and
  $A - c_n[r]$ for $:bn$ and $:nn$ rows (this is where the regional cost
  differences enter);
- $\mathbf{M}$ has the coefficients derived above, with the $(+1)$
  appearing on own-market own-slot entries and, for "both"-type rows,
  *also* on the cross-market own-slot entry.

The code solves it directly with `q_vec = Mat \ rhs`.

## 5. Corner handling

If the interior solution returns $q \le 0$ for some slot, that slot's
product is dropped from the system (its activity flag goes to `false`) and
the remaining system is re-solved. The loop iterates at most $4R+2$ times
because each pass strictly shrinks the active set. Economically, this
captures the "strong cannibalisation" case where a "both" firm abandons
old-gen production entirely ($q_{bo}[r] = 0$) when its new-gen margin is
large enough.

## 6. Per-firm profits from equilibrium quantities

Having solved for $q^*$, profits follow directly from the FOCs rather than
requiring a separate substitution into the demand system.

**Old-type and new-type**. The singleton FOCs give $P_o - c_o = b\,q_{oo}[r]$
and $P_n - c_n[r] = b\,q_{nn}[r]$, hence
$$
\pi_o[r] = (P_o - c_o)\,q_{oo}[r] = b\,q_{oo}[r]^2,
\qquad
\pi_n[r] = (P_n - c_n[r])\,q_{nn}[r] = b\,q_{nn}[r]^2.
$$

**"Both"-type**. The two FOCs give
$P_o - c_o = b\,q_{bo}[r] + s\,q_{bn}[r]$ and
$P_n - c_n[r] = b\,q_{bn}[r] + s\,q_{bo}[r]$, hence
$$
\pi_b[r]
= (b\,q_{bo}[r] + s\,q_{bn}[r])\,q_{bo}[r]
+ (b\,q_{bn}[r] + s\,q_{bo}[r])\,q_{bn}[r]
= b\,(q_{bo}[r]^2 + q_{bn}[r]^2) + 2s\,q_{bo}[r]\,q_{bn}[r].
$$

These are exactly the return values at the end of
`cournot_profits_regional`. The costs $c_o, c_n[r]$ and demand intercept
$A$ do not appear explicitly in the profit formulas because they have
already been substituted out via the FOCs — they enter only through the
equilibrium $q^*$ delivered by the linear solve in step 4.
