# Premier League Monte Carlo Dashboard

Interactive dashboard for simulating the remainder of a Premier League season
and visualising model-based projections for:

- final league table (distribution of points and ranks),
- probabilities of key outcomes (champions, top-4, top-6, relegation),
- win/draw/loss probabilities and expected goals for the next matchweek.

The site is **static** (HTML/CSS/JS).  
All heavy statistical modelling and Monte Carlo simulation is done **offline**
in Python using Bayesian models implemented with [PyMC](https://www.pymc.io/).
The backend writes JSON files under `data/`, and the frontend reads those files.

The project is primarily **educational**: the goal is to show how modern
football analytics models can be used in practice.

The statistical modelling is inspired by:

> Leonardo Egidi, Dimitris Karlis, Ioannis Ntzoufras  
> *Predictive Modelling for Football Analytics*  
> Chapman & Hall/CRC Data Science Series, 2024.  
> https://www.amazon.com/Predictive-Modelling-Football-Analytics-Chapman-ebook/dp/B0FMPHTCXH  

The models here are **simplified implementations** of the Poisson, Skellam and
heavy-tailed goal-difference models discussed in Chapters 4–5 of the book,
adapted to a single-season Premier League setting and to a static web
deployment.

---

## 1. Project structure

```text
backend/
  __init__.py
  config.py
  data_sources.py
  evaluation.py
  logging_conf.py
  models.py
  simulate_league.py
  utils.py

data/
  premierleague.xlsx                 # raw Excel, fixtures + results
  premier_results_raw_2025_2026.csv  # cleaned, standardised match data
  teams_premier_2025_2026.xlsx       # preseason ranking info
  team_metadata.json                 # team IDs, names, colours, logos

  premier_outcomes_poisson_2000.json
  premier_outcomes_skellam_2000.json
  premier_outcomes_student_t_2000.json

  matchday_predictions_poisson.json
  matchday_predictions_skellam.json
  matchday_predictions_student_t.json

images/
  ... club logo PNG files ...

index.html
style.css
app.js
requirements.txt


## 2. **Poisson goals model (`PoissonGoalsModel`)**

The Poisson model is the baseline specification and follows the “double Poisson”
framework described in *Predictive Modelling for Football Analytics* (Egidi, Karlis, Ntzoufras),
adapted to the Premier League data used in this project.

**Model structure**

For each match \(i\):

- home team: \(h(i)\)
- away team: \(a(i)\)

We model the goals scored by home and away teams as **independent Poisson**:

\[
Y_{\text{home},i} \sim \text{Poisson}(\lambda_{\text{home},i}), \quad
Y_{\text{away},i} \sim \text{Poisson}(\lambda_{\text{away},i})
\]

with log-linear predictors:

\[
\begin{aligned}
\log \lambda_{\text{home},i} &= \mu + \text{home} + \text{att}_{h(i)} + \text{def}_{a(i)} + \beta_{\text{pre}} \cdot \text{pre}_{h(i)} \\
\log \lambda_{\text{away},i} &= \mu + \text{att}_{a(i)} + \text{def}_{h(i)} + \beta_{\text{pre}} \cdot \text{pre}_{a(i)}
\end{aligned}
\]

where:

- \(\mu\): global intercept;
- `home`: shared home advantage parameter;
- \(\text{att}_j\): attacking strength of team \(j\);
- \(\text{def}_j\): defensive strength of team \(j\);
- \(\text{pre}_j\): preseason strength covariate for team \(j\), constructed from
  `data/teams_premier_2025_2026.xlsx` via `backend.utils.build_preseason_strength`;
- \(\beta_{\text{pre}}\): regression coefficient for preseason strength.

**Hierarchical priors**

Team parameters share hierarchical priors (implemented in `backend/models.py`):

- `att_j ~ Normal(0, σ_att)`
- `def_j ~ Normal(0, σ_def)`
- `σ_att, σ_def ~ HalfNormal` (positive scale parameters)
- `home ~ Normal(0, 0.5)`
- `β_pre ~ Normal(0, 0.5)`

Soft sum-to-zero constraints on attack and defence (sum of attacks ≈ 0, sum of
defences ≈ 0) ensure identifiability, as in the book’s Poisson models.

This hierarchical structure induces:

- **shrinkage** across teams (extreme teams are pulled toward the league average unless supported by data);
- **posterior correlation** across matches (because they share the same latent team parameters).

**Implementation details**

- Code: class `PoissonGoalsModel` in `backend/models.py`.
- Fitting: the `fit` method builds a PyMC model with Poisson likelihood on
  `home_goals` and `away_goals` from `data/premier_results_raw_2025_2026.csv`,
  and samples from the posterior using NUTS.
- Integration: `backend/simulate_league.py` instantiates this class, calls `fit`,
  evaluates scoring metrics, runs Monte Carlo simulations and writes JSON outputs.

**Match-level prediction**

Given a fitted model, `predict_match(home_team, away_team)`:

1. Extracts posterior samples of \(\lambda_{\text{home}}\) and \(\lambda_{\text{away}}\)
   for the requested teams (using the learned `att`, `def`, `home`, `β_pre`, and preseason strengths).
2. For each posterior sample, computes Poisson probabilities over a goal grid
   (e.g. 0–10 goals per side).
3. Aggregates these to obtain predictive probabilities:

   - `p_home = P(home win)`
   - `p_draw = P(draw)`
   - `p_away = P(away win)`

4. Also returns `exp_home_goals` and `exp_away_goals` as the posterior mean of
   the Poisson rates.

These predictions for the next matchweek are written into:

- `data/matchday_predictions_poisson.json` (via `backend/simulate_league.py`,
  after renaming from `matchday_predictions.json`).

**Season simulation**

`simulate_season(current_table, remaining_fixtures, n_sim)`:

- Uses the Poisson predictive distribution for each remaining fixture.
- Simulates `n_sim` full seasons by:
  - sampling a score for every future match,
  - updating the league table,
  - storing final points and rank for each team.

Aggregated simulation results are exported to:

- `data/premier_outcomes_poisson_2000.json`

with, for each team:

- mean and standard deviation of final points,
- 5th / 25th / 50th / 75th / 95th percentiles of points,
- probability of finishing in each rank (1–20),
- most likely rank.

This model is intentionally simple and close to the standard “independent
Poisson” textbook example, and serves as a baseline against which the more
advanced Skellam and Student-t models can be compared.



## 3.**Skellam goal-difference model with draw inflation (`SkellamGoalDiffModel`)**

The Skellam model is built on the same hierarchical structure as the Poisson
model but focuses directly on the **goal difference** rather than modelling home
and away goals separately.

The construction takes inspiration from the Skellam and bivariate Poisson models
in *Predictive Modelling for Football Analytics* (Egidi, Karlis, Ntzoufras),
especially the discussion on goal-difference distributions and draw inflation.

**From Poisson goals to Skellam goal difference**

If home and away goals for a match were independent Poisson:

\[
Y_{\text{home}} \sim \text{Poisson}(\lambda_1), \quad
Y_{\text{away}} \sim \text{Poisson}(\lambda_2),
\]

then the goal difference

\[
Z = Y_{\text{home}} - Y_{\text{away}}
\]

follows a **Skellam distribution**:

\[
Z \sim \text{Skellam}(\lambda_1, \lambda_2).
\]

In this project:

- The log-linear structure for \(\lambda_1 = \lambda_{\text{home}}\) and
  \(\lambda_2 = \lambda_{\text{away}}\) is **identical** to the Poisson model:

  \[
  \begin{aligned}
  \log \lambda_{\text{home},i} &= \mu + \text{home} + \text{att}_{h(i)} + \text{def}_{a(i)} + \beta_{\text{pre}} \cdot \text{pre}_{h(i)} \\
  \log \lambda_{\text{away},i} &= \mu + \text{att}_{a(i)} + \text{def}_{h(i)} + \beta_{\text{pre}} \cdot \text{pre}_{a(i)}
  \end{aligned}
  \]

- The same hierarchical priors are used for `att`, `def`, `home`, `β_pre`,
  `σ_att`, `σ_def` as in the Poisson model.

The crucial difference is **how we turn \((\lambda_1,\lambda_2)\) into win/draw/loss
probabilities**.

**Implementation of Skellam probabilities**

In `backend/models.py`, the class `SkellamGoalDiffModel`:

- reuses the hierarchical structure and PyMC fitting machinery;
- but for prediction, uses the Skellam distribution for the goal difference.

Given `λ_home` and `λ_away` for a match, we compute:

- `P(Z = k)` for integer goal differences `k` using the Skellam pmf
  (via `scipy.stats.skellam.pmf` in the helper `_skellam_probs`);
- outcome probabilities:

  ```text
  p_draw = P(Z = 0)
  p_home = P(Z > 0)
  p_away = P(Z < 0)





## 4.**Student-t goal-difference model with draw inflation (`StudentTGoalDiffModel`)**

The Student-t model extends the Skellam approach by allowing **heavier tails**
in the distribution of goal differences. This addresses the fact that real
football data sometimes show more large wins/losses than a Poisson/Skellam
model would predict.

The construction is inspired by the discussion of over-dispersed and heavy-tailed
models in *Predictive Modelling for Football Analytics* (Egidi, Karlis, Ntzoufras),
where Student-t-like models for goal differences and dynamic variants are
considered.

**Core structure**

The Student-t model shares the **same hierarchical backbone** as the Poisson
and Skellam models:

- log-linear predictors for `λ_home` and `λ_away`:

  \[
  \begin{aligned}
  \log \lambda_{\text{home},i} &= \mu + \text{home} + \text{att}_{h(i)} + \text{def}_{a(i)} + \beta_{\text{pre}} \cdot \text{pre}_{h(i)} \\
  \log \lambda_{\text{away},i} &= \mu + \text{att}_{a(i)} + \text{def}_{h(i)} + \beta_{\text{pre}} \cdot \text{pre}_{a(i)}
  \end{aligned}
  \]

- hierarchical priors on `att`, `def`, `home`, `β_pre`, `σ_att`, `σ_def` as in
  the other models.

The difference lies in how we model the **goal difference**:

\[
Z_i = Y_{\text{home},i} - Y_{\text{away},i}.
\]

Instead of assuming that `Z_i` is Skellam (implied by independent Poisson
goals), we allow for heavier tails.

**Heavy-tailed goal-difference likelihood**

In `backend/models.py`, the class `StudentTGoalDiffModel` constructs a
goal-difference–based likelihood with a Student-t–like shape. Conceptually:

- compute an expected goal difference for match \(i\), e.g.

  \[
  \delta_i = \mathbb{E}[Z_i] \approx f(\lambda_{\text{home},i}, \lambda_{\text{away},i})
  \]

  (for instance, a function of the Poisson means);

- model the observed goal difference as

  \[
  Z_i \sim \text{Student-\(t\)}(\text{location} = \delta_i,\ \text{scale} = \tau,\ \text{df} = \nu)
  \]

  where:

  - `τ` is a scale parameter,
  - `ν` (degrees of freedom) controls tail thickness (smaller ν → heavier tails).

In practice, this is implemented through a PyMC likelihood that approximates a
Student-t distribution for the goal difference, while still being driven by
the same underlying attack/defence parameters and home advantage as the
Poisson/Skellam models.

The key effect is:

- the **mean structure** of the goal difference is similar,
- the variance and tail behaviour are more flexible, allowing more probability
  mass for larger |Z|.

**Draw-inflation re-calibration**

Even with heavier tails, a Student-t goal-difference model does not automatically
fix the draw rate bias. Therefore we reuse the **draw-inflation re-calibration**
introduced for the Skellam model.

During `fit`, `StudentTGoalDiffModel`:

1. Computes baseline win/draw/loss probabilities from the fitted
   Student-t-like distribution of `Z`:

   ```text
   p_draw = P(Z = 0)
   p_home = P(Z > 0)
   p_away = P(Z < 0)
