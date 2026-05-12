# EGGROLL O(1/r) Convergence Rate Verification

## What You Are

You are Claude Code. Read this entire file before doing anything. This is a self-contained
experiment plan. Your job is to implement it, run it, fix any errors, and save all results.
Do not ask for confirmation at any step. Just execute.

---

## What This Experiment Is

This is an empirical verification of Theorem 2 from the paper:
"Evolution Strategies at the Hyperscale" (arXiv:2511.16652).

The theorem states:

    || g_EGGROLL^r - g_True ||_F = O(1/r)

Meaning: as rank r increases, the EGGROLL gradient estimate converges to the true
ES gradient at rate 1/r. This is quadratically faster than the naive CLT rate of
1/sqrt(r).

We verify this empirically on a consumer GPU (RTX 3050, 4GB VRAM) by:
1. Computing a high-quality reference ES gradient
2. Computing EGGROLL estimates at varying ranks r
3. Plotting the error on a log-log scale and fitting the slope
4. Showing the slope is -1 (EGGROLL) not -0.5 (naive CLT)

If this works, it is an independent empirical confirmation of their theorem on
hardware the authors did not test on.

---

## Directory Structure To Create

```
eggroll_experiment/
    src/
        reference_gradient.py
        eggroll_gradient.py
        experiment.py
        plot.py
    results/
        data/
            errors.json
        figures/
            convergence_loglog.png
            convergence_loglog_annotated.png
            slope_fit.png
    requirements.txt
    run.py
    README.md
```

Create all of these. Results directory should be created at runtime if it doesn't exist.

---

## Requirements

Write requirements.txt with:
- torch
- numpy
- matplotlib
- scipy
- seaborn
- tqdm
- json (stdlib)

---

## Implementation Details

### Fitness Function

Use a quadratic fitness function:

    f(W) = -|| W - W_star ||_F^2

where W_star is a fixed target matrix sampled once at the start and frozen.

Why this choice:
- The true gradient is analytically known: grad f(mu) = -2(mu - W_star)
- This lets us verify against both the analytical answer AND the reference ES gradient
- No neural network needed, so no VRAM pressure
- Clean, interpretable results

### Weight Matrix

    mu: shape (128, 128), sampled from N(0, 0.01)
    W_star: shape (128, 128), sampled from N(0, 1), frozen

### Reference Gradient (ground truth)

Compute using full rank Gaussian ES with very large population:

    g_ref = (1 / (sigma * N_ref)) * sum_i [ f(mu + sigma * E_i) * E_i ]

where:
- E_i ~ N(0, I) full rank matrix, shape (128, 128)
- N_ref = 50000 (large enough to be stable)
- sigma = 0.01

Also compute analytical gradient:
    g_analytical = -2 * (mu - W_star)

Report the distance between g_ref and g_analytical as a sanity check.
If || g_ref - g_analytical ||_F is large, something is wrong. Stop and debug.

### EGGROLL Gradient Estimates

For each rank r in [1, 2, 4, 8, 16, 32, 64, 100]:

    For each trial t in range(N_trials=20):
        Compute EGGROLL estimate:
            g_eggroll = (1 / (sigma * N_eggroll)) * sum_i [ f(mu + sigma * E_i) * E_i ]
        where:
            A_i ~ N(0, 1), shape (128, r)
            B_i ~ N(0, 1), shape (128, r)
            E_i = (1/sqrt(r)) * A_i @ B_i.T   shape (128, 128)
            N_eggroll = 5000

        error_t = || g_eggroll - g_ref ||_F

    Store mean and std of error across trials for this r.

This gives us error(r) with uncertainty bands.

### Baseline CLT Estimate

For comparison, also compute what O(1/sqrt(r)) would look like.
Fit it to match at r=1 so the comparison is fair:

    baseline(r) = error(r=1) / sqrt(r)

### Slope Fitting

Fit a line to log(error) vs log(r) using scipy.stats.linregress.

    slope, intercept, r_value, p_value, std_err = linregress(log(r_values), log(mean_errors))

The fitted slope should be close to -1.
The CLT baseline slope is -0.5.

Print this to console clearly:
    "Fitted slope: {slope:.4f} (expected: -1.0)"
    "R^2: {r_value**2:.4f}"
    "Standard error: {std_err:.4f}"

---

## Batching Strategy for RTX 3050 (4GB VRAM)

Do NOT try to stack all N_ref=50000 perturbations at once. That will OOM.

Process in minibatches:

    BATCH_SIZE = 500  # for reference gradient
    BATCH_SIZE_EGGROLL = 500  # for eggroll

Accumulate the sum incrementally:

    total = torch.zeros_like(mu)
    for batch in range(N // BATCH_SIZE):
        E_batch = sample_perturbations(BATCH_SIZE)   # shape (BATCH_SIZE, 128, 128)
        f_batch = compute_fitness(mu, E_batch, sigma) # shape (BATCH_SIZE,)
        total += (f_batch.view(-1,1,1) * E_batch).sum(dim=0)
    g = total / (sigma * N)

Keep everything on GPU but never materialize more than BATCH_SIZE matrices at once.

---

## Plotting Instructions

### Figure 1: convergence_loglog.png

- X axis: rank r, log scale
- Y axis: || error ||_F, log scale
- Plot: EGGROLL mean error with shaded std band (color: deep blue)
- Plot: O(1/r) fitted line (color: orange, dashed)
- Plot: O(1/sqrt(r)) CLT baseline (color: red, dashed)
- Plot: analytical reference line with slope exactly -1 (color: gray, dotted)
- Add slope annotation on the EGGROLL line showing the fitted slope value
- Legend, grid, clean axes labels
- Title: "EGGROLL Gradient Error vs Rank r"

### Figure 2: convergence_loglog_annotated.png

Same as Figure 1 but add:
- Annotation box in the plot: "Fitted slope: {slope:.3f}, Expected: -1.0"
- Annotation box: "RTX 3050, 4GB VRAM"
- Annotation box: "128x128 weight matrix, N=5000 workers, 20 trials per r"

### Figure 3: slope_fit.png

- Scatter plot of log(r) vs log(mean_error)
- Overlay the fitted regression line
- Show R^2 and slope in the title
- This is the raw proof the relationship is linear in log-log space

Use seaborn style "whitegrid" for all figures.
Save at 300 DPI.
Use tight_layout().

---

## Data To Save

Save results/data/errors.json with this structure:

```json
{
    "experiment_config": {
        "matrix_size": [128, 128],
        "sigma": 0.01,
        "N_ref": 50000,
        "N_eggroll": 5000,
        "N_trials": 20,
        "r_values": [1, 2, 4, 8, 16, 32, 64, 100],
        "device": "cuda or cpu",
        "fitness_function": "quadratic: -||W - W_star||_F^2"
    },
    "sanity_check": {
        "ref_vs_analytical_error": 0.0,
        "passed": true
    },
    "results": {
        "r_values": [],
        "mean_errors": [],
        "std_errors": [],
        "clt_baseline": []
    },
    "slope_fit": {
        "fitted_slope": 0.0,
        "expected_slope": -1.0,
        "r_squared": 0.0,
        "std_err": 0.0,
        "intercept": 0.0
    }
}
```

---

## run.py

This is the single entry point. Running `python run.py` should:

1. Print "Setting up experiment..."
2. Create all result directories
3. Run reference gradient computation with progress bar
4. Run EGGROLL estimates for each r with progress bar
5. Run sanity check and print result
6. Fit slope and print result
7. Save errors.json
8. Generate all three figures
9. Print "Done. Results saved to results/"

Add a --device flag defaulting to "cuda" with fallback to "cpu" if cuda not available.
Add a --quick flag that uses N_ref=5000, N_eggroll=1000, N_trials=5 for a fast test run.

---

## README.md To Generate

Write a README that explains:
- What the experiment verifies (Theorem 2 from arXiv:2511.16652)
- How to run it (pip install -r requirements.txt, python run.py)
- What the expected output looks like
- What the slope result means
- Hardware used (RTX 3050 4GB)
- A results section with placeholder text "See results/figures/"

---

## Error Handling

- If CUDA OOM: catch the error, halve BATCH_SIZE, retry automatically, print a warning
- If sanity check fails (|| g_ref - g_analytical ||_F > 1.0): print a loud warning but continue
- If slope fit R^2 < 0.95: print a warning saying the relationship may not be clean
- Wrap the entire run in a try/except and save whatever results exist even if something fails midway

---

## Final Check Before You Start

Before writing any code:
1. Confirm torch is available
2. Confirm CUDA is available, if not fall back to CPU and print a warning
3. Print GPU name and VRAM if CUDA is available
4. Print estimated runtime

Estimated runtime on RTX 3050:
- Full run: ~15-25 minutes
- Quick run (--quick flag): ~2-3 minutes

## Mandatory Execution Order

You must follow this exact order. Do not skip steps.

Step 1: Run the quick version first.
    python run.py --quick

Step 2: After the quick run finishes, read the printed fitted slope.
    - If the slope is between -0.8 and -1.2, print "QUICK RUN PASSED. Starting full run." and proceed.
    - If the slope is outside that range, stop. Print "QUICK RUN FAILED. Slope is {value}. Debugging before full run."
      Then inspect the EGGROLL implementation. The most likely cause is missing the 1/sqrt(r) factor.
      Fix it and re-run the quick version. Do not proceed to the full run until the quick run passes.

Step 3: Only after quick run passes, run the full version.
    python run.py

Step 4: After the full run, print a final summary to console:
    "============================================"
    "EXPERIMENT COMPLETE"
    "Fitted slope    : {slope:.4f}  (target: -1.0)"
    "CLT baseline    : -0.5"
    "R squared       : {r2:.4f}"
    "Figures saved to: results/figures/"
    "Data saved to   : results/data/errors.json"
    "============================================"
    "Result: PASSED" if slope is between -0.8 and -1.2, else "Result: CHECK MANUALLY"

---

## The Single Most Important Thing

The log-log plot must show a clearly steeper slope than -0.5.
If the fitted slope is between -0.8 and -1.2, the experiment succeeded.
If it is close to -0.5, something is wrong with the EGGROLL implementation.
The most common mistake is forgetting the 1/sqrt(r) scaling in E = (1/sqrt(r)) * A @ B.T.
Double check this is in the code.
