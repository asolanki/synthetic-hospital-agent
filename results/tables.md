## Agent baselines

| Model | Split | n | Agent F1 (mean reward) | 95% CI | Official F1 (strict) | Precision | Weighted recall | Recall, earlier visits | Recall, final visit | Acuity acc | Valid submit % | Actions | Tokens | Budget exhausted % |
|---|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| qwen3-30b-strongprompt-dev | dev | 120 | 0.439 | 0.377–0.501 | 0.456 | 0.457 | 0.456 | 0.493 | 0.214 | 0.705 | 100.0 | 7.7 | 39,689 | 0.0 |
| qwen3.5-9b-nothink-bf16-dev | dev | 120 | 0.415 | 0.344–0.483 | 0.441 | 0.514 | 0.410 | 0.470 | 0.171 | 0.800 | 96.7 | 16.2 | 118,804 | 5.8 |
| qwen_qwen3-30b-a3b-instruct-2507-dev | dev | 120 | 0.463 | 0.405–0.525 | 0.494 | 0.544 | 0.467 | 0.487 | 0.200 | 0.730 | 98.3 | 11.2 | 59,280 | 0.0 |

## Task/reward audit (dev, n=120)

| Strategy | Group | n | Mean reward | 95% CI | Official F1 | Official F1 (strict) | Precision | Weighted recall | Acuity acc | Mean # predictions |
|---|---|--:|--:|---|--:|--:|--:|--:|--:|--:|
| oracle | oracle | 120 | 1.000 | 1.000–1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3.5 |
| empty submission | invalid | 120 | 0.000 | 0.000–0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — | 0.0 |
| malformed submission | invalid | 120 | 0.000 | 0.000–0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — | 0.0 |
| profile problem list (view_problem_list) | extraction | 120 | 0.138 | 0.102–0.186 | 0.275 | 0.166 | 0.902 | 0.162 | 0.905 | 1.3 |
| note problem list, dated entries | extraction | 120 | 0.859 | 0.827–0.888 | 0.875 | 0.875 | 0.985 | 0.787 | 0.434 | 2.9 |
| note problem list, all entries | extraction | 120 | 0.861 | 0.828–0.889 | 0.876 | 0.876 | 0.976 | 0.795 | 0.433 | 3.7 |
| all chart-mentioned diagnosis names | extraction | 120 | 0.412 | 0.381–0.443 | 0.443 | 0.443 | 0.298 | 0.859 | 0.440 | 16.5 |
| top-5 train categories | prior | 120 | 0.029 | 0.015–0.044 | 0.032 | 0.032 | 0.030 | 0.034 | 0.562 | 5.0 |
| top-20 train categories | prior | 120 | 0.021 | 0.014–0.030 | 0.022 | 0.022 | 0.013 | 0.062 | 0.419 | 20.0 |
| top-100 train categories | prior | 120 | 0.017 | 0.013–0.020 | 0.017 | 0.017 | 0.009 | 0.229 | 0.375 | 100.0 |
| scorer's 11 critical categories | prior | 120 | 0.012 | 0.004–0.019 | 0.012 | 0.012 | 0.008 | 0.028 | 1.000 | 11.0 |
| oracle + 5 wrong codes | perturbation | 120 | 0.518 | 0.487–0.547 | 0.537 | 0.537 | 0.367 | 1.000 | 1.000 | 8.5 |
| oracle + 20 wrong codes | perturbation | 120 | 0.236 | 0.214–0.256 | 0.245 | 0.245 | 0.140 | 1.000 | 1.000 | 23.5 |
| oracle + all chart-neutral categories (n=40) | perturbation | 40 | 1.000 | 1.000–1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 10.0 |
| oracle, acuity flipped | perturbation | 120 | 1.000 | 1.000–1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 3.5 |
| oracle, 3-char categories only | perturbation | 120 | 0.921 | 0.877–0.954 | 0.921 | 0.921 | 0.927 | 0.915 | 0.992 | 3.5 |
| oracle, every entry duplicated | perturbation | 120 | 0.748 | 0.731–0.766 | 0.756 | 0.756 | 0.608 | 1.000 | 1.000 | 7.0 |
| oracle minus final-encounter diagnoses | perturbation | 120 | 0.880 | 0.853–0.902 | 0.895 | 0.895 | 1.000 | 0.810 | 1.000 | 2.9 |

## Task/reward audit (train, n=680)

| Strategy | Group | n | Mean reward | 95% CI | Official F1 | Official F1 (strict) | Precision | Weighted recall | Acuity acc | Mean # predictions |
|---|---|--:|--:|---|--:|--:|--:|--:|--:|--:|
| oracle | oracle | 680 | 1.000 | 1.000–1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3.6 |
| empty submission | invalid | 680 | 0.000 | 0.000–0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — | 0.0 |
| malformed submission | invalid | 680 | 0.000 | 0.000–0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — | 0.0 |
| profile problem list (view_problem_list) | extraction | 680 | 0.145 | 0.127–0.160 | 0.282 | 0.172 | 0.921 | 0.167 | 0.926 | 1.4 |
| note problem list, dated entries | extraction | 680 | 0.818 | 0.804–0.832 | 0.839 | 0.835 | 0.977 | 0.736 | 0.408 | 2.9 |
| note problem list, all entries | extraction | 680 | 0.821 | 0.806–0.836 | 0.843 | 0.840 | 0.966 | 0.747 | 0.407 | 3.8 |
| all chart-mentioned diagnosis names | extraction | 680 | 0.375 | 0.361–0.389 | 0.403 | 0.403 | 0.267 | 0.820 | 0.416 | 17.5 |
| top-5 train categories | prior | 680 | 0.037 | 0.030–0.047 | 0.040 | 0.040 | 0.037 | 0.045 | 0.728 | 5.0 |
| top-20 train categories | prior | 680 | 0.030 | 0.027–0.034 | 0.032 | 0.032 | 0.019 | 0.094 | 0.470 | 20.0 |
| top-100 train categories | prior | 680 | 0.023 | 0.021–0.025 | 0.023 | 0.023 | 0.012 | 0.314 | 0.436 | 100.0 |
| scorer's 11 critical categories | prior | 680 | 0.021 | 0.017–0.025 | 0.022 | 0.022 | 0.014 | 0.048 | 0.981 | 11.0 |
| oracle + 5 wrong codes | perturbation | 680 | 0.532 | 0.519–0.546 | 0.549 | 0.549 | 0.379 | 1.000 | 1.000 | 8.6 |
| oracle + 20 wrong codes | perturbation | 680 | 0.245 | 0.235–0.254 | 0.253 | 0.253 | 0.145 | 1.000 | 1.000 | 23.6 |
| oracle + all chart-neutral categories (n=40) | perturbation | 40 | 1.000 | 1.000–1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 11.9 |
| oracle, acuity flipped | perturbation | 680 | 1.000 | 1.000–1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 3.6 |
| oracle, 3-char categories only | perturbation | 680 | 0.908 | 0.894–0.923 | 0.909 | 0.909 | 0.917 | 0.901 | 0.985 | 3.6 |
| oracle, every entry duplicated | perturbation | 680 | 0.743 | 0.734–0.750 | 0.752 | 0.752 | 0.602 | 1.000 | 1.000 | 7.2 |
| oracle minus final-encounter diagnoses | perturbation | 680 | 0.845 | 0.833–0.856 | 0.865 | 0.861 | 1.000 | 0.763 | 1.000 | 2.9 |
