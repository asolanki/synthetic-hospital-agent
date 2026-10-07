## Agent baselines

| Model | Split | n | Agent F1 (mean reward) | 95% CI | Official F1 (strict) | Precision | Weighted recall | Recall, earlier visits | Recall, final visit | Acuity acc | Valid submit % | Actions | Tokens | Budget exhausted % |
|---|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| qwen3-30b-strongprompt-dev | dev | 120 | 0.439 | 0.377–0.501 | 0.456 | 0.457 | 0.456 | 0.493 | 0.214 | 0.705 | 100.0 | 7.7 | 39,689 | 0.0 |
| qwen3.5-9b-nothink-bf16-dev | dev | 120 | 0.415 | 0.344–0.483 | 0.441 | 0.514 | 0.410 | 0.470 | 0.171 | 0.800 | 96.7 | 16.2 | 118,804 | 5.8 |
| qwen_qwen3-30b-a3b-instruct-2507-dev | dev | 120 | 0.463 | 0.405–0.525 | 0.494 | 0.544 | 0.467 | 0.487 | 0.200 | 0.730 | 98.3 | 11.2 | 59,280 | 0.0 |
| gemma-3-27b-env-public | public | 200 | 0.235 | 0.198–0.276 | 0.262 | 0.297 | 0.251 | 0.267 | 0.176 | 0.717 | 96.5 | 4.3 | 137,696 | 22.6 |
| gpt-6-luna-env-public | public | 200 | 0.654 | 0.620–0.691 | 0.689 | 0.683 | 0.696 | 0.703 | 0.654 | 0.801 | 100.0 | 15.0 | 10,757 | 0.0 |
| kimi-k2.5-env-public | public | 200 | 0.693 | 0.656–0.730 | 0.726 | 0.778 | 0.681 | 0.698 | 0.581 | 0.821 | 100.0 | 29.0 | 54,813 | 19.5 |
| qwen3-30b-env-dev-t0 | dev | 120 | 0.420 | 0.357–0.481 | 0.446 | 0.499 | 0.423 | 0.447 | 0.200 | 0.725 | 97.5 | 12.9 | 63,451 | 2.5 |
| qwen3-30b-env-public-t0 | public | 200 | 0.427 | 0.380–0.474 | 0.457 | 0.494 | 0.433 | 0.445 | 0.279 | 0.749 | 99.0 | 11.8 | 33,302 | 2.0 |
| qwen3-30b-icd10-dev-t0 | dev | 120 | 0.599 | 0.550–0.656 | 0.634 | 0.738 | 0.663 | 0.594 | 0.257 | 0.732 | 90.8 | 8.1 | 40,409 | 0.0 |
| qwen3-30b-icd10-public-t0 | public | 200 | 0.614 | 0.572–0.659 | 0.644 | 0.726 | 0.635 | 0.645 | 0.375 | 0.766 | 95.0 | 8.1 | 52,895 | 0.0 |
| qwen3-30b-verifiers-dev | dev | 120 | 0.466 | 0.406–0.531 | 0.500 | 0.549 | 0.459 | 0.499 | 0.157 | 0.731 | 100.0 | 12.5 | 32,090 | 0.0 |
| qwen3-4b-icd10-dev | dev | 120 | 0.482 | 0.425–0.539 | 0.547 | 0.650 | 0.473 | 0.527 | 0.157 | 0.683 | 100.0 | 7.0 | 137,046 | 29.2 |
| qwen3-4b-icd10-public | public | 200 | 0.420 | 0.373–0.468 | 0.486 | 0.625 | 0.411 | 0.473 | 0.206 | 0.717 | 98.0 | 6.6 | 132,498 | 26.5 |
| qwen3-4b-paper-dev | dev | 120 | 0.102 | 0.072–0.136 | 0.113 | 0.122 | 0.115 | 0.153 | 0.043 | 0.843 | 95.8 | 8.8 | 202,109 | 50.0 |
| qwen3-4b-paper-public | public | 200 | 0.158 | 0.120–0.198 | 0.172 | 0.210 | 0.153 | 0.178 | 0.081 | 0.790 | 97.5 | 9.0 | 203,347 | 49.2 |
| qwen3-4b-rl-paper-dev | dev | 120 | 0.406 | 0.338–0.473 | 0.413 | 0.429 | 0.398 | 0.501 | 0.114 | 0.921 | 100.0 | 3.0 | 7,577 | 0.0 |
| qwen3-4b-rl-paper-public | public | 200 | 0.387 | 0.340–0.433 | 0.398 | 0.434 | 0.367 | 0.453 | 0.235 | 0.873 | 100.0 | 3.0 | 7,567 | 0.0 |
| qwen3-4b-rl-paper-train | train | 680 | 0.502 | 0.476–0.528 | 0.511 | 0.538 | 0.487 | 0.560 | 0.342 | 0.931 | 100.0 | 3.0 | 7,575 | 0.0 |
| qwen3-4b-sft-continued-paper-dev | dev | 120 | 0.477 | 0.414–0.538 | 0.485 | 0.498 | 0.472 | 0.550 | 0.143 | 0.948 | 100.0 | 3.0 | 7,581 | 0.0 |
| qwen3-4b-sft-continued-paper-public | public | 200 | 0.425 | 0.381–0.468 | 0.436 | 0.456 | 0.418 | 0.506 | 0.272 | 0.916 | 100.0 | 3.0 | 7,593 | 0.0 |
| qwen3-4b-sft-continued-paper-train | train | 680 | 0.696 | 0.672–0.721 | 0.702 | 0.718 | 0.689 | 0.757 | 0.544 | 0.964 | 99.9 | 3.0 | 7,568 | 0.0 |
| qwen3-4b-sft-icd10-dev-seed0 | dev | 120 | 0.762 | 0.716–0.805 | 0.770 | 0.779 | 0.762 | 0.882 | 0.286 | 0.899 | 100.0 | 3.0 | 12,678 | 0.0 |
| qwen3-4b-sft-icd10-public | public | 200 | 0.726 | 0.689–0.764 | 0.738 | 0.759 | 0.718 | 0.854 | 0.412 | 0.876 | 100.0 | 3.0 | 12,650 | 0.0 |
| qwen3-4b-sft-paper-dev | dev | 120 | 0.396 | 0.331–0.463 | 0.402 | 0.412 | 0.398 | 0.490 | 0.114 | 0.897 | 99.2 | 3.0 | 7,579 | 0.0 |
| qwen3-4b-sft-paper-public | public | 200 | 0.379 | 0.331–0.429 | 0.388 | 0.416 | 0.364 | 0.455 | 0.228 | 0.877 | 100.0 | 3.0 | 7,569 | 0.0 |
| qwen3-4b-sft-paper-train | train | 680 | 0.482 | 0.458–0.506 | 0.489 | 0.509 | 0.471 | 0.545 | 0.333 | 0.937 | 100.0 | 3.0 | 7,577 | 0.0 |
| qwen3-4b-sft-strong_v1-dev | dev | 120 | 0.394 | 0.330–0.461 | 0.401 | 0.410 | 0.392 | 0.484 | 0.100 | 0.882 | 100.0 | 3.0 | 8,646 | 0.0 |
| qwen3-4b-sft-sweep-e1-paper-dev-seed0 | dev | 120 | 0.230 | 0.177–0.285 | 0.235 | 0.254 | 0.219 | 0.314 | 0.071 | 0.908 | 100.0 | 3.0 | 7,608 | 0.0 |
| qwen3-4b-sft-sweep-e10-paper-dev-seed0 | dev | 120 | 0.555 | 0.495–0.611 | 0.562 | 0.567 | 0.557 | 0.660 | 0.129 | 0.921 | 100.0 | 3.0 | 7,583 | 0.0 |
| qwen3-4b-sft-sweep-e10-paper-train | train | 680 | 0.986 | 0.980–0.991 | 0.986 | 0.989 | 0.984 | 0.993 | 0.962 | 0.995 | 100.0 | 3.0 | 7,585 | 0.0 |
| qwen3-4b-sft-sweep-e2-paper-dev-seed0 | dev | 120 | 0.334 | 0.277–0.393 | 0.342 | 0.365 | 0.321 | 0.429 | 0.057 | 0.912 | 100.0 | 3.0 | 7,579 | 0.0 |
| qwen3-4b-sft-sweep-e3-paper-dev-seed0 | dev | 120 | 0.442 | 0.381–0.500 | 0.452 | 0.478 | 0.428 | 0.530 | 0.114 | 0.913 | 100.0 | 3.0 | 7,576 | 0.0 |
| qwen3-4b-sft-sweep-e3-paper-train | train | 680 | 0.609 | 0.584–0.633 | 0.618 | 0.652 | 0.588 | 0.673 | 0.407 | 0.944 | 100.0 | 3.0 | 7,589 | 0.0 |
| qwen3-4b-sft-sweep-e4-paper-dev-seed0 | dev | 120 | 0.492 | 0.431–0.552 | 0.502 | 0.523 | 0.483 | 0.591 | 0.143 | 0.899 | 100.0 | 3.0 | 7,577 | 0.0 |
| qwen3-4b-sft-sweep-e5-paper-dev-seed0 | dev | 120 | 0.545 | 0.487–0.604 | 0.555 | 0.569 | 0.542 | 0.631 | 0.171 | 0.905 | 100.0 | 3.0 | 7,612 | 0.0 |
| qwen3-4b-sft-sweep-e6-paper-dev-seed0 | dev | 120 | 0.536 | 0.474–0.597 | 0.545 | 0.557 | 0.543 | 0.643 | 0.171 | 0.924 | 99.2 | 3.0 | 7,536 | 0.0 |
| qwen3-4b-sft-sweep-e6-paper-train | train | 680 | 0.887 | 0.871–0.902 | 0.889 | 0.899 | 0.880 | 0.921 | 0.764 | 0.983 | 100.0 | 3.0 | 7,586 | 0.0 |
| qwen3-4b-sft-sweep-e7-paper-dev-seed0 | dev | 120 | 0.567 | 0.510–0.626 | 0.576 | 0.585 | 0.567 | 0.666 | 0.171 | 0.919 | 100.0 | 3.0 | 7,583 | 0.0 |
| qwen3-4b-sft-sweep-e8-icd10-dev-seed0 | dev | 120 | 0.567 | 0.508–0.628 | 0.573 | 0.581 | 0.565 | 0.674 | 0.143 | 0.911 | 100.0 | 3.0 | 8,043 | 0.0 |
| qwen3-4b-sft-sweep-e8-icd10-public | public | 200 | 0.517 | 0.474–0.561 | 0.529 | 0.542 | 0.516 | 0.608 | 0.331 | 0.943 | 100.0 | 3.0 | 8,033 | 0.0 |
| qwen3-4b-sft-sweep-e8-paper-dev-seed0 | dev | 120 | 0.573 | 0.518–0.633 | 0.579 | 0.583 | 0.576 | 0.677 | 0.143 | 0.923 | 100.0 | 3.0 | 7,583 | 0.0 |
| qwen3-4b-sft-sweep-e8-paper-public | public | 200 | 0.520 | 0.477–0.564 | 0.532 | 0.550 | 0.519 | 0.599 | 0.324 | 0.940 | 99.5 | 3.0 | 7,571 | 0.0 |
| qwen3-4b-sft-sweep-e9-paper-dev-seed0 | dev | 120 | 0.566 | 0.509–0.625 | 0.573 | 0.576 | 0.569 | 0.674 | 0.143 | 0.919 | 100.0 | 3.0 | 7,583 | 0.0 |
| qwen3-4b-strong_v1-dev | dev | 120 | 0.107 | 0.074–0.143 | 0.116 | 0.117 | 0.120 | 0.159 | 0.014 | 0.704 | 97.5 | 5.9 | 299,994 | 54.6 |

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

## Task/reward audit (public, n=200)

| Strategy | Group | n | Mean reward | 95% CI | Official F1 | Official F1 (strict) | Precision | Weighted recall | Acuity acc | Mean # predictions |
|---|---|--:|--:|---|--:|--:|--:|--:|--:|--:|
| oracle | oracle | 200 | 1.000 | 1.000–1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3.5 |
| empty submission | invalid | 200 | 0.000 | 0.000–0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — | 0.0 |
| malformed submission | invalid | 200 | 0.000 | 0.000–0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — | 0.0 |
| profile problem list (view_problem_list) | extraction | 200 | 0.126 | 0.094–0.162 | 0.266 | 0.146 | 0.920 | 0.156 | 0.957 | 1.2 |
| note problem list, dated entries | extraction | 200 | 0.804 | 0.771–0.831 | 0.828 | 0.824 | 0.965 | 0.725 | 0.408 | 2.8 |
| note problem list, all entries | extraction | 200 | 0.807 | 0.775–0.834 | 0.830 | 0.826 | 0.957 | 0.733 | 0.407 | 3.6 |
| all chart-mentioned diagnosis names | extraction | 200 | 0.385 | 0.361–0.410 | 0.419 | 0.419 | 0.282 | 0.814 | 0.414 | 16.7 |
| top-5 train categories | prior | 200 | 0.028 | 0.017–0.039 | 0.029 | 0.029 | 0.027 | 0.032 | 0.826 | 5.0 |
| top-20 train categories | prior | 200 | 0.023 | 0.016–0.029 | 0.024 | 0.024 | 0.014 | 0.078 | 0.481 | 20.0 |
| top-100 train categories | prior | 200 | 0.020 | 0.017–0.023 | 0.020 | 0.020 | 0.010 | 0.283 | 0.440 | 100.0 |
| scorer's 11 critical categories | prior | 200 | 0.019 | 0.013–0.026 | 0.020 | 0.020 | 0.013 | 0.042 | 1.000 | 11.0 |
| oracle + 5 wrong codes | perturbation | 200 | 0.528 | 0.506–0.551 | 0.545 | 0.545 | 0.374 | 1.000 | 1.000 | 8.5 |
| oracle + 20 wrong codes | perturbation | 200 | 0.241 | 0.226–0.259 | 0.249 | 0.249 | 0.142 | 1.000 | 1.000 | 23.5 |
| oracle + all chart-neutral categories (n=40) | perturbation | 40 | 1.000 | 1.000–1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 8.2 |
| oracle, acuity flipped | perturbation | 200 | 1.000 | 1.000–1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 3.5 |
| oracle, 3-char categories only | perturbation | 200 | 0.913 | 0.880–0.938 | 0.914 | 0.914 | 0.919 | 0.908 | 0.986 | 3.5 |
| oracle, every entry duplicated | perturbation | 200 | 0.739 | 0.727–0.753 | 0.749 | 0.749 | 0.598 | 1.000 | 1.000 | 7.0 |
| oracle minus final-encounter diagnoses | perturbation | 200 | 0.841 | 0.814–0.862 | 0.864 | 0.860 | 1.000 | 0.761 | 1.000 | 2.8 |

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
