# Two-centre federation benchmark (fedhpc.geo_gen)

32 instances from `bench.jsonl` (1 infeasible). Load H, output level M unless the name says `iH`. `n` = nodes per on-prem type per centre (2 centres × {c40, c64}, so n50/n150 = class A's n100/n300 in total capacity). Gurobi: MIPGap 0, 6 threads, 3600 s per solve, lex phase-2 Presolve {0}. † = a lex anchor is not certified optimal (time limit); values are incumbents.

## 1. Effect of inter-centre transfer (paired: same jobs, capacity, cloud)

Rows of one (workload, n) differ only in the S1↔S2 link. `cols`: SpaceTime columns with / without the dominance reductions. Root gaps: LP relaxation vs certified optimum of min f1 and of the λ = 0.5 scalar ĝ. f1_T / f1_0: minimum turnaround with / without cloud spending (job-slots). λ=.5: the weighted-sum optimum's f1, cost ($), jobs at the *other* centre / in the cloud, and staging slots paid.

| wl | n | WAN | cols | cols plain | build s | Gurobi s | root f1 % | root ĝ % | f1_T | f1_0 | λ=.5 f1 | λ=.5 $ | other/cloud | stage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fedhpc | 50 | co-loc. | 1.05M | 1.57M | 3.5 | 62 | 0.000 | 0.000 | 11,151 | 17,392 | 11,838 | 1,169 | 230/474 | 29 |
| fedhpc | 50 | 10 Gb/s | 1.05M | 1.57M | 4.3 | 41 | 0.000 | 0.000 | 11,151 | 17,392 | 11,838 | 1,169 | 226/474 | 37 |
| fedhpc | 50 | 1 Gb/s | 1.05M | 1.57M | 2.6 | 25 | 0.000 | 0.000 | 11,174 | 17,392 | 11,923 | 1,118 | 149/475 | 54 |
| fedhpc | 150 | co-loc. | 1.05M | 1.57M | 4.6 | 67 | 0.000 | 0.000 | 11,151 | 11,417 | 11,287 | 76 | 400/112 | 5 |
| fedhpc | 150 | 10 Gb/s | 1.05M | 1.57M | 6.9 | 80 | 0.000 | 0.000 | 11,151 | 11,417 | 11,287 | 76 | 387/112 | 13 |
| fedhpc | 150 | 1 Gb/s | 1.05M | 1.57M | 6.6 | 60 | 0.000 | 0.000 | 11,151 | 11,417 | 11,287 | 76 | 324/112 | 44 |
| pos† | 50 | co-loc. | 3.47M | 5.17M | 17.0 | 4,874 | 0.000 | 0.000 | 44,352 | 139,986 | 63,381 | 6,796 | 547/2079 | 129 |
| pos† | 50 | 10 Gb/s | 3.47M | 5.17M | 12.1 | 5,538 | 0.000 | 0.000 | 44,352 | 139,981 | 63,381 | 6,796 | 532/2079 | 151 |
| pos† | 50 | 1 Gb/s | 3.47M | 5.17M | 25.9 | 4,892 | 0.000 | 0.000 | 44,393 | 139,994 | 63,381 | 6,796 | 347/2079 | 200 |
| pos | 150 | co-loc. | 3.47M | 5.17M | 12.6 | 1,090 | 0.000 | 0.000 | 44,164 | 60,318 | 49,846 | 2,508 | 1155/917 | 45 |
| pos | 150 | 10 Gb/s | 3.47M | 5.17M | 12.6 | 276 | 0.000 | 0.000 | 44,164 | 60,318 | 49,846 | 2,508 | 1111/917 | 72 |
| pos | 150 | 1 Gb/s | 3.47M | 5.17M | 12.1 | 291 | 0.000 | 0.000 | 44,164 | 60,318 | 49,846 | 2,508 | 948/917 | 159 |

## 2a. Value of federating — no cloud (1 Gb/s WAN), min turnaround f1

`iso` = isolated centres, each job only at its origin. Heuristic policies: per-centre FCFS earliest start (isolated) and the Slurm-federation-inspired earliest-start policy over both centres (transfer-blind). Optimised: GA-mid (mean of 3 seeds) and the MILP over the federation; co-loc. = the same federation with free, instant transfer (a lower bound on what any transfer model allows).

| wl | n | iso MILP | iso FCFS | fed earliest-start | fed GA-mid | fed MILP | co-loc. MILP | MILP fed vs iso % |
|---|---|---|---|---|---|---|---|---|
| fedhpc | 50 | 18,528 | 22,475 | 20,792 | 17,609 | 17,392 | 17,392 | 6.1 |
| fedhpc | 150 | 11,608 | 11,962 | 11,713 | 11,504 | 11,417 | 11,417 | 1.6 |
| pos | 50 | infeasible | 311,419 (35 unpl.) | 295,924 (15 unpl.) | 607,513 | 140,010† | 139,987† | – |
| pos | 150 | 63,591 | 94,809 (1 unpl.) | 86,880 (1 unpl.) | 139,808 | 60,318 | 60,318 | 5.1 |

## 2b. Value of federating — with cloud (1 Gb/s WAN)

Cost of the minimum-turnaround schedule (f2_T), the best turnaround and the cost of the cheapest schedule (f1_0, f2_0), isolated vs federated, and the weighted-sum optima as f1/$ (iso → fed). Isolated centres share one cloud account.

| wl | n | iso f2_T $ | iso f1_0 | iso f2_0 $ | fed f2_T $ | fed f1_0 | fed f2_0 $ | λ .25 | λ .5 | λ .75 |
|---|---|---|---|---|---|---|---|---|---|---|
| fedhpc | 50 | 3,405 | 18,528 | 0 | 3,155 | 17,392 | 0 | 15,905/300 → 14,901/340 | 12,587/1,097 → 11,923/1,118 | 11,520/2,038 → 11,466/1,490 |
| fedhpc | 150 | 330 | 11,608 | 0 | 200 | 11,417 | 0 | 11,551/10 → 11,417/0 | 11,377/87 → 11,287/76 | 11,174/273 → 11,151/200 |
| pos | 50 | 21,373† | 434,515† | 212 | 21,323† | 139,994† | 0 | 142,445/214 → 100,779/2,257 | 121,800/1,164 → 63,381/6,796 | 69,978/5,880 → 51,450/11,215 |
| pos | 150 | 7,334 | 63,591 | 0 | 6,799 | 60,318 | 0 | 60,683/308 → 59,261/118 | 50,291/2,582 → 49,846/2,508 | 44,365/6,804 → 44,294/6,521 |

## 3. Heuristics vs Gurobi (gap % to the best known ĝ; λ = .25 / .5 / .75)

GA-mid = the paper's memetic GA (pop 16, 120 generations; mean of seeds 42/1/7), given Gurobi's exact reference points (*benchmark mode*). *deploy* = GA-mid seed 42 on reference points it estimates itself (no Gurobi anywhere). Dispatch = the ILS heuristic. Threshold = the best threshold-bursting baseline. Times per λ.

| instance | GA-mid | GA-mid deploy | dispatch ILS | threshold | Gurobi s | GA-mid s |
|---|---|---|---|---|---|---|
| fedhpc_n150_lH_iM_w0_fedC | 0.0 / 0.0 / 6.7 | 0.4 / 5.6 / 7.9 | 0.0 / 0.4 / 7.3 | 83 / 71 / 88 | 8.8 | 3.4 |
| fedhpc_n150_lH_iM_w10_fedC | 1.0 / 0.0 / 6.1 | 1.1 / 3.0 / 6.8 | 18.6 / 0.4 / 7.3 | 84 / 73 / 144 | 6.7 | 2.8 |
| fedhpc_n150_lH_iM_w1_fedC | 21.6 / 3.1 / 5.1 | 12.5 / 7.2 / 6.6 | 23.0 / 3.4 / 15.5 | 102 / 88 / 219 | 5.1 | 2.8 |
| fedhpc_n150_lH_iM_w1_isoC | 4.3 / 3.4 / 0.7 | 6.9 / 7.4 / 3.8 | 8.2 / 3.8 / 8.7 | 83 / 48 / 75 | 0.7 | 1.2 |
| fedhpc_n50_lH_iH_w0_fedC | 0.9 / 0.8 / 1.7 | 0.9 / 0.8 / 1.4 | 7.2 / 7.2 / 6.3 | 68 / 197 / 396 | 5.8 | 2.6 |
| fedhpc_n50_lH_iH_w1_fedC | 8.8 / 12.4 / 12.7 | 7.6 / 12.6 / 11.3 | 28.1 / 17.5 / 21.6 | 78 / 226 / 426 | 6.0 | 3.4 |
| fedhpc_n50_lH_iM_w0_fedC | 0.6 / 2.4 / 1.5 | 0.7 / 2.5 / 1.7 | 3.8 / 3.9 / 9.0 | 83 / 251 / 179 | 2.2 | 1.9 |
| fedhpc_n50_lH_iM_w10_fedC | 0.9 / 2.8 / 1.9 | 0.9 / 3.0 / 1.5 | 4.5 / 4.4 / 7.0 | 83 / 252 / 180 | 1.8 | 1.8 |
| fedhpc_n50_lH_iM_w1_fedC | 5.2 / 7.2 / 4.2 | 12.6 / 9.0 / 2.9 | 7.5 / 9.6 / 10.1 | 64 / 219 / 220 | 2.3 | 2.4 |
| fedhpc_n50_lH_iM_w1_isoC | 0.8 / 1.0 / 1.3 | 3.5 / 2.8 / 1.4 | 2.8 / 5.5 / 3.5 | 69 / 191 / 144 | 1.6 | 1.7 |
| pos_n150_lH_iM_w0_fedC | 15.2 / 19.5 / 4.4 | 15.4 / 24.0 / 3.8 | 24.0 / 21.1 / 8.5 | 505 / 207 / 138 | 23.0 | 8.4 |
| pos_n150_lH_iM_w10_fedC | 16.0 / 21.1 / 7.5 | 10.5 / 24.2 / 6.4 | 24.2 / 21.5 / 8.3 | 527 / 221 / 144 | 17.6 | 9.8 |
| pos_n150_lH_iM_w1_fedC | 18.4 / 22.8 / 7.4 | 11.4 / 24.6 / 3.7 | 25.8 / 23.7 / 10.0 | 592 / 259 / 180 | 15.6 | 9.2 |
| pos_n150_lH_iM_w1_isoC | 6.6 / 7.7 / 2.5 | 9.9 / 15.5 / 2.7 | 20.7 / 18.6 / 5.3 | 454 / 210 / 132 | 9.8 | 5.9 |
| pos_n50_lH_iH_w0_fedC† | 62.1 / 26.2 / 13.4 | 71.5 / 29.8 / 16.6 | 63.1 / 26.5 / 13.9 | 566 / 300 / 193 | 132.8 | 22.8 |
| pos_n50_lH_iH_w1_fedC† | 65.0 / 29.9 / 17.0 | 80.8 / 37.6 / 25.4 | 68.2 / 33.3 / 22.7 | 572 / 304 / 195 | 94.1 | 28.7 |
| pos_n50_lH_iM_w0_fedC† | 63.1 / 21.5 / 6.8 | 39.9 / 24.1 / 8.5 | 38.3 / 21.8 / 7.2 | 329 / 155 / 81 | 269.0 | 23.2 |
| pos_n50_lH_iM_w10_fedC† | 63.1 / 21.5 / 7.0 | 38.1 / 24.7 / 9.3 | 42.5 / 22.0 / 7.3 | 333 / 158 / 82 | 161.7 | 18.4 |
| pos_n50_lH_iM_w1_fedC† | 62.5 / 21.9 / 7.5 | 40.7 / 25.5 / 9.7 | 47.0 / 22.6 / 8.7 | 339 / 161 / 86 | 149.0 | 24.3 |
| pos_n50_lH_iM_w1_isoC† | 210.3 / 37.8 / 30.0 | 297.0 / 100.9 / 38.0 | 230.4 / 49.5 / 31.4 | 1429 / 429 / 181 | 1,251.6 | 12.4 |

| subset | cells | GA-mid mean % | median | max | GA = BKS | deploy mean | dispatch mean | Gurobi mean |
|---|---|---|---|---|---|---|---|---|
| all | 60 | 17.24 | 7.11 | 210.3 | 2 | 20.27 | 20.41 | 0.000 |
| W1 (fedhpc) | 30 | 3.96 | 2.13 | 21.6 | 2 | 4.87 | 8.54 | 0.000 |
| W2 (pos) | 30 | 30.52 | 20.31 | 210.3 | 0 | 35.67 | 32.27 | 0.000 |
| fed + cloud, WAN > 0 | 30 | 16.05 | 8.15 | 65.0 | 0 | 15.70 | 18.89 | 0.000 |
| co-located control or iso | 30 | 18.43 | 4.38 | 210.3 | 2 | 24.84 | 21.93 | 0.000 |

## 4. End-to-end time for three operating points (critique §4.2)

Gurobi: two lex anchors plus three weighted sums. Gurobi-free: heuristic reference points (once) plus GA-mid at three λ.

| instance | Gurobi anchors s | Gurobi 3×WS s | Gurobi total s | heur. refs s | Gurobi-free total s |
|---|---|---|---|---|---|
| fedhpc_n150_lH_iM_w0_fedC | 40 | 26 | 67 | 47 | 57 |
| fedhpc_n150_lH_iM_w10_fedC | 60 | 20 | 80 | 57 | 65 |
| fedhpc_n150_lH_iM_w1_fedC | 45 | 15 | 60 | 22 | 30 |
| fedhpc_n150_lH_iM_w1_isoC | 11 | 2 | 13 | 6 | 10 |
| fedhpc_n50_lH_iH_w0_fedC | 95 | 17 | 112 | 46 | 54 |
| fedhpc_n50_lH_iH_w1_fedC | 44 | 18 | 62 | 73 | 83 |
| fedhpc_n50_lH_iM_w0_fedC | 56 | 7 | 62 | 51 | 57 |
| fedhpc_n50_lH_iM_w10_fedC | 36 | 5 | 41 | 24 | 30 |
| fedhpc_n50_lH_iM_w1_fedC | 18 | 7 | 25 | 19 | 25 |
| fedhpc_n50_lH_iM_w1_isoC | 13 | 5 | 18 | 11 | 15 |
| pos_n150_lH_iM_w0_fedC | 1,021 | 69 | 1,090 | 256 | 278 |
| pos_n150_lH_iM_w10_fedC | 223 | 53 | 276 | 358 | 385 |
| pos_n150_lH_iM_w1_fedC | 244 | 47 | 291 | 352 | 377 |
| pos_n150_lH_iM_w1_isoC | 48 | 30 | 78 | 104 | 123 |
| pos_n50_lH_iH_w0_fedC† | 4,080 | 398 | 4,479 | 798 | 867 |
| pos_n50_lH_iH_w1_fedC† | 4,577 | 282 | 4,859 | 586 | 681 |
| pos_n50_lH_iM_w0_fedC† | 4,067 | 807 | 4,874 | 631 | 697 |
| pos_n50_lH_iM_w10_fedC† | 5,053 | 485 | 5,538 | 360 | 417 |
| pos_n50_lH_iM_w1_fedC† | 4,445 | 447 | 4,892 | 699 | 763 |
| pos_n50_lH_iM_w1_isoC† | 3,609 | 3,755 | 7,363 | 163 | 201 |

## 5. Instance checks (critique §3.6)

A free option for every job is necessary, not sufficient, for a free schedule: f2_0 > 0 means the free capacity cannot finish every job inside the horizon.

| instance | free option per job | f2_0 $ | volumes capped | in TB | out TB | origin-only jobs |
|---|---|---|---|---|---|---|
| fedhpc_n150_lH_iM_w0_fedC | True | 0.00 | 0 | 3.3 | 20.9 | 19 |
| fedhpc_n150_lH_iM_w0_fedN | True | 0.00 | 0 | 3.3 | 20.9 | 19 |
| fedhpc_n150_lH_iM_w10_fedC | True | 0.00 | 0 | 3.3 | 20.9 | 19 |
| fedhpc_n150_lH_iM_w1_fedC | True | 0.00 | 0 | 3.3 | 20.9 | 19 |
| fedhpc_n150_lH_iM_w1_fedN | True | 0.00 | 0 | 3.3 | 20.9 | 19 |
| fedhpc_n150_lH_iM_w1_isoC | True | 0.00 | 0 | 3.3 | 20.9 | 19 |
| fedhpc_n150_lH_iM_w1_isoN | True | 0.00 | 0 | 3.3 | 20.9 | 964 |
| fedhpc_n50_lH_iH_w0_fedC | True | 0.00 | 0 | 3.3 | 142.2 | 19 |
| fedhpc_n50_lH_iH_w1_fedC | True | 0.00 | 0 | 3.3 | 142.2 | 19 |
| fedhpc_n50_lH_iM_w0_fedC | True | 0.00 | 0 | 3.3 | 20.9 | 19 |
| fedhpc_n50_lH_iM_w0_fedN | True | 0.00 | 0 | 3.3 | 20.9 | 19 |
| fedhpc_n50_lH_iM_w10_fedC | True | 0.00 | 0 | 3.3 | 20.9 | 19 |
| fedhpc_n50_lH_iM_w1_fedC | True | 0.00 | 0 | 3.3 | 20.9 | 19 |
| fedhpc_n50_lH_iM_w1_fedN | True | 0.00 | 0 | 3.3 | 20.9 | 19 |
| fedhpc_n50_lH_iM_w1_isoC | True | 0.00 | 0 | 3.3 | 20.9 | 19 |
| fedhpc_n50_lH_iM_w1_isoN | True | 0.00 | 0 | 3.3 | 20.9 | 964 |
| pos_n150_lH_iM_w0_fedC | True | 0.00 | 0 | 9.1 | 147.5 | 146 |
| pos_n150_lH_iM_w0_fedN | True | 0.00 | 0 | 9.1 | 147.5 | 146 |
| pos_n150_lH_iM_w10_fedC | True | 0.00 | 0 | 9.1 | 147.5 | 146 |
| pos_n150_lH_iM_w1_fedC | True | 0.00 | 0 | 9.1 | 147.5 | 146 |
| pos_n150_lH_iM_w1_fedN | True | 0.00 | 0 | 9.1 | 147.5 | 146 |
| pos_n150_lH_iM_w1_isoC | True | 0.00 | 0 | 9.1 | 147.5 | 146 |
| pos_n150_lH_iM_w1_isoN | True | 0.00 | 0 | 9.1 | 147.5 | 3340 |
| pos_n50_lH_iH_w0_fedC | True | 0.00† | 4 | 9.1 | 640.1 | 146 |
| pos_n50_lH_iH_w1_fedC | True | 0.00† | 4 | 9.1 | 639.4 | 146 |
| pos_n50_lH_iM_w0_fedC | True | 0.00† | 0 | 9.1 | 147.5 | 146 |
| pos_n50_lH_iM_w0_fedN | True | 0.00† | 0 | 9.1 | 147.5 | 146 |
| pos_n50_lH_iM_w10_fedC | True | 0.00† | 0 | 9.1 | 147.5 | 146 |
| pos_n50_lH_iM_w1_fedC | True | 0.00† | 0 | 9.1 | 147.5 | 146 |
| pos_n50_lH_iM_w1_fedN | True | 0.00† | 0 | 9.1 | 147.5 | 146 |
| pos_n50_lH_iM_w1_isoC | True | 212.25† | 0 | 9.1 | 147.5 | 146 |
| pos_n50_lH_iM_w1_isoN | – | infeasible | 0 | 9.1 | 147.5 | 3340 |
