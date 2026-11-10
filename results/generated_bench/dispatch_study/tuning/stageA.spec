# Stage A: one factor at a time around adapt 215 (flags 12352), tune split.
# GA defaults: pop 32, n_gen 60, ls_moves 8 (child LS ls_moves/2), shortlist 24, restart_patience 6, xover 2 (MSXF)
T_ils       | dispatch |
T_ilslong   | dispatch | {"n_kicks": 120, "kick_patience": 0}
T_ga        | memetic  | {"flags": 12352, "adapt": 215}
T_gen15     | memetic  | {"flags": 12352, "adapt": 215, "n_gen": 15}
T_gen30     | memetic  | {"flags": 12352, "adapt": 215, "n_gen": 30}
T_gen120    | memetic  | {"flags": 12352, "adapt": 215, "n_gen": 120}
T_pop16     | memetic  | {"flags": 12352, "adapt": 215, "pop_size": 16}
T_pop64     | memetic  | {"flags": 12352, "adapt": 215, "pop_size": 64}
T_ls4       | memetic  | {"flags": 12352, "adapt": 215, "ls_moves": 4}
T_ls16      | memetic  | {"flags": 12352, "adapt": 215, "ls_moves": 16}
T_ls32      | memetic  | {"flags": 12352, "adapt": 215, "ls_moves": 32}
T_sl12      | memetic  | {"flags": 12352, "adapt": 215, "shortlist": 12}
T_sl48      | memetic  | {"flags": 12352, "adapt": 215, "shortlist": 48}
T_sl96      | memetic  | {"flags": 12352, "adapt": 215, "shortlist": 96}
T_rp3       | memetic  | {"flags": 12352, "adapt": 215, "restart_patience": 3}
T_rp12      | memetic  | {"flags": 12352, "adapt": 215, "restart_patience": 12}
T_x2pt      | memetic  | {"flags": 12352, "adapt": 215, "xover_mode": 0}
T_xnone     | memetic  | {"flags": 12352, "adapt": 215, "xover_mode": 1}
