# Ablation of GA-mid (T_sl48p16g120: flags 12352, adapt 215 = 1 mu grid + 2 deadline bit + 4 decoder bit + 16 converged seeds + 64 niche + 128 two seeds), tune split; one change each
A_noseed      | memetic | {"flags": 12352, "shortlist": 48, "pop_size": 16, "n_gen": 120, "adapt": 71}
A_seedall     | memetic | {"flags": 12352, "shortlist": 48, "pop_size": 16, "n_gen": 120, "adapt": 87}
A_noniche     | memetic | {"flags": 12352, "shortlist": 48, "pop_size": 16, "n_gen": 120, "adapt": 151}
A_noedf       | memetic | {"flags": 64, "shortlist": 48, "pop_size": 16, "n_gen": 120, "adapt": 215}
A_edfdl       | memetic | {"flags": 4160, "shortlist": 48, "pop_size": 16, "n_gen": 120, "adapt": 215}
A_nomugrid    | memetic | {"flags": 12352, "shortlist": 48, "pop_size": 16, "n_gen": 120, "adapt": 214}
A_nodlbit     | memetic | {"flags": 12352, "shortlist": 48, "pop_size": 16, "n_gen": 120, "adapt": 213}
A_nodecbit    | memetic | {"flags": 12352, "shortlist": 48, "pop_size": 16, "n_gen": 120, "adapt": 211}
A_noxover     | memetic | {"flags": 12352, "shortlist": 48, "pop_size": 16, "n_gen": 120, "adapt": 215, "ablate": 32}
A_nomut       | memetic | {"flags": 12352, "shortlist": 48, "pop_size": 16, "n_gen": 120, "adapt": 215, "ablate": 1024}
A_nokick      | memetic | {"flags": 12352, "shortlist": 48, "pop_size": 16, "n_gen": 120, "adapt": 215, "ablate": 64}
A_noheur      | memetic | {"flags": 12352, "shortlist": 48, "pop_size": 16, "n_gen": 120, "adapt": 215, "ablate": 1}
A_polishfull  | memetic | {"flags": 12352, "shortlist": 48, "pop_size": 16, "n_gen": 120, "adapt": 247}
