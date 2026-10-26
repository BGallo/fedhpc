/**
 * FED-HPC — the weighted.hpp memetic GA ported onto the dispatch decoder.
 *
 * Same control flow as run_weighted (weighted.hpp): heuristic + random seeds,
 * binary tournament, MSXF / two-point crossover, mutation, a capped local
 * search on every child (ls_moves/2 moves, ls_moves for seeds and kicks,
 * ·3 for the final polish), elitism and an ILS kick of max(3, n/40) flips on
 * `restart_patience` stagnant generations. What changes is what was learned
 * building weighted_dispatch.hpp:
 *
 *   genome  — per job 0 = auto, 1 = cloud, 2 = on-prem (no slot index: the
 *             decoder picks the pool and the start, the scalar picks the cloud
 *             type), so pool↔pool and cloud↔cloud plateaus are gone;
 *   decoder — time-driven non-delay dispatch with deadline offload
 *             (`decode`), replacing schedule_repair + local_search;
 *   LS      — dispatch_ls over single on-prem/cloud flips. DF_OLD_EST keeps
 *             weighted.hpp's improving-estimate shortlist instead of the
 *             half-ranked / half-random one (DF_MIX_RANK);
 *   mu      — the λ-aware dispatch priority, picked once by decode-only pen_g
 *             (mu < 0) or fixed (mu = 0 is WSPT).
 *
 * Self-adaptive decoder settings (`adapt`, MAdapt bits): instead of fixing the
 * priority mix, the deadline rule and the decoder once for the run, every
 * individual carries its own (mu grid index, deadline on/off, dispatch/serial)
 * — spread over the initial population, inherited by crossover (MSXF treats
 * "adopt the other parent's settings" as one more step candidate), perturbed
 * by mutation and, with MA_LS_MOVES, also searched by the local search
 * (neighbouring settings tried at each local optimum). Motivation: the
 * decode-only mu choice is the wrong one at λ = 0.5 on the congested instance
 * and no single (mu, deadline) wins at every λ.
 *
 * The heuristic seeds of ga_common are mapped to genomes by type class
 * (cloud type → 1, on-prem → 0), as are make_random individuals. Every flag
 * is an ablation switch; the ABL_* bits of ga_common keep their meaning.
 *
 * Deterministic for a fixed seed and independent of n_threads: per-child RNGs
 * are drawn from the master RNG, and a child's local search decodes serially
 * on its own thread.
 */
#pragma once

#include "weighted_dispatch.hpp"

enum MAdapt {
    MA_MU       = 1 << 0,   // per-individual priority mix (index into the mu grid)
    MA_DEADLINE = 1 << 1,   // per-individual deadline offload on/off
    MA_DECODER  = 1 << 2,   // per-individual decoder: dispatch / serial SGS
    MA_LS_MOVES = 1 << 3,   // local search also tries neighbouring settings
};

struct MInd {
    std::vector<char> gen;
    DSched s;
    int mu = 0;              // index into the run's scalars
    char dl = 1, dec = 0;    // deadline offload on; 0 = dispatch, 1 = serial SGS
};

struct MemeticResult {
    std::vector<std::tuple<int,int>> asgn;
    DSched best;
    long long decodes = 0;
    int ls_calls = 0, kicks = 0;
    double mu = 0, total_ms = 0;
    int dl = 1, dec = 0;
    std::vector<int> mu_hist, dl_hist, dec_hist;    // final population settings
    std::vector<std::pair<double, double>> trace;   // (elapsed ms, best pen_g)
};

inline std::vector<char> genome_from_genes(const Problem& prob, const std::vector<int>& genes) {
    std::vector<char> g(prob.n_jobs);
    for (int j = 0; j < prob.n_jobs; j++)
        g[j] = prob.type_cap[prob.job_slots[j][genes[j]].type_id] < 0 ? 1 : 0;
    return g;
}

// One flip: an on-prem job goes to cloud, a cloud job comes back (auto, or
// forced on-prem if the decoder had offloaded it).
inline void flip_job(const DProblem& P, const DScalar& S, const MInd& x, std::vector<char>& g, int j) {
    if (x.s.on[j]) { if (S.cloud_opt[j] >= 0) g[j] = 1; }
    else if (!P.jobs[j].onprem.empty()) g[j] = (g[j] == 1) ? 0 : 2;
}

inline bool same_settings(const MInd& a, const MInd& b) {
    return a.mu == b.mu && a.dl == b.dl && a.dec == b.dec;
}

// Settings one step away from x: mu index ±1, deadline flipped, decoder flipped
// (only the adaptive ones).
inline void setting_neighbours(const MInd& x, int adapt, int n_mu, std::vector<MInd>& out) {
    out.clear();
    auto push = [&](int mu, char dl, char dec) {
        MInd y; y.mu = mu; y.dl = dl; y.dec = dec; out.push_back(std::move(y));
    };
    if (adapt & MA_MU) {
        if (x.mu > 0)        push(x.mu - 1, x.dl, x.dec);
        if (x.mu + 1 < n_mu) push(x.mu + 1, x.dl, x.dec);
    }
    if (adapt & MA_DEADLINE) push(x.mu, (char)!x.dl, x.dec);
    if (adapt & MA_DECODER)  push(x.mu, x.dl, (char)!x.dec);
}

inline MemeticResult run_weighted_memetic(
    const Problem& prob, const DProblem& P, double w1, double w2, double f1_cap,
    int pop_size, int n_gen, int seed, int n_threads, int ls_moves, int restart_patience,
    int shortlist, int xover_mode, int flags, double mu, int ablate, int adapt) {
    py::gil_scoped_release release;
    set_num_threads(n_threads);
    const TimePoint t0 = Clock::now();
    const bool abl_seeds = ablate & ABL_NO_HEURISTIC_SEEDS;
    const bool abl_unif  = ablate & ABL_UNIFORM_RANDOM;
    const bool abl_xover = ablate & ABL_NO_CROSSOVER;
    const bool abl_kick  = ablate & ABL_NO_ILS_KICK;
    const bool abl_elit  = ablate & ABL_NO_ELITISM;
    const bool abl_mut   = ablate & ABL_NO_MUTATION;

    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif
    std::vector<std::vector<DWork>> works(nt, std::vector<DWork>(1));
    std::vector<DWork> top_works(nt);           // for LS called outside the GA loop
    std::mt19937 rng(seed);
    MemeticResult R;

    // One scalar per priority mix the run can use: the whole grid when mu is
    // adaptive, else the fixed mu (or the best of the grid by all-auto decode).
    static const double grid[] = {0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0};
    const int n_grid = (int)(sizeof(grid) / sizeof(grid[0]));
    std::vector<DScalar> SS;
    std::vector<double> mu_val;
    {
        DScalar S = make_scalar(P, w1, w2, f1_cap);
        if (adapt & MA_MU) {
            for (double m : grid) { set_priority(P, S, m); SS.push_back(S); mu_val.push_back(m); }
        } else {
            if (mu < 0) {
                std::vector<char> g0(P.n, 0);
                double best = 0; mu = 0;
                for (double m : grid) {
                    set_priority(P, S, m);
                    DSched c;
                    decode(P, S, g0, flags, top_works[0], c);
                    R.decodes++;
                    if (m == grid[0] || c.pen_g < best - 1e-12) { best = c.pen_g; mu = m; }
                }
            }
            set_priority(P, S, mu);
            SS.push_back(std::move(S));
            mu_val.push_back(mu);
        }
    }
    const int n_mu = (int)SS.size();

    // Decoder flags of an individual's settings.
    auto fl = [&](const MInd& x) {
        int f = flags;
        if (adapt & MA_DEADLINE) f = x.dl ? (f & ~DF_NO_DEADLINE) : (f | DF_NO_DEADLINE);
        if (adapt & MA_DECODER)  f = (f & ~(DF_SERIAL | DF_BEST_OF)) | (x.dec ? DF_SERIAL : 0);
        return f;
    };
    auto run_decode = [&](MInd& x, DWork& W) { decode(P, SS[x.mu], x.gen, fl(x), W, x.s); };

    // Decode + capped local search of one individual on this thread; with
    // MA_LS_MOVES a setting step is taken at each local optimum (one move).
    auto improve = [&](MInd& x, int moves, std::mt19937& lrng, std::vector<DWork>& W,
                       long long& dec) {
        run_decode(x, W[0]); dec++;
        int left = moves;
        std::vector<MInd> nb;
        while (true) {
            DStats st;
            dispatch_ls(P, SS[x.mu], x.gen, x.s, fl(x), shortlist, left, lrng, W, st, t0);
            dec += st.decodes; left -= st.ls_moves;
            if (!(adapt & MA_LS_MOVES) || left <= 0) break;
            setting_neighbours(x, adapt, n_mu, nb);
            int bi = -1;
            for (int i = 0; i < (int)nb.size(); i++) {
                nb[i].gen = x.gen;
                run_decode(nb[i], W[0]); dec++;
                const double ref = bi < 0 ? x.s.pen_g : nb[bi].s.pen_g;
                if (nb[i].s.pen_g < ref - 1e-12) bi = i;
            }
            if (bi < 0) break;
            x = std::move(nb[bi]);
            left--;
        }
    };

    const double p_mut = 2.0 / std::max(P.n, 1);
    const double p_set = 0.2;   // per-child probability of a setting mutation
    std::vector<uint32_t> init_seeds(pop_size);
    for (auto& s : init_seeds) s = rng();
    auto h_seeds = make_seeds(prob, {}, abl_seeds);
    const int n_hs = std::min<int>((int)h_seeds.size(), pop_size);
    const char dl0 = !(flags & DF_NO_DEADLINE), dec0 = (flags & DF_SERIAL) ? 1 : 0;

    std::vector<MInd> pop(pop_size);
    long long dec = 0;
    int calls = 0;
    #pragma omp parallel for schedule(dynamic, 1) reduction(+:dec, calls)
    for (int k = 0; k < pop_size; k++) {
        std::mt19937 lrng(init_seeds[k]);
        MInd& x = pop[k];
        x.gen = genome_from_genes(prob, k < n_hs ? h_seeds[k].genes
                                                 : make_random(prob, lrng, abl_unif).genes);
        // spread the adaptive settings over the population
        int c = k;
        x.dl = dl0; x.dec = dec0;
        if (adapt & MA_MU)       { x.mu = c % n_mu; c /= n_mu; }
        if (adapt & MA_DEADLINE) { x.dl = (char)(1 - c % 2); c /= 2; }
        if (adapt & MA_DECODER)  { x.dec = (char)(c % 2); }
#ifdef _OPENMP
        auto& W = works[omp_get_thread_num()];
#else
        auto& W = works[0];
#endif
        improve(x, ls_moves, lrng, W, dec);
        calls++;
    }

    auto score = [](const MInd& x) { return x.s.pen_g; };
    int inc = 0;
    for (int k = 1; k < pop_size; k++) if (score(pop[k]) < score(pop[inc])) inc = k;
    MInd incumbent = pop[inc];
    R.trace.emplace_back(ms_since(t0), incumbent.s.pen_g);
    int stagnation = 0;

    std::vector<MInd> offspring(pop_size);
    std::vector<uint32_t> child_seeds(pop_size);
    const int gen_moves = std::max(1, ls_moves / 2);

    for (int g = 0; g < n_gen; g++) {
        for (auto& s : child_seeds) s = rng();
        const int k0 = abl_elit ? 0 : 1;
        if (!abl_elit) offspring[0] = incumbent;

        #pragma omp parallel for schedule(dynamic, 1) reduction(+:dec, calls)
        for (int k = k0; k < pop_size; k++) {
            std::mt19937 lrng(child_seeds[k]);
#ifdef _OPENMP
            auto& W = works[omp_get_thread_num()];
#else
            auto& W = works[0];
#endif
            std::uniform_int_distribution<int> ri(0, pop_size - 1);
            std::uniform_real_distribution<double> u(0.0, 1.0);
            auto pick = [&]() {
                const int a = ri(lrng), b = ri(lrng);
                return score(pop[a]) <= score(pop[b]) ? a : b;
            };
            MInd c;
            if (abl_xover || xover_mode == 1) {
                c = pop[pick()];
            } else if (xover_mode == 2) {
                // MSXF: walk from b toward a; each step adopts the best of up
                // to 12 sampled differing genes — or a's settings, as one more
                // candidate — keeping the best point on the path.
                const MInd& a = pop[pick()];
                const MInd& b = pop[pick()];
                MInd cur = b;
                c = b;
                std::vector<int> diff;
                for (int s = 0; s < gen_moves; s++) {
                    diff.clear();
                    for (int j = 0; j < P.n; j++) if (cur.gen[j] != a.gen[j]) diff.push_back(j);
                    const bool sdiff = !same_settings(cur, a);
                    if (diff.empty() && !sdiff) break;
                    const int K = std::min<int>((int)diff.size(), 12);
                    for (int i = 0; i < K; i++) {
                        std::uniform_int_distribution<int> di(i, (int)diff.size() - 1);
                        std::swap(diff[i], diff[di(lrng)]);
                    }
                    MInd step; bool have = false;
                    for (int i = (sdiff ? -1 : 0); i < K; i++) {
                        MInd w;
                        w.gen = cur.gen; w.mu = cur.mu; w.dl = cur.dl; w.dec = cur.dec;
                        if (i < 0) { w.mu = a.mu; w.dl = a.dl; w.dec = a.dec; }
                        else       w.gen[diff[i]] = a.gen[diff[i]];
                        run_decode(w, W[0]); dec++;
                        if (!have || w.s.pen_g < step.s.pen_g) { step = std::move(w); have = true; }
                    }
                    cur = std::move(step);
                    if (cur.s.pen_g < c.s.pen_g) c = cur;
                }
            } else {
                const MInd& p1 = pop[pick()]; const MInd& p2 = pop[pick()];
                c.gen = p1.gen;
                std::uniform_int_distribution<int> pt(1, std::max(1, P.n - 1));
                int c1 = pt(lrng), c2 = pt(lrng);
                if (c1 > c2) std::swap(c1, c2);
                for (int j = c1; j < c2; j++) c.gen[j] = p2.gen[j];
                c.mu = p1.mu; c.dl = p1.dl; c.dec = p1.dec;
                if (adapt & MA_MU)       c.mu  = (u(lrng) < 0.5 ? p1 : p2).mu;
                if (adapt & MA_DEADLINE) c.dl  = (u(lrng) < 0.5 ? p1 : p2).dl;
                if (adapt & MA_DECODER)  c.dec = (u(lrng) < 0.5 ? p1 : p2).dec;
            }
            if (!abl_mut) {
                for (int j = 0; j < P.n; j++)
                    if (u(lrng) < p_mut) c.gen[j] = (c.gen[j] == 1) ? 0 : 1;
                if (adapt & (MA_MU | MA_DEADLINE | MA_DECODER)) {
                    std::vector<MInd> nb;
                    setting_neighbours(c, adapt, n_mu, nb);
                    if (!nb.empty() && u(lrng) < p_set) {
                        const MInd& y = nb[std::uniform_int_distribution<int>(0, (int)nb.size() - 1)(lrng)];
                        c.mu = y.mu; c.dl = y.dl; c.dec = y.dec;
                    }
                }
            }
            improve(c, gen_moves, lrng, W, dec);
            calls++;
            offspring[k] = std::move(c);
        }
        pop.swap(offspring);

        bool advanced = false;
        for (int k = 0; k < pop_size; k++)
            if (score(pop[k]) < score(incumbent) - 1e-9) { incumbent = pop[k]; advanced = true; }
        if (advanced) R.trace.emplace_back(ms_since(t0), incumbent.s.pen_g);
        stagnation = advanced ? 0 : stagnation + 1;

        if (!abl_kick && restart_patience > 0 && stagnation >= restart_patience) {
            MInd kick = incumbent;
            const int nflips = std::max(3, P.n / 40);
            for (int f = 0; f < nflips; f++)
                flip_job(P, SS[incumbent.mu], incumbent, kick.gen, rng() % P.n);
            improve(kick, ls_moves, rng, top_works, dec);
            calls++; R.kicks++;
            pop[pop_size - 1] = kick;
            if (score(kick) < score(incumbent) - 1e-9) {
                incumbent = kick;
                R.trace.emplace_back(ms_since(t0), incumbent.s.pen_g);
            }
            stagnation = 0;
        }
    }

    // Final polish on the incumbent.
    {
        DStats st;
        dispatch_ls(P, SS[incumbent.mu], incumbent.gen, incumbent.s, fl(incumbent), shortlist,
                    ls_moves * 3, rng, top_works, st, t0);
        dec += st.decodes; calls++;
        R.trace.emplace_back(ms_since(t0), incumbent.s.pen_g);
    }

    R.decodes += dec;
    R.ls_calls = calls;
    R.mu = mu_val[incumbent.mu]; R.dl = incumbent.dl; R.dec = incumbent.dec;
    R.mu_hist.assign(n_mu, 0); R.dl_hist.assign(2, 0); R.dec_hist.assign(2, 0);
    for (const MInd& x : pop) { R.mu_hist[x.mu]++; R.dl_hist[x.dl]++; R.dec_hist[x.dec]++; }
    R.asgn.resize(P.n);
    for (int j = 0; j < P.n; j++) R.asgn[j] = {incumbent.s.type[j], incumbent.s.start[j]};
    R.best = std::move(incumbent.s);
    R.total_ms = ms_since(t0);
    return R;
}
