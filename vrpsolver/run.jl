# Build and solve a fedhpc model (exported by fedhpc.vrpsolver.export_model) with
# VRPSolver. Usage:
#
#   julia --project=vrpsolver vrpsolver/run.jl model.json result.json \
#         [--dry-run] [--time-limit SECONDS] [--no-rank1]
#
# --dry-run builds the VRPSolver model and stops before VrpOptimizer, which is
# the first call into BaPCod: it needs neither CPLEX nor BAPCOD_RCSP_LIB.
# See docs/vrpsolver_integration.md and the module docstring of
# src/fedhpc/vrpsolver.py for the model.

using JSON, JuMP, VrpSolver

const FORMAT = "fedhpc-vrpsolver/1"

struct Built
    model::VrpModel
    x::Vector{JuMP.VariableRef}
    s::Vector{JuMP.VariableRef}
    path::Vector{NTuple{3,Int}}     # (job index 1-based, type, start) per x
    static::Vector{NTuple{3,Int}}   # same, per s
    n_graphs::Int
    n_arcs::Int
    n_packing::Int
    n_activation::Int
end

function build(m::Dict, rank1::Bool)::Built
    m["format"] == FORMAT || error("unknown model format $(m["format"])")
    H = Int(m["horizon"])
    n = length(m["jobs"])
    model = VrpModel()
    F = model.formulation

    # placements: (job, type, start, duration, objective, c1, c2)
    P = Tuple{Int,Int,Int,Int,Float64,Float64,Float64}[]
    for pt in m["path_types"], a in pt["arcs"]
        push!(P, (Int(a[1]) + 1, Int(pt["type"]), Int(a[2]), Int(a[3]),
                  Float64(a[4]), Float64(a[5]), Float64(a[6])))
    end
    S = [(Int(a[1]) + 1, Int(a[2]), Int(a[3]), Float64(a[4]), Float64(a[5]), Float64(a[6]))
         for a in m["statics"]]
    fixed = m["fixed"]

    @variable(F, x[k in 1:length(P)] >= 0, Int)   # mapped: lb 0, no ub
    @variable(F, s[k in 1:length(S)], Bin)         # master-only
    @variable(F, z[k in 1:length(fixed)], Bin)     # master-only
    @objective(F, Min,
        sum(P[k][5] * x[k] for k in eachindex(P); init = AffExpr(0.0)) +
        sum(S[k][4] * s[k] for k in eachindex(S); init = AffExpr(0.0)) +
        sum(Float64(fixed[k]["obj"]) * z[k] for k in eachindex(fixed); init = AffExpr(0.0)))

    xs_of = [Int[] for _ in 1:n]
    ss_of = [Int[] for _ in 1:n]
    for k in eachindex(P)
        push!(xs_of[P[k][1]], k)
    end
    for k in eachindex(S)
        push!(ss_of[S[k][1]], k)
    end
    @constraint(F, cover[j in 1:n],
        sum(x[k] for k in xs_of[j]; init = AffExpr(0.0)) +
        sum(s[k] for k in ss_of[j]; init = AffExpr(0.0)) == 1.0)
    if m["bound1"] !== nothing
        @constraint(F, cap1,
            sum(P[k][6] * x[k] for k in eachindex(P); init = AffExpr(0.0)) +
            sum(S[k][5] * s[k] for k in eachindex(S); init = AffExpr(0.0)) <= Float64(m["bound1"]))
    end
    if m["bound2"] !== nothing
        @constraint(F, cap2,
            sum(P[k][7] * x[k] for k in eachindex(P); init = AffExpr(0.0)) +
            sum(S[k][6] * s[k] for k in eachindex(S); init = AffExpr(0.0)) +
            sum(Float64(fixed[k]["cost"]) * z[k] for k in eachindex(fixed); init = AffExpr(0.0))
            <= Float64(m["bound2"]))
    end
    # fixed-cost links, one per (job, fixed-cost type) pair
    for (fk, f) in enumerate(fixed)
        mt = Int(f["type"])
        for j in 1:n
            ks = [k for k in xs_of[j] if P[k][2] == mt]
            qs = [k for k in ss_of[j] if S[k][2] == mt]
            (isempty(ks) && isempty(qs)) && continue
            @constraint(F, sum(x[k] for k in ks; init = AffExpr(0.0)) +
                           sum(s[k] for k in qs; init = AffExpr(0.0)) - z[fk] <= 0.0)
        end
    end

    # Activation rows (bpc._build_rmp's `activation`): u[m] counts the paths of
    # a fixed-cost path type (mapped to every graph's start arc), and
    # u[m] <= N_m z[m]. Valid: a schedule never needs an all-idle path.
    fixed_of = Dict(Int(f["type"]) => k for (k, f) in enumerate(fixed))
    act_types = [Int(pt["type"]) for pt in m["path_types"] if haskey(fixed_of, Int(pt["type"]))]
    @variable(F, u[mt in act_types] >= 0, Int)
    for pt in m["path_types"]
        mt = Int(pt["type"])
        haskey(fixed_of, mt) || continue
        npaths = sum(Int(rel[2]) for rel in pt["releases"])
        @constraint(F, u[mt] - npaths * z[fixed_of[mt]] <= 0.0)
    end

    # one graph per release group: a start vertex H+1, nodes r..H, wait arcs,
    # one arc per placement
    pjobs = [Int(j) + 1 for j in m["packing_jobs"]]
    ps_of = Dict(j => i for (i, j) in enumerate(pjobs))
    packing = [Tuple{VrpGraph,Int}[] for _ in pjobs]
    graphs = Tuple{VrpGraph,Vector{Vector{Float64}}}[]
    n_arcs = 0
    for pt in m["path_types"]
        mt = Int(pt["type"])
        dist = [Float64.(row) for row in pt["dist"]]
        for rel in pt["releases"]
            r, cnt = Int(rel[1]), Int(rel[2])
            start = H + 1
            G = VrpGraph(model, [start; collect(r:H)], start, H, (0, cnt))
            time_res = add_resource!(G; main = true)
            for v in [start; collect(r:H)]
                set_resource_bounds!(G, v, time_res, 0, H)
            end
            a0 = mt in act_types ? add_arc!(G, start, r, u[mt]) : add_arc!(G, start, r)
            set_arc_consumption!(G, a0, time_res, 0)
            n_arcs += 1
            for v in r:(H - 1)
                a = add_arc!(G, v, v + 1)
                set_arc_consumption!(G, a, time_res, 1)
                n_arcs += 1
            end
            for k in eachindex(P)
                (j, pm, t, p) = P[k][1:4]
                (pm == mt && t >= r) || continue
                a = add_arc!(G, t, t + p, x[k])
                set_arc_consumption!(G, a, time_res, p)
                push!(packing[ps_of[j]], (G, a))
                n_arcs += 1
            end
            add_graph!(model, G)
            push!(graphs, (G, dist))
        end
    end
    set_arc_packing_sets!(model, packing)
    for (G, dist) in graphs
        define_elementarity_sets_distance_matrix!(model, G, dist)
    end

    # branching: a job's type first, then its placement
    pairs = unique([(P[k][1], P[k][2]) for k in eachindex(P)])
    if !isempty(pairs)
        @expression(F, y[jm in pairs],
            sum(x[k] for k in xs_of[jm[1]] if P[k][2] == jm[2]))
        set_branching_priority!(model, y, "y", 2)
        set_branching_priority!(model, "x", 1)
    end
    rank1 && enable_rank1_cuts!(model)

    return Built(model, collect(x), collect(s),
                 [(P[k][1], P[k][2], P[k][3]) for k in eachindex(P)],
                 [(S[k][1], S[k][2], S[k][3]) for k in eachindex(S)],
                 length(graphs), n_arcs, length(packing), length(act_types))
end

function main(args)
    length(args) >= 2 || error("usage: run.jl model.json result.json [--dry-run] [--time-limit S] [--no-rank1]")
    src, out = args[1], args[2]
    dry = "--dry-run" in args
    rank1 = !("--no-rank1" in args)
    tl = nothing
    i = findfirst(==("--time-limit"), args)
    i === nothing || (tl = parse(Int, args[i + 1]))

    m = JSON.parsefile(src)
    t0 = time()
    b = build(m, rank1)
    result = Dict{String,Any}(
        "graphs" => b.n_graphs, "arcs" => b.n_arcs, "packing_sets" => b.n_packing,
        "path_variables" => length(b.x), "static_variables" => length(b.s),
        "activation_rows" => b.n_activation,
        "build_s" => time() - t0)
    if dry
        result["status"] = "dry_run"
        open(io -> JSON.print(io, result), out, "w")
        return
    end

    cfg = tempname() * ".cfg"
    cp(joinpath(@__DIR__, "config", "fedhpc.cfg"), cfg)
    open(cfg, "a") do io
        println(io, "RCSPrankOneCutsMaxNumRows = ", rank1 ? 100 : 0)
        tl === nothing || println(io, "GlobalTimeLimit = ", tl)
    end
    opt = VrpOptimizer(b.model, cfg, get(m, "name", ""))
    m["cutoff"] === nothing || set_cutoff!(opt, Float64(m["cutoff"]))
    t1 = time()
    status, found = optimize!(opt)
    result["solve_s"] = time() - t1
    result["status"] = string(status)
    result["found"] = found
    if found
        obj = m["objective"]
        v = get_objective_value(opt)
        result["vrp_objective"] = v
        result["objective"] = Float64(obj["scale"]) * v + Float64(obj["offset"])
        pl = Vector{Vector{Int}}()
        for (k, (j, mt, t)) in enumerate(b.path)
            get_value(opt, b.x[k]) > 0.5 && push!(pl, [j - 1, mt, t])
        end
        for (k, (j, mt, t)) in enumerate(b.static)
            get_value(opt, b.s[k]) > 0.5 && push!(pl, [j - 1, mt, t])
        end
        result["placements"] = pl
    end
    result["stats"] = Dict(string(k) => v for (k, v) in opt.stats)
    open(io -> JSON.print(io, result), out, "w")
end

main(ARGS)
