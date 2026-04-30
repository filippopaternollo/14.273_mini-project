"""
plot_joint_subsidy.jl — Render the two joint-subsidy heatmaps from cached CSVs.

Reads:
  - output/cache/joint_subsidy_gamma_hat.csv   (γ = γ̂ sweep)
  - output/cache/joint_subsidy_gamma_zero.csv  (γ = 0 counterfactual)

Writes:
  - output/figures/joint_subsidy_heatmap.pdf      (ΔΣW %, γ = γ̂, both argmaxes)
  - output/figures/joint_subsidy_heatmap_r3.pdf   (ΔW₃ %, γ = γ̂)

Standalone usage (after `run_joint_subsidy.jl` has cached the data):
    cd code
    julia --project=. scripts/plot_joint_subsidy.jl

Designed to also be `include`d at the end of `run_joint_subsidy.jl`. All
script-specific bindings live inside `plot_joint_subsidy_main()` so they
do not collide with constants defined by the caller (e.g. `TAU_FRAC`).
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Plots, CSV, DataFrames

function plot_joint_subsidy_main()
    output_dir = joinpath(@__DIR__, "../../output")
    cache_dir  = joinpath(output_dir, "cache")
    fig_dir    = joinpath(output_dir, "figures")
    mkpath(fig_dir)

    df_hat = CSV.read(joinpath(cache_dir, "joint_subsidy_gamma_hat.csv"), DataFrame)
    df_zg  = CSV.read(joinpath(cache_dir, "joint_subsidy_gamma_zero.csv"), DataFrame)

    tau_frac = sort!(unique(df_hat.tau_frac))
    psi_frac = sort!(unique(df_hat.psi_frac))
    nT, nP   = length(tau_frac), length(psi_frac)

    # Reshape long-form CSV to (nT × nP) matrix indexed by (τ-row, ψ-col)
    function pivot(df::DataFrame, col::Symbol)
        M = Matrix{Float64}(undef, nT, nP)
        for row in eachrow(df)
            i = findfirst(==(row.tau_frac), tau_frac)
            j = findfirst(==(row.psi_frac), psi_frac)
            M[i,j] = row[col]
        end
        M
    end

    dWtot_pct = pivot(df_hat, :dW_total_pct)
    dW_r3_pct = pivot(df_hat, :dW_r3_pct)

    i_best,    j_best    = Tuple(argmax(pivot(df_hat, :dW_total)))
    i_best_zg, j_best_zg = Tuple(argmax(pivot(df_zg,  :dW_total)))
    i_best_r3, j_best_r3 = Tuple(argmax(dW_r3_pct))

    # ── Heatmap helper ─────────────────────────────────────────────────────
    # Drops the rotated `colorbar_title` (which collides with tick labels)
    # and uses a transparent inset subplot positioned above the colorbar to
    # host the label. Annotation in data coords does not work because
    # Plots.jl expands the y-axis to encompass it, dragging the bar up.
    function draw_heatmap(M::Matrix{Float64}, label::AbstractString,
                          title::AbstractString)
        hmax = maximum(M)
        plt = heatmap(psi_frac, tau_frac, M;
                      xlabel = "Entry-subsidy fraction  ψ / φ̂",
                      ylabel = "Innovation-subsidy fraction  τ / κ̂",
                      title  = title,
                      c = cgrad(:Blues),
                      clims = (0.0, hmax),
                      colorbar_title = "",
                      size = (780, 560),
                      dpi = 200,
                      titlefontsize = 13, guidefontsize = 11,
                      tickfontsize = 10,
                      xticks = 0.0:0.20:1.00, yticks = 0.0:0.20:1.00,
                      framestyle = :box,
                      left_margin = 6Plots.mm, bottom_margin = 6Plots.mm,
                      right_margin = 12Plots.mm, top_margin = 12Plots.mm)
        plot!(plt, [0.0], [0.0];
              inset = (1, bbox(0.91, -0.06, 0.12, 0.06, :top, :left)),
              subplot = 2, framestyle = :none, ticks = nothing, axis = false,
              legend = false, background_color_inside = :transparent,
              background_color_subplot = :transparent,
              annotation = (0.5, 0.5, text(label, 10, :center)))
        plt
    end

    # ── ΔΣW (%) heatmap with both argmaxes overlaid ───────────────────────
    plt = draw_heatmap(dWtot_pct, "ΔΣW (%)",
                       "Aggregate welfare change ΔΣW (% of baseline)")
    scatter!(plt, [psi_frac[j_best]], [tau_frac[i_best]];
             marker = :star5, ms = 14, color = :gold,
             markerstrokecolor = :black, markerstrokewidth = 1.5,
             label = "Planner argmax  (γ = γ̂)",
             legend = :topleft, foreground_color_legend = nothing,
             background_color_legend = RGBA(1.0, 1.0, 1.0, 0.85))
    scatter!(plt, [psi_frac[j_best_zg]], [tau_frac[i_best_zg]];
             marker = :circle, ms = 9, color = :black,
             markerstrokecolor = :white, markerstrokewidth = 1.5,
             label = "Planner argmax  (γ = 0)")
    fig_path = joinpath(fig_dir, "joint_subsidy_heatmap.pdf")
    savefig(plt, fig_path)
    println("Saved figure: $fig_path")

    # ── ΔW₃ (%) heatmap ───────────────────────────────────────────────────
    plt3 = draw_heatmap(dW_r3_pct, "ΔW₃ (%)",
                        "Region-3 welfare change ΔW₃ (% of baseline)")
    scatter!(plt3, [psi_frac[j_best_r3]], [tau_frac[i_best_r3]];
             marker = :star5, ms = 14, color = :gold,
             markerstrokecolor = :black, markerstrokewidth = 1.5,
             label = "Region-3 argmax",
             legend = :topleft, foreground_color_legend = nothing,
             background_color_legend = RGBA(1.0, 1.0, 1.0, 0.85))
    fig_path_r3 = joinpath(fig_dir, "joint_subsidy_heatmap_r3.pdf")
    savefig(plt3, fig_path_r3)
    println("Saved figure: $fig_path_r3")
end

plot_joint_subsidy_main()
