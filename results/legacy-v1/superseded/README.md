Artifacts from earlier runs, kept only as provenance. Nothing here is current
and nothing here is referenced by the README.

round4_cpu_metrics.json   The previous full run, before the zz ansatz depth was
                          pinned to ry's. Superseded by results/metrics.json.
stale_round4_figures/     scores.png and sculpting.png from that run. They were
                          NOT regenerated for the current run -- the Colab
                          session lacked pylatexenc, so Qiskit's circuit drawer
                          raised inside make_all() before those two figures were
                          drawn, and the per-event scores they need were not
                          saved. Moved here rather than left in results/figures/,
                          where they would have looked like current output.
