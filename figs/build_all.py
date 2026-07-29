"""Build all five NHB main figures. Run: python build_all.py
Assumes the data exports already exist (see README). Re-runs each figure script."""
import runpy, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for name in ["fig1_reliability", "fig2_invariants", "fig3_satisfice",
             "fig4_model", "fig5_capacity"]:
    print(f"--- {name} ---")
    runpy.run_path(os.path.join(HERE, f"{name}.py"), run_name="__main__")
print("all figures rebuilt into outputs/")
