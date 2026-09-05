"""
Prescient Coding Challenge 2026 -- pre-submission validator.

RUN THIS BEFORE YOU SUBMIT.  python validate.py

We score your solution on a window you have never seen, PLUS the four historical
windows below. If your solution breaks a constraint or crashes on any of them,
it scores nothing. This script runs every window we will run, so you can find
that out now rather than after the deadline.

A green board here does not mean you will win. It means you will be scored.
"""
from __future__ import annotations

import datetime as dt
import time

import harness
import solution


def main() -> None:
    data = harness.load_data()
    last = data[0].index.max().date()
    print(f"---> data available to {last}")
    print(f"---> the scoring window starts after this date and you do not have it\n")

    windows = {"practice (2025)": harness.PRACTICE_WINDOW, **harness.ROBUSTNESS_WINDOWS}

    print(f"{'window':>16} {'result':>10} {'excess':>9} {'IR':>7} {'active':>8} "
          f"{'turnover':>9} {'cost':>7}")
    print("-" * 73)

    failures = 0
    t0 = time.time()
    for label, (start, end) in windows.items():
        try:
            bt = harness.run_backtest(solution.generate_weights, solution.PARAMS,
                                      start=start, end=end, data=data, verbose=False)
        except Exception as exc:
            failures += 1
            first = str(exc).splitlines()[0]
            print(f"{label:>16} {'FAILED':>10}   {first[:60]}")
            continue
        m = harness.metrics(bt)
        print(f"{label:>16} {'ok':>10} {100 * m['excess_return']:8.2f}% "
              f"{m['information_ratio']:7.2f} {100 * m['mean_active_weight']:7.2f}% "
              f"{100 * m['avg_turnover']:8.2f}% {100 * m['cost_drag']:6.2f}%")

    elapsed = time.time() - t0
    print("-" * 73)
    print(f"---> declared parameters : {len(solution.PARAMS)}")
    print(f"---> total run time      : {elapsed:.1f}s "
          f"({'within' if elapsed < 600 else 'OVER'} the 10 minute limit)")

    # determinism: the same inputs must give the same answer twice
    a = harness.run_backtest(solution.generate_weights, solution.PARAMS,
                             *harness.PRACTICE_WINDOW, data=data, verbose=False)
    b = harness.run_backtest(solution.generate_weights, solution.PARAMS,
                             *harness.PRACTICE_WINDOW, data=data, verbose=False)
    stable = bool((a["port_return"] - b["port_return"]).abs().max() < 1e-12)
    print(f"---> deterministic       : {'yes' if stable else 'NO -- seed your randomness'}")

    if failures or not stable:
        print("\n*** NOT READY TO SUBMIT ***")
        print("Fix the failures above. A submission that fails any window scores nothing.")
    else:
        print("\nAll windows passed. You are ready to submit.")


if __name__ == "__main__":
    main()
