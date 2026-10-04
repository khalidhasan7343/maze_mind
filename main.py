"""
Maze Mind — entry point.

Wires the game loop to the logger and the model outputs (per the
architecture slide). Usage:

    python main.py play [--tag NAME] [--width N] [--height N]
    python main.py simulate [--rounds-per-profile N]
    python main.py train
    python main.py dashboard

First-time setup, in order:
    python main.py simulate     # generate synthetic training rounds
    python main.py train        # fit KNN / K-Means / logistic regression
    python main.py dashboard    # render report figures
    python main.py play         # play for real — round is logged too
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def cmd_play(args):
    from game.engine import RoundConfig
    from game.ui.pygame_ui import run_round
    from data.logger import RoundLogger
    from ml.knn_skill import skill_from_history

    logger = RoundLogger()
    skill_hint = skill_from_history(logger.fetch_by_player(args.tag))

    config = RoundConfig(
        width=args.width,
        height=args.height,
        round_time_limit_s=args.time_limit,
        seed=args.seed,
    )
    outcome = run_round(config, player_tag=args.tag, skill_hint=skill_hint, logger=logger,
                        initial_tier=args.difficulty)
    print(f"Round finished: {outcome.value}")


def cmd_simulate(args):
    from data.simulate import generate_dataset
    n = generate_dataset(n_per_profile=args.rounds_per_profile)
    print(f"Logged {n} synthetic rounds.")


def cmd_train(args):
    from ml.knn_skill import train as train_knn
    from ml.kmeans_strategy import train as train_kmeans
    from ml.logreg_winprob import train as train_logreg

    print("=== KNN skill classifier ===")
    train_knn()
    print("\n=== K-Means playstyle clustering ===")
    train_kmeans()
    print("\n=== Logistic regression win probability ===")
    train_logreg()


def cmd_dashboard(args):
    from dashboard.report_plots import generate_all
    for path in generate_all():
        print(f"wrote {path}")


def cmd_records(args):
    """Display comprehensive career statistics and match log from the terminal."""
    from data.logger import RoundLogger
    logger = RoundLogger()
    summary = logger.get_player_summary(args.tag)

    print(f"\n{'=' * 75}")
    print(f"               [DOSSIER] MAZE MIND -- OPERATIVE CAREER RECORDS")
    print(f"{'=' * 75}")
    print(f"  Operator Tag     : {args.tag}")
    print(f"  Total Operations : {summary['total_rounds']}")
    print(f"  Extractions      : {summary['escapes']} Escaped | {summary['caught']} Caught | {summary['timeouts']} Timeouts")
    print(f"  Escape Win Rate  : {summary['win_rate_pct']:.1f}%")
    best_t = f"{summary['best_time_s']:.1f}s" if summary['best_time_s'] else "N/A"
    print(f"  Best Escape Time : {best_t}")
    print(f"  Peak Efficiency  : {summary['best_efficiency_pct']:.1f}%")
    print(f"{'=' * 75}")
    print(f" {'ID':<6} {'Sector':<8} {'Outcome':<10} {'Time':<8} {'Steps/Opt':<12} {'Efficiency':<12} {'Skill Tier':<12}")
    print(f" " + "-" * 72)
    recent = summary.get("recent_rounds", [])
    if not recent:
        print("  No mission telemetry logged yet. Play your first round to establish records!")
    else:
        for r in recent[:10]:
            r_id = f"#{r['round_id']}"
            sec = f"{r['maze_width']}x{r['maze_height']}"
            out = r.get("outcome", "timeout").upper()
            t = f"{r['time_taken_s']:.1f}s"
            steps = f"{r['player_path_len']}/{r['optimal_path_len']}"
            opt = r.get("optimal_path_len", 0)
            act = max(1, r.get("player_path_len", 1))
            eff = f"{min(100, int(100 * opt / act))}%" if r.get("outcome") == "escaped" else "N/A"
            skill = r.get("predicted_skill") or r.get("skill_label") or "Unrated"
            print(f" {r_id:<6} {sec:<8} {out:<10} {t:<8} {steps:<12} {eff:<12} {skill:<12}")
    print(f"{'=' * 75}\n")


def cmd_label(args):
    """Record an independently reviewed skill label for supervised training."""
    from data.logger import RoundLogger
    RoundLogger().update_skill_label(args.round_id, args.skill)
    print(f"Round {args.round_id} verified as {args.skill}.")


def main():
    parser = argparse.ArgumentParser(description="Maze Mind")
    sub = parser.add_subparsers(dest="command", required=True)

    p_play = sub.add_parser("play", help="Play one round")
    p_play.add_argument("--tag", default="player1")
    p_play.add_argument("--width", type=int, default=14)
    p_play.add_argument("--height", type=int, default=14)
    p_play.add_argument("--time-limit", type=float, default=60.0)
    p_play.add_argument("--difficulty", choices=["Recruit", "Runner", "Master", "Adaptive", "Custom"], default="Recruit")
    p_play.add_argument("--seed", type=int, default=None, help="Reproduce a maze")
    p_play.set_defaults(func=cmd_play)

    p_sim = sub.add_parser("simulate", help="Generate synthetic bot rounds")
    p_sim.add_argument("--rounds-per-profile", type=int, default=120)
    p_sim.set_defaults(func=cmd_simulate)

    p_train = sub.add_parser("train", help="Train all three ML models")
    p_train.set_defaults(func=cmd_train)

    p_dash = sub.add_parser("dashboard", help="Render report figures")
    p_dash.set_defaults(func=cmd_dashboard)

    p_records = sub.add_parser("records", help="View player career records and match history")
    p_records.add_argument("--tag", default="player1", help="Player tag to inspect")
    p_records.set_defaults(func=cmd_records)

    p_label = sub.add_parser("label", help="Set a verified skill label for a reviewed round")
    p_label.add_argument("--round-id", type=int, required=True)
    p_label.add_argument("--skill", choices=["Beginner", "Intermediate", "Expert"], required=True)
    p_label.set_defaults(func=cmd_label)

    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
