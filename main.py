import argparse
from typing import List, Optional

from modules import LogAnalyzer, LogFilterConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log Analyzer CLI")
    parser.add_argument("--log-file", default="app.log", help="Path to the log file (default: app.log)")
    parser.add_argument("--keywords", default="", help="Comma-separated keywords to filter (e.g., error,timeout)")
    parser.add_argument("--start-date", default=None, help="Start date (YYYY-MM-DD), optional")
    parser.add_argument("--end-date", default=None, help="End date (YYYY-MM-DD), optional")
    parser.add_argument("--max-tokens", type=int, default=3000, help="Token budget for output (default: 3000)")
    parser.add_argument("--context-lines", type=int, default=2, help="Context lines around matches (default: 2)")
    parser.add_argument("--deduplicate", action="store_true", help="Enable deduplication of similar entries")
    parser.add_argument("--prioritize-severity", action="store_true", help="Prioritize ERROR/EXCEPTION lines")
    parser.add_argument("--output", default="filtered_logs.txt", help="Output file path (default: filtered_logs.txt)")
    return parser.parse_args()


def main():
    args = parse_args()

    keywords: List[str] = [k.strip() for k in args.keywords.split(",") if k.strip()] if args.keywords else []

    config = LogFilterConfig(
        log_file_path=args.log_file,
        max_tokens=args.max_tokens,
        context_lines=args.context_lines,
        deduplicate=args.deduplicate,
        prioritize_by_severity=args.prioritize_severity,
        prioritize_matches=False,
        max_results=None,
    )

    analyzer = LogAnalyzer(config)
    result = analyzer.analyze(
        keywords=keywords,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result)
    
    print(f"\nFiltered logs saved to: {args.output}")
    print("\nPreview:")
    print(result[:1000] + "..." if len(result) > 1000 else result)


if __name__ == "__main__":
    main()