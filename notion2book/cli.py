#!/usr/bin/env python3
"""Unified CLI for the notion2book pipeline."""

import argparse

from . import convert, generate, sync


def build_parser():
    parser = argparse.ArgumentParser(
        prog="notion2book",
        description="Convert OCR output, generate chapter summaries, and sync to Notion.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_convert = sub.add_parser("convert", help="Build outputs/{book}/content.json")
    p_convert.add_argument("--book", help="Book folder name under outputs/")

    p_generate = sub.add_parser(
        "generate",
        help="Generate chapter JSON/Markdown in outputs/{book}/chapters/",
    )
    p_generate.add_argument("--book", help="Book folder name under outputs/")
    p_generate.add_argument(
        "--chapter",
        type=int,
        action="append",
        default=None,
        help="Chapter index (1-based). Can be repeated.",
    )
    p_generate.add_argument(
        "--prompt",
        default=generate.PROMPT_NAME,
        help=(
            "Prompt name from prompts/ (without .md). "
            f"Default: {generate.PROMPT_NAME}"
        ),
    )

    p_sync = sub.add_parser("sync", help="Sync generated chapters to Notion")
    p_sync.add_argument("--book", help="Sync only this book")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "convert":
        return convert.run(book=args.book)

    if args.command == "generate":
        return generate.run(
            book=args.book,
            chapter_indexes=args.chapter,
            prompt=args.prompt,
        )

    if args.command == "sync":
        return sync.run(book=args.book)

    parser.error(f"Unknown command: {args.command}")
    return 2
