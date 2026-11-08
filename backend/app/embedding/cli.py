from __future__ import annotations

import argparse
import asyncio
import json
import sys

from app.config import get_settings
from app.embedding.profiles import ProfileDefinition
from app.embedding.service import EmbeddingProfileService
from app.infra.db import session_factory


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Explicit embedding profile lifecycle (never auto-activates)")
    sub = result.add_subparsers(dest="operation", required=True)
    register = sub.add_parser("register", help="Register immutable capabilities from a JSON file, without credentials")
    register.add_argument("--file", required=True)
    for name in ("coverage", "compatibility", "index", "backfill", "validate", "activate"):
        command = sub.add_parser(name)
        command.add_argument("--profile", required=True, help="Registered profile name or UUID")
        if name == "validate":
            command.add_argument("--retrieval-evidence", required=True,
                                 help="Non-sensitive operator note identifying a passing retrieval smoke/eval")
    return result


async def run(arguments: argparse.Namespace) -> int:
    async with session_factory() as session:
        service = EmbeddingProfileService(session, get_settings())
        if arguments.operation == "register":
            with open(arguments.file, encoding="utf-8") as source:
                definition = ProfileDefinition(**json.load(source))
            profile = await service.register(definition)
            await session.commit()
            print(json.dumps({"profile_id": str(profile.id), "profile_name": profile.name,
                              "status": profile.status.value}, ensure_ascii=False))
            return 0
        profile = await service.get(arguments.profile)
        if arguments.operation == "coverage":
            report = await service.coverage(profile)
            print(json.dumps(report.as_dict(), ensure_ascii=False))
            return 0 if report.complete else 2
        if arguments.operation == "compatibility":
            report = await service.compatibility(profile)
            print(json.dumps(report.as_dict(), ensure_ascii=False))
            return 0 if report.compatible else 2
        if arguments.operation == "index":
            await service.ensure_indexes(profile)
            await session.commit()
            print(json.dumps({"profile_id": str(profile.id), "indexes_ready": True}))
            return 0
        if arguments.operation == "backfill":
            report = await service.backfill(profile)
            print(json.dumps(report.as_dict(), ensure_ascii=False))
            return 0 if report.status == "READY" else 2
        if arguments.operation == "validate":
            report = await service.validate(profile, note=arguments.retrieval_evidence)
            await session.commit()
            print(json.dumps({**report.as_dict(), "validation_passed": True}, ensure_ascii=False))
            return 0
        if arguments.operation == "activate":
            profile = await service.activate(profile)
            await session.commit()
            print(json.dumps({"active_profile_id": str(profile.id), "active_profile_name": profile.name}, ensure_ascii=False))
            return 0
        raise ValueError("Unknown embedding operation")


def main(operation: str | None = None) -> None:
    args = parser().parse_args(([operation] if operation else []) + sys.argv[1:])
    try:
        code = asyncio.run(run(args))
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        code = 2
    except Exception as exc:
        # Driver/provider exceptions can include bound private data or URLs.
        print(f"ERROR: {type(exc).__name__}; operation aborted", file=sys.stderr)
        code = 2
    raise SystemExit(code)
