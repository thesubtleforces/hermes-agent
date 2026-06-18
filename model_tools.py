#!/usr/bin/env python3
"""
Model Tools Module

Thin orchestration layer over the tool registry. Each tool file in tools/
self-registers its schema, handler, and metadata via tools.registry.register().
This module triggers discovery (by importing all tool modules), then provides
the public API that run_agent.py, cli.py, batch_runner.py, and the RL
environments consume.

Public API (signatures preserved from the original 2,400-line version):
    get_tool_definitions(enabled_toolsets, disabled_toolsets, quiet_mode) -> list
    handle_function_call(function_name, function_args, task_id, user_task) -> str
    TOOL_TO_TOOLSET_MAP: dict          (for batch_runner.py)
    TOOLSET_REQUIREMENTS: dict         (for cli.py, doctor.py)
    get_all_tool_names() -> list
    get_toolset_for_tool(name) -> str
    get_available_toolsets() -> dict
    check_toolset_requirements() -> dict
    check_tool_availability(quiet) -> tuple
"""

import os
import json
import re
import asyncio
import logging
import threading
import time
from typing import Dict, Any, List, Optional, Tuple

from tools.registry import discover_builtin_tools, registry
from toolsets import resolve_toolset, validate_toolset

logger = logging.getLogger(__name__)

def _supervisor_path_allowlist_match(
*,
    function_name: str,
    action: str,
    function_args: Dict[str, Any],
    entries: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the matching supervisor path-allowlist entry's details, else None.

    Fail-closed in every direction: any missing/malformed field, any failed
    check, or any uncertainty yields None (gate blocks). A non-None return is a
    deliberate, fully-validated allowance for ONE write_file under ONE
    time-boxed, byte-capped, prefix-contained allowlist entry.

    Hardened invariants:
      * Containment uses resolve()+relative_to() (component-wise), so '..'
        traversal, symlink escape, and 'prefix-as-substring' siblings
        (/staging/lorenzo-evil vs /staging/lorenzo) are all rejected.
      * expires_at is REQUIRED -- no standing grants.
      * max_bytes_per_file is REQUIRED and enforced -- no silent uncapped writes.
      * A hardcoded forbid floor (.env, config.yaml, AGENTS.md, SOUL.md) is
        denied by basename even if an entry forgets its own forbid_targets.
      * Only write_file is handled today; other tools fail closed until their
        arg-shape gets an explicit handler.
    """
    from datetime import datetime, timezone
    from pathlib import Path

    ALWAYS_FORBID = {".env", "config.yaml", "config.yml", "AGENTS.md", "SOUL.md"}

    if not isinstance(entries, list):
        return None
    if not isinstance(function_args, dict):
        return None
    if function_name != "write_file":
        return None
    if action != "edit_handler":
        return None

    raw_path = function_args.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None

    content = function_args.get("content", "")
    if isinstance(content, bytes):
        payload_bytes = len(content)
    elif isinstance(content, str):
        payload_bytes = len(content.encode("utf-8"))
    else:
        payload_bytes = len(str(content).encode("utf-8"))

    target_path = Path(raw_path).expanduser().resolve(strict=False)
    target_name = target_path.name
    now = datetime.now(timezone.utc)

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("enabled", False) is not True:
            continue
        if function_name not in set(entry.get("tools") or []):
            continue
        if action not in set(entry.get("actions") or []):
            continue

        expires_at = entry.get("expires_at")
        if not expires_at:
            continue
        try:
            expires = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
        except ValueError:
            continue
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires <= now:
            continue

        max_bytes = entry.get("max_bytes_per_file")
        try:
            max_bytes = int(max_bytes)
        except (TypeError, ValueError):
            continue
        if max_bytes <= 0:
            continue
        if payload_bytes > max_bytes:
            continue

        entry_forbid = entry.get("forbid_targets") or []
        if not isinstance(entry_forbid, list):
            entry_forbid = []
        forbid = ALWAYS_FORBID | {str(x) for x in entry_forbid}
        if target_name in forbid:
            continue

        prefix = entry.get("path_prefix")
        if not isinstance(prefix, str) or not prefix:
            continue
        prefix_path = Path(prefix).expanduser().resolve(strict=False)
        try:
            rel = target_path.relative_to(prefix_path)
        except ValueError:
            continue
        if rel == Path("."):
            continue

        return {
            "initiative_id": entry.get("initiative_id"),
            "target_paths": [raw_path],
            "resolved_path": str(target_path),
            "path_prefix": str(prefix_path),
            "max_bytes_per_file": max_bytes,
            "payload_bytes": payload_bytes,
            "reason": entry.get("reason", ""),
        }

    return None



# =============================================================================
# Async Bridging  (single source of truth -- used by registry.dispatch too)
# =============================================================================

_tool_loop = None          # persistent loop for the main (CLI) thread
_tool_loop_lock = threading.Lock()
_worker_thread_local = threading.local()  # per-worker-thread persistent loops


def _get_tool_loop():
    """Return a long-lived event loop for running async tool handlers.

    Using a persistent loop (instead of asyncio.run() which creates and
    *closes* a fresh loop every time) prevents "Event loop is closed"
    errors that occur when cached httpx/AsyncOpenAI clients attempt to
    close their transport on a dead loop during garbage collection.
    """
    global _tool_loop
    with _tool_loop_lock:
        if _tool_loop is None or _tool_loop.is_closed():
            _tool_loop = asyncio.new_event_loop()
        return _tool_loop


def _get_worker_loop():
    """Return a persistent event loop for the current worker thread.

    Each worker thread (e.g., delegate_task's ThreadPoolExecutor threads)
    gets its own long-lived loop stored in thread-local storage.  This
    prevents the "Event loop is closed" errors that occurred when
    asyncio.run() was used per-call: asyncio.run() creates a loop, runs
    the coroutine, then *closes* the loop — but cached httpx/AsyncOpenAI
    clients remain bound to that now-dead loop and raise RuntimeError
    during garbage collection or subsequent use.

    By keeping the loop alive for the thread's lifetime, cached clients
    stay valid and their cleanup runs on a live loop.
    """
    loop = getattr(_worker_thread_local, 'loop', None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _worker_thread_local.loop = loop
    return loop


def _run_async(coro):
    """Run an async coroutine from a sync context.

    If the current thread already has a running event loop (e.g., inside
    the gateway's async stack or Atropos's event loop), we spin up a
    disposable thread so asyncio.run() can create its own loop without
    conflicting.

    For the common CLI path (no running loop), we use a persistent event
    loop so that cached async clients (httpx / AsyncOpenAI) remain bound
    to a live loop and don't trigger "Event loop is closed" on GC.

    When called from a worker thread (parallel tool execution), we use a
    per-thread persistent loop to avoid both contention with the main
    thread's shared loop AND the "Event loop is closed" errors caused by
    asyncio.run()'s create-and-destroy lifecycle.

    This is the single source of truth for sync->async bridging in tool
    handlers. Each handler is self-protecting via this function.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Inside an async context (gateway, RL env) — run in a fresh thread
        # with its own event loop we own a reference to, so on timeout we
        # can cancel the task inside that loop (ThreadPoolExecutor.cancel()
        # only works on not-yet-started futures — it's a no-op on a running
        # worker, which previously leaked the thread on every 300 s timeout).
        import concurrent.futures

        worker_loop: Optional[asyncio.AbstractEventLoop] = None
        loop_ready = threading.Event()

        def _run_in_worker():
            nonlocal worker_loop
            worker_loop = asyncio.new_event_loop()
            loop_ready.set()
            try:
                asyncio.set_event_loop(worker_loop)
                return worker_loop.run_until_complete(coro)
            finally:
                try:
                    # Cancel anything still pending (e.g. task cancelled
                    # externally via call_soon_threadsafe on timeout).
                    pending = asyncio.all_tasks(worker_loop)
                    for t in pending:
                        t.cancel()
                    if pending:
                        worker_loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    pass
                worker_loop.close()

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(_run_in_worker)
        try:
            return future.result(timeout=300)
        except concurrent.futures.TimeoutError:
            # Cancel the coroutine inside its own loop so the worker thread
            # can wind down instead of running forever.
            if loop_ready.wait(timeout=1.0) and worker_loop is not None:
                try:
                    for t in asyncio.all_tasks(worker_loop):
                        worker_loop.call_soon_threadsafe(t.cancel)
                except RuntimeError:
                    # Loop already closed — nothing to cancel.
                    pass
            raise
        finally:
            # wait=False: don't block the caller on a stuck coroutine. We've
            # already requested cancellation above; the worker will exit
            # once the coroutine observes it (usually at the next await).
            pool.shutdown(wait=False)

    # If we're on a worker thread (e.g., parallel tool execution in
    # delegate_task), use a per-thread persistent loop.  This avoids
    # contention with the main thread's shared loop while keeping cached
    # httpx/AsyncOpenAI clients bound to a live loop for the thread's
    # lifetime — preventing "Event loop is closed" on GC cleanup.
    if threading.current_thread() is not threading.main_thread():
        worker_loop = _get_worker_loop()
        return worker_loop.run_until_complete(coro)

    tool_loop = _get_tool_loop()
    return tool_loop.run_until_complete(coro)


# =============================================================================
# Tool Discovery  (importing each module triggers its registry.register calls)
# =============================================================================

discover_builtin_tools()

# MCP tool discovery (external MCP servers from config) used to run here as
# a module-level side effect.  It was removed because discover_mcp_tools()
# internally uses a blocking future.result(timeout=120) wait, and the
# gateway lazy-imports this module from inside the asyncio event loop on
# the first user message — freezing Discord/Telegram heartbeats for up to
# 120s whenever any configured MCP server was slow or unreachable (#16856).
#
# Each entry point now runs discovery explicitly at its own startup:
#   - gateway/run.py            -> start_gateway() uses run_in_executor
#   - cli.py, hermes_cli/*      -> inline on startup (no event loop)
#   - tui_gateway/server.py     -> inline on startup (no event loop)
#   - acp_adapter/server.py     -> asyncio.to_thread on session init

# Plugin tool discovery (user/project/pip plugins)
try:
    from hermes_cli.plugins import discover_plugins
    discover_plugins()
except Exception as e:
    logger.debug("Plugin discovery failed: %s", e)


# =============================================================================
# Backward-compat constants  (built once after discovery)
# =============================================================================

TOOL_TO_TOOLSET_MAP: Dict[str, str] = registry.get_tool_to_toolset_map()

TOOLSET_REQUIREMENTS: Dict[str, dict] = registry.get_toolset_requirements()

# Resolved tool names from the last get_tool_definitions() call.
# Used by code_execution_tool to know which tools are available in this session.
_last_resolved_tool_names: List[str] = []


# =============================================================================
# Legacy toolset name mapping  (old _tools-suffixed names -> tool name lists)
# =============================================================================

_LEGACY_TOOLSET_MAP = {
    "web_tools": ["web_search", "web_extract"],
    "terminal_tools": ["terminal"],
    "vision_tools": ["vision_analyze"],
    "moa_tools": ["mixture_of_agents"],
    "image_tools": ["image_generate"],
    "skills_tools": ["skills_list", "skill_view", "skill_manage"],
    "browser_tools": [
        "browser_navigate", "browser_snapshot", "browser_click",
        "browser_type", "browser_scroll", "browser_back",
        "browser_press", "browser_get_images",
        "browser_vision", "browser_console"
    ],
    "cronjob_tools": ["cronjob"],
    "file_tools": ["read_file", "write_file", "patch", "search_files"],
    "tts_tools": ["text_to_speech"],
}


# =============================================================================
# get_tool_definitions  (the main schema provider)
# =============================================================================

# Module-level memoization for get_tool_definitions(). Keyed on
# (frozenset(enabled_toolsets), frozenset(disabled_toolsets), registry._generation).
# Hot callers (gateway runner, AIAgent.__init__) invoke this on every turn
# with quiet_mode=True; caching avoids ~7 ms of registry walking + schema
# filtering + check_fn probing per call. Only active when quiet_mode=True
# because quiet_mode=False has stdout side effects (tool-selection prints).
#
# Invalidation happens transparently via the registry's _generation counter,
# which bumps on register() / deregister() / register_toolset_alias(). The
# inner check_fn TTL cache in registry.py handles environment drift (Docker
# daemon start/stop, env var changes, etc.) on a 30 s horizon.
_tool_defs_cache: Dict[tuple, List[Dict[str, Any]]] = {}

# Hard cap on memoized get_tool_definitions() results. A long-lived Gateway
# process sees many distinct toolset/config fingerprints over its lifetime
# (per-session toolset sets, config edits, kanban-task toggles); without a
# bound the cache grows unboundedly. 8 comfortably covers the warm working
# set (the handful of distinct platform/toolset combos a gateway actually
# serves) while keeping the cap small. (#19251)
_TOOL_DEFS_CACHE_MAX = 8


def _clear_tool_defs_cache() -> None:
    """Drop memoized get_tool_definitions() results. Called when dynamic
    schema dependencies change (e.g. discord capability cache reset,
    execute_code sandbox reconfigured)."""
    _tool_defs_cache.clear()


def get_tool_definitions(
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
    quiet_mode: bool = False,
    skip_tool_search_assembly: bool = False,
) -> List[Dict[str, Any]]:
    """
    Get tool definitions for model API calls with toolset-based filtering.

    All tools must be part of a toolset to be accessible.

    Args:
        enabled_toolsets: Only include tools from these toolsets.
        disabled_toolsets: Exclude tools from these toolsets (if enabled_toolsets is None).
        quiet_mode: Suppress status prints.
        skip_tool_search_assembly: When True, return the pre-assembly tool list
            (raw schemas for every enabled tool). Used internally by the
            tool_search / tool_describe bridge handlers so they can read the
            real catalog, not the already-collapsed one. Public callers should
            leave this False.

    Returns:
        Filtered list of OpenAI-format tool definitions.
    """
    # Fast path: memoized result when the caller doesn't need stdout prints.
    # The cache key captures every argument-level input; the registry
    # generation captures registry mutations (MCP refresh, plugin load).
    # check_fn results are TTL-cached one level down, inside
    # registry.get_definitions. The config-mtime fingerprint below captures
    # user-visible config edits that affect dynamic schemas (execute_code
    # mode, discord action allowlist, etc.) without needing an explicit
    # invalidate hook on every config-writer.
    if quiet_mode:
        try:
            from hermes_cli.config import get_config_path
            cfg_path = get_config_path()
            cfg_stat = cfg_path.stat()
            cfg_fp = (cfg_stat.st_mtime_ns, cfg_stat.st_size)
        except (FileNotFoundError, OSError, ImportError):
            cfg_fp = None
        cache_key = (
            frozenset(enabled_toolsets) if enabled_toolsets is not None else None,
            frozenset(disabled_toolsets) if disabled_toolsets else None,
            registry._generation,
            cfg_fp,
            bool(os.environ.get("HERMES_KANBAN_TASK")),
            bool(skip_tool_search_assembly),
        )
        cached = _tool_defs_cache.get(cache_key)
        if cached is not None:
            # Update _last_resolved_tool_names so downstream callers see
            # consistent state even on a cache hit.
            global _last_resolved_tool_names
            _last_resolved_tool_names = [t["function"]["name"] for t in cached]
            # Return a shallow copy of the list but share the dict references —
            # schemas are treated as read-only by all known callers.
            return list(cached)

    result = _compute_tool_definitions(enabled_toolsets, disabled_toolsets, quiet_mode,
                                       skip_tool_search_assembly=skip_tool_search_assembly)
    if quiet_mode:
        # Cache the freshly-computed list, but hand callers a shallow copy so
        # downstream mutations (e.g. run_agent appending memory/LCM tool
        # schemas to self.tools) don't poison the cache. Without this, a
        # long-lived Gateway process accumulates duplicate tool names across
        # agent inits and providers that enforce unique tool names
        # (DeepSeek, Xiaomi MiMo, Moonshot Kimi) reject the request with
        # HTTP 400. Mirrors the cache-hit path above. (issue #17335)
        # Bound the cache with LRU eviction so a long-lived Gateway process
        # doesn't accumulate entries unboundedly across the many distinct
        # toolset/config fingerprints it sees over its lifetime (#19251).
        if len(_tool_defs_cache) >= _TOOL_DEFS_CACHE_MAX:
            _tool_defs_cache.pop(next(iter(_tool_defs_cache)))  # evict oldest
        _tool_defs_cache[cache_key] = result
        return list(result)
    return result


def _compute_tool_definitions(
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
    quiet_mode: bool = False,
    skip_tool_search_assembly: bool = False,
) -> List[Dict[str, Any]]:
    """Uncached implementation of :func:`get_tool_definitions`."""
    # Determine which tool names the caller wants
    tools_to_include: set = set()

    if enabled_toolsets is not None:
        effective_enabled_toolsets = list(enabled_toolsets)
        if os.environ.get("HERMES_KANBAN_TASK") and "kanban" not in effective_enabled_toolsets:
            # Dispatcher-spawned workers are scoped by HERMES_KANBAN_TASK and
            # must always receive the lifecycle handoff tools. Assignee
            # profiles may intentionally restrict their normal chat toolsets
            # (for token/cost reasons), but that should not strip the kanban
            # worker's completion/block/heartbeat surface.
            effective_enabled_toolsets.append("kanban")
        for toolset_name in effective_enabled_toolsets:
            if validate_toolset(toolset_name):
                resolved = resolve_toolset(toolset_name)
                tools_to_include.update(resolved)
                if not quiet_mode:
                    print(f"✅ Enabled toolset '{toolset_name}': {', '.join(resolved) if resolved else 'no tools'}")
            elif toolset_name in _LEGACY_TOOLSET_MAP:
                legacy_tools = _LEGACY_TOOLSET_MAP[toolset_name]
                tools_to_include.update(legacy_tools)
                if not quiet_mode:
                    print(f"✅ Enabled legacy toolset '{toolset_name}': {', '.join(legacy_tools)}")
            elif not quiet_mode:
                print(f"⚠️  Unknown toolset: {toolset_name}")
    else:
        # Default: start with everything
        from toolsets import get_all_toolsets
        for ts_name in get_all_toolsets():
            tools_to_include.update(resolve_toolset(ts_name))

    # Always apply disabled toolsets as a subtraction step at the end.
    # This ensures that even if a composite toolset (like hermes-cli)
    # is enabled, any tools belonging to a disabled toolset are strictly
    # stripped out. See issue #17309.
    if disabled_toolsets:
        for toolset_name in disabled_toolsets:
            if validate_toolset(toolset_name):
                resolved = resolve_toolset(toolset_name)
                tools_to_include.difference_update(resolved)
                if not quiet_mode:
                    print(f"🚫 Disabled toolset '{toolset_name}': {', '.join(resolved) if resolved else 'no tools'}")
            elif toolset_name in _LEGACY_TOOLSET_MAP:
                legacy_tools = _LEGACY_TOOLSET_MAP[toolset_name]
                tools_to_include.difference_update(legacy_tools)
                if not quiet_mode:
                    print(f"🚫 Disabled legacy toolset '{toolset_name}': {', '.join(legacy_tools)}")
            elif not quiet_mode:
                print(f"⚠️  Unknown toolset: {toolset_name}")

    # Plugin-registered tools are now resolved through the normal toolset
    # path — validate_toolset() / resolve_toolset() / get_all_toolsets()
    # all check the tool registry for plugin-provided toolsets.  No bypass
    # needed; plugins respect enabled_toolsets / disabled_toolsets like any
    # other toolset.

    # Ask the registry for schemas (only returns tools whose check_fn passes)
    filtered_tools = registry.get_definitions(tools_to_include, quiet=quiet_mode)

    # The set of tool names that actually passed check_fn filtering.
    # Use this (not tools_to_include) for any downstream schema that references
    # other tools by name — otherwise the model sees tools mentioned in
    # descriptions that don't actually exist, and hallucinates calls to them.
    available_tool_names = {t["function"]["name"] for t in filtered_tools}

    # Rebuild execute_code schema to only list sandbox tools that are actually
    # available.  Without this, the model sees "web_search is available in
    # execute_code" even when the API key isn't configured or the toolset is
    # disabled (#560-discord).
    if "execute_code" in available_tool_names:
        from tools.code_execution_tool import SANDBOX_ALLOWED_TOOLS, build_execute_code_schema, _get_execution_mode
        sandbox_enabled = SANDBOX_ALLOWED_TOOLS & available_tool_names
        dynamic_schema = build_execute_code_schema(sandbox_enabled, mode=_get_execution_mode())
        for i, td in enumerate(filtered_tools):
            if td.get("function", {}).get("name") == "execute_code":
                filtered_tools[i] = {"type": "function", "function": dynamic_schema}
                break

    # Rebuild discord / discord_admin schemas based on the bot's privileged
    # intents (detected from GET /applications/@me) and the user's action
    # allowlist in config.  Hides actions the bot's intents don't support so
    # the model never attempts them, and annotates fetch_messages when the
    # MESSAGE_CONTENT intent is missing.
    _discord_schema_fns = {
        "discord": "get_dynamic_schema_core",
        "discord_admin": "get_dynamic_schema_admin",
    }
    for discord_tool_name in _discord_schema_fns:
        if discord_tool_name in available_tool_names:
            try:
                from tools import discord_tool as _dt
                schema_fn = getattr(_dt, _discord_schema_fns[discord_tool_name])
                dynamic = schema_fn()
            except Exception:
                dynamic = None
            if dynamic is None:
                filtered_tools = [
                    t for t in filtered_tools
                    if t.get("function", {}).get("name") != discord_tool_name
                ]
                available_tool_names.discard(discord_tool_name)
            else:
                for i, td in enumerate(filtered_tools):
                    if td.get("function", {}).get("name") == discord_tool_name:
                        filtered_tools[i] = {"type": "function", "function": dynamic}
                        break

    # Strip web tool cross-references from browser_navigate description when
    # web_search / web_extract are not available.  The static schema says
    # "prefer web_search or web_extract" which causes the model to hallucinate
    # those tools when they're missing.
    if "browser_navigate" in available_tool_names:
        web_tools_available = {"web_search", "web_extract"} & available_tool_names
        if not web_tools_available:
            for i, td in enumerate(filtered_tools):
                if td.get("function", {}).get("name") == "browser_navigate":
                    desc = td["function"].get("description", "")
                    desc = desc.replace(
                        " For simple information retrieval, prefer web_search or web_extract (faster, cheaper).",
                        "",
                    )
                    filtered_tools[i] = {
                        "type": "function",
                        "function": {**td["function"], "description": desc},
                    }
                    break

    if not quiet_mode:
        if filtered_tools:
            tool_names = [t["function"]["name"] for t in filtered_tools]
            print(f"🛠️  Final tool selection ({len(filtered_tools)} tools): {', '.join(tool_names)}")
        else:
            print("🛠️  No tools selected (all filtered out or unavailable)")

    global _last_resolved_tool_names
    _last_resolved_tool_names = [t["function"]["name"] for t in filtered_tools]

    # Sanitize schemas for broad backend compatibility. llama.cpp's
    # json-schema-to-grammar converter (used by its OAI server to build
    # GBNF tool-call parsers) rejects some shapes that cloud providers
    # silently accept — bare "type": "object" with no properties,
    # string-valued schema nodes from malformed MCP servers, etc. This
    # is a no-op for schemas that are already well-formed.
    try:
        from tools.schema_sanitizer import sanitize_tool_schemas
        filtered_tools = sanitize_tool_schemas(filtered_tools)
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("Schema sanitization skipped: %s", e)

    # ── Tool Search (progressive disclosure) ────────────────────────────
    # Conditionally replace MCP + plugin (non-core) tools with three bridge
    # tools (tool_search / tool_describe / tool_call) when the deferrable
    # surface exceeds the configured threshold (default 10% of context
    # window). Core Hermes tools (toolsets._HERMES_CORE_TOOLS) are NEVER
    # deferred. See tools/tool_search.py for full design notes.
    #
    # This is deliberately the last step before returning — sanitization
    # has already normalized schemas, and the assembly is idempotent in
    # case some caller invokes get_tool_definitions twice.
    try:
        from tools.tool_search import assemble_tool_defs, load_config as _load_ts_config
        ts_cfg = _load_ts_config()
        if not skip_tool_search_assembly and ts_cfg.enabled != "off":
            context_length = _resolve_active_context_length()
            assembly = assemble_tool_defs(
                filtered_tools,
                context_length=context_length,
                config=ts_cfg,
            )
            if assembly.activated and not quiet_mode:
                print(
                    f"🔎 Tool Search: {assembly.deferred_count} MCP/plugin tools deferred "
                    f"(~{assembly.deferred_tokens} tokens) behind tool_search/describe/call. "
                    f"Threshold ~{assembly.threshold_tokens} tokens."
                )
            filtered_tools = assembly.tool_defs
    except Exception as e:  # pragma: no cover — never break tool loading
        logger.warning("Tool search assembly skipped: %s", e)

    return filtered_tools


def _resolve_active_context_length() -> int:
    """Look up the active model's context length for the tool-search gate.

    Returns 0 when the model can't be resolved — ``should_activate`` falls
    back to a fixed token cutoff in that case.
    """
    try:
        from hermes_cli.config import load_config as _load
        cfg = _load() or {}
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        if not isinstance(model_cfg, dict):
            model_cfg = {}
        model_id = (model_cfg.get("model") or model_cfg.get("default") or "").strip()
        if not model_id:
            return 0
        from agent.model_metadata import get_model_context_length
        return int(get_model_context_length(model_id) or 0)
    except Exception as e:
        logger.debug("Could not resolve active context length: %s", e)
        return 0


# =============================================================================
# handle_function_call  (the main dispatcher)
# =============================================================================

# Tools whose execution is intercepted by the agent loop (run_agent.py)
# because they need agent-level state (TodoStore, MemoryStore, etc.).
# The registry still holds their schemas; dispatch just returns a stub error
# so if something slips through, the LLM sees a sensible message.
_AGENT_LOOP_TOOLS = {"todo", "memory", "session_search", "delegate_task"}
_READ_SEARCH_TOOLS = {"read_file", "search_files"}

# Supervisor gate is wired but disabled by default.  When enabled, only tools
# mapped to MEDIUM/HIGH risk are sent to the external supervisor; clearly-safe
# read/concierge actions stay on the fast path.
_SUPERVISOR_GATE_ENV = "SUPERVISOR_GATE_ENABLED"
_SUPERVISOR_IMPORT_PATH_ENV = "SUPERVISOR_IMPORT_PATH"
_SUPERVISOR_PATH_ALLOWLIST_ENV = "SUPERVISOR_PATH_ALLOWLIST"
_DEFAULT_SUPERVISOR_IMPORT_PATHS = (
    "/home/transversed/hermes_supervisor",
    "/home/transversed/hermes_supervisor/hermes_supervisor",
)

# Hermes tool name -> supervisor action key.  The action keys are deliberately
# the existing supervisor.RISK_TABLE vocabulary; risk_for(action) determines
# whether the gate actually runs.
_SUPERVISOR_TOOL_ACTION_MAP = {
    "terminal": "run_command",              # arbitrary shell command execution
    "execute_code": "run_command",          # arbitrary Python can spawn/mutate directly
    "write_file": "edit_handler",
    "patch": "edit_handler",
    "skill_manage": "wire_tool",
    "cronjob": "wire_tool",                 # durable autonomous behavior
    "mcp_higgsfield_deploy_game": "deploy",
    "mcp_higgsfield_publish_game": "deploy",
    "mcp_higgsfield_confirm_billing_purchase": "external_send",
    "mcp_era_billing__confirm_subscription_change": "external_send",
    "mcp_era_billing__upgrade": "external_send",
    "mcp_era_billing__cancel_subscription": "external_send",
    "mcp_era_billing__uncancel_subscription": "external_send",
    "mcp_era_connections__disconnect_institution": "delete",
    "mcp_era_accounts__manage_account": "edit_config",
    "mcp_era_accounts__set_account_visibility": "edit_config",
    "mcp_era_transactions__update_transactions": "edit_config",
    "mcp_era_transactions__manage_manual_transaction": "edit_config",
    "mcp_era_transactions__import_csv_transactions": "migrate_data",
    "mcp_era_transactions__manage_automation_rules": "wire_tool",
    "mcp_era_transactions__manage_categories": "edit_config",
    "mcp_era_transactions__manage_transaction_tags": "edit_config",
    "mcp_era_transactions__manage_transfer_links": "edit_config",
    "mcp_era_knowledge__remember": "edit_config",
    "mcp_era_knowledge__forget": "delete",
    "mcp_era_knowledge__reset_pack_questions": "edit_config",
}

# Explicit fast-path safe tools.  This includes read-only tools, normal
# messaging/clarification surfaces, and concierge/research surfaces whose normal
# use is informational rather than mutating local files, config, money, or infra.
_SUPERVISOR_SAFE_TOOL_NAMES = {
    "read_file", "search_files", "web_search", "web_extract",
    "browser_snapshot", "browser_get_images", "browser_console",
    "browser_vision", "browser_navigate", "browser_back", "browser_scroll",
    "session_search", "todo", "clarify", "send_message", "text_to_speech",
    "vision_analyze", "image_generate", "skills_list", "skill_view",
    "mcp_era_accounts__list_financial_accounts",
    "mcp_era_accounts__check_account_balance",
    "mcp_era_accounts__manage_account_groups",
    "mcp_era_insights__analyze_spending",
    "mcp_era_insights__compare_spending_periods",
    "mcp_era_insights__forecast_spending",
    "mcp_era_insights__get_cash_flow",
    "mcp_era_insights__get_daily_financial_summary",
    "mcp_era_transactions__list_transactions",
    "mcp_era_transactions__search_transactions",
    "mcp_era_transactions__list_spending_categories",
    "mcp_era_transactions__list_recurring_charges",
    "mcp_era_billing__get_current_plan",
    "mcp_era_billing__list_plans",
    "mcp_era_billing__manage",
    "mcp_era_billing__preview_subscription_change",
    "mcp_era_help__get_help",
    "mcp_era_knowledge__get_financial_context_and_overview",
    "mcp_era_knowledge__get_pending_questions",
    "mcp_era_knowledge__recall_history",
    "mcp_era_referral__get_referral_link",
    "mcp_era_referral__get_referral_stats",
    "mcp_higgsfield_balance", "mcp_higgsfield_transactions",
    "mcp_higgsfield_models_explore", "mcp_higgsfield_job_status",
    "mcp_higgsfield_job_display", "mcp_higgsfield_show_generations",
    "mcp_higgsfield_show_medias", "mcp_higgsfield_animation_actions",
    "mcp_higgsfield_list_workspaces", "mcp_higgsfield_show_plans_and_credits",
    "mcp_higgsfield_personal_clipper_jobs", "mcp_higgsfield_personal_clipper_status",
    "mcp_higgsfield_video_analysis_jobs", "mcp_higgsfield_video_analysis_status",
    "mcp_higgsfield_presets_show", "mcp_higgsfield_show_marketing_studio",
    "mcp_higgsfield_show_reference_elements", "mcp_higgsfield_show_characters",
}

_SUPERVISOR_SAFE_PREFIXES = (
    "mcp_era_insights__",
    "mcp_era_help__",
    "mcp_era_list_",
    "mcp_era_read_",
    "mcp_higgsfield_list_",
    "mcp_higgsfield_read_",
)

_SUPERVISOR_HIGH_RISK_NAME_FRAGMENTS = (
    "deploy", "publish", "confirm_billing", "upgrade", "cancel_subscription",
    "disconnect", "delete", "remove", "import_csv", "manage_", "update_",
    "set_account", "trigger_connection_resync", "join_referral_program",
    "switch_referral_campaign", "generate_", "motion_control", "outpaint",
    "reframe", "upscale", "remove_background", "reveal_generation",
    "select_workspace", "sync_agents", "virality_predictor",
)


def _supervisor_gate_enabled() -> bool:
    value = os.getenv(_SUPERVISOR_GATE_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _supervisor_load_path_allowlist() -> List[Dict[str, Any]]:
    """Load optional path-scoped supervisor allowlist entries from JSON env."""
    raw = os.getenv(_SUPERVISOR_PATH_ALLOWLIST_ENV, "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _supervisor_append_decision(record: Dict[str, Any]) -> None:
    """Best-effort JSONL audit for wrapper-level fail-closed outcomes."""
    try:
        from datetime import datetime, timezone
        from pathlib import Path

        default_path = Path.home() / "hermes_supervisor" / "gate_decisions.log"
        path = Path(os.getenv("SUPERVISOR_GATE_DECISIONS_LOG", str(default_path))).expanduser()
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "model_tools.handle_function_call",
            **record,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as exc:
        logger.warning("failed to append supervisor wrapper decision log: %s", exc)


def _supervisor_block_result(
    *,
    function_name: str,
    action: str,
    risk: str,
    reason: str,
    verdict: str = "BLOCK",
) -> str:
    return json.dumps({
        "error": (
            f"BLOCKED by supervisor gate ({reason}). Tool {function_name!r} "
            "was not executed."
        ),
        "status": "blocked",
        "supervisor_gate": {
            "enabled": True,
            "tool": function_name,
            "action": action,
            "risk": risk,
            "verdict": verdict,
            "reason": reason,
        },
    }, ensure_ascii=False)


def _supervisor_import_modules():
    """Import supervisor/gate modules lazily so disabled mode has no dependency."""
    import sys

    raw_path = os.getenv(_SUPERVISOR_IMPORT_PATH_ENV, "")
    for part in raw_path.split(os.pathsep):
        if part and part not in sys.path:
            sys.path.insert(0, part)
    for part in _DEFAULT_SUPERVISOR_IMPORT_PATHS:
        if part and part not in sys.path:
            sys.path.insert(0, part)

    try:
        from hermes_supervisor import supervisor as supervisor_mod
        from hermes_supervisor import gate as gate_mod
    except Exception:
        import supervisor as supervisor_mod  # type: ignore
        import gate as gate_mod  # type: ignore
    return supervisor_mod, gate_mod


def _supervisor_action_for_tool(function_name: str, supervisor_mod=None) -> tuple[str, Any] | tuple[None, None]:
    """Return (supervisor_action_key, risk) for tools that should be gated."""
    action = _SUPERVISOR_TOOL_ACTION_MAP.get(function_name)
    if action is None and function_name in _SUPERVISOR_SAFE_TOOL_NAMES:
        return None, None
    if action is None and function_name.startswith(_SUPERVISOR_SAFE_PREFIXES):
        return None, None
    if action is None and function_name.startswith("mcp_") and any(
        fragment in function_name for fragment in _SUPERVISOR_HIGH_RISK_NAME_FRAGMENTS
    ):
        action = "external_send"
    if action is None:
        return None, None

    if supervisor_mod is None:
        supervisor_mod, _ = _supervisor_import_modules()
    risk = supervisor_mod.risk_for(action)
    return action, risk


def _maybe_apply_supervisor_gate(
    *,
    function_name: str,
    function_args: Dict[str, Any],
    user_task: Optional[str],
    task_id: Optional[str],
    session_id: Optional[str],
    tool_call_id: Optional[str],
    turn_id: Optional[str],
    api_request_id: Optional[str],
) -> Optional[str]:
    """Return a JSON block result when the supervisor gate blocks, else None."""
    if not _supervisor_gate_enabled():
        return None

    action = "unknown"
    risk_value = "unknown"
    try:
        action = _SUPERVISOR_TOOL_ACTION_MAP.get(function_name)
        if action is None and function_name in _SUPERVISOR_SAFE_TOOL_NAMES:
            return None
        if action is None and function_name.startswith(_SUPERVISOR_SAFE_PREFIXES):
            return None
        if action is None and function_name.startswith("mcp_") and any(
            fragment in function_name for fragment in _SUPERVISOR_HIGH_RISK_NAME_FRAGMENTS
        ):
            action = "external_send"
        if action is None:
            return None

        # From here onward, the tool is in a potentially risky bucket.  Import
        # failures are fail-closed for that bucket, but clearly-safe/unmapped
        # reads and concierge tools above never pay this import/network cost.
        supervisor_mod, gate_mod = _supervisor_import_modules()
        risk = supervisor_mod.risk_for(action)
        risk_value = getattr(risk, "value", str(risk))
        if risk not in {supervisor_mod.Risk.MEDIUM, supervisor_mod.Risk.HIGH}:
            return None

        allowlist_match = _supervisor_path_allowlist_match(
            function_name=function_name,
            action=action,
            function_args=function_args,
            entries=_supervisor_load_path_allowlist(),
        )
        if allowlist_match is not None:
            _supervisor_append_decision({
                "action": action,
                "risk": risk_value,
                "tool": function_name,
                "verdict": "APPROVE",
                "blocking": False,
                "reasoning": "approved by supervisor path allowlist",
                "issues": [],
                "wrapper_enforced": True,
                "allowlist_match": allowlist_match,
            })
            return None

        requirement = (
            user_task.strip()
            if isinstance(user_task, str) and user_task.strip()
            else f"Hermes-initiated {function_name} action; no explicit user task provided."
        )
        proposed = json.dumps({
            "tool": function_name,
            "supervisor_action": action,
            "risk": risk_value,
            "arguments": function_args,
        }, ensure_ascii=False, default=str, indent=2)
        # Normal command/content payloads stay fully legible.  The cap is only
        # a last-resort context guard for unusually large write_file/patch args.
        if len(proposed) > 12000:
            proposed = proposed[:12000] + "... [truncated]"
        context = json.dumps({
            "task_id": task_id or "",
            "session_id": session_id or "",
            "tool_call_id": tool_call_id or "",
            "turn_id": turn_id or "",
            "api_request_id": api_request_id or "",
            "note": (
                "The supervisor is reviewing the exact Hermes tool call before "
                "dispatch. The proposed JSON contains the full command/content "
                "unless explicitly marked truncated."
            ),
        }, ensure_ascii=False, indent=2)

        try:
            review = gate_mod.gate(
                action=action,
                requirement=requirement,
                proposed=proposed,
                context=context,
                execute=None,
                risk_override=risk,
            )
        except gate_mod.GateBlocked as blocked:
            review = blocked.review
            verdict = getattr(getattr(review, "verdict", None), "value", getattr(review, "verdict", ""))
            reason = f"supervisor returned blocking {verdict or 'ambiguous verdict'}"
            _supervisor_append_decision({
                "action": action,
                "risk": risk_value,
                "tool": function_name,
                "verdict": verdict or "AMBIGUOUS",
                "blocking": True,
                "reasoning": getattr(review, "reasoning", reason),
                "issues": list(getattr(review, "issues", []) or []),
                "wrapper_enforced": True,
            })
            return _supervisor_block_result(
                function_name=function_name,
                action=action,
                risk=risk_value,
                verdict=verdict or "AMBIGUOUS",
                reason=reason,
            )

        verdict = getattr(getattr(review, "verdict", None), "value", getattr(review, "verdict", ""))
        blocking = bool(getattr(review, "blocking", True))
        if verdict == "APPROVE" and not blocking:
            return None

        reason = f"supervisor returned {verdict or 'ambiguous verdict'}"
        _supervisor_append_decision({
            "action": action,
            "risk": risk_value,
            "tool": function_name,
            "verdict": verdict or "AMBIGUOUS",
            "blocking": True,
            "reasoning": getattr(review, "reasoning", reason),
            "issues": list(getattr(review, "issues", []) or []),
            "wrapper_enforced": True,
        })
        return _supervisor_block_result(
            function_name=function_name,
            action=action,
            risk=risk_value,
            verdict=verdict or "AMBIGUOUS",
            reason=reason,
        )
    except Exception as exc:
        reason = f"gate error: {type(exc).__name__}: {exc}"
        _supervisor_append_decision({
            "action": action,
            "risk": risk_value,
            "tool": function_name,
            "verdict": "ERROR",
            "blocking": True,
            "reasoning": reason,
            "issues": [reason],
            "wrapper_enforced": True,
        })
        return _supervisor_block_result(
            function_name=function_name,
            action=action,
            risk=risk_value,
            verdict="ERROR",
            reason=reason,
        )


# =========================================================================
# Tool error sanitization
# =========================================================================
#
# Tool exceptions can carry arbitrary text into the model's context as the
# `tool` message content. json.dumps() handles quote/backslash escaping so a
# raw injection of `</tool_call>` won't break message framing, but the model
# still *reads* those tokens and they can confuse downstream tool-call
# parsing or, in adversarial cases, nudge it toward role-confusion framing.
#
# This helper strips structural framing tokens (XML role tags, CDATA,
# markdown code fences) and caps the message at a sane upper bound before it
# becomes part of the conversation. It's defense-in-depth — the json layer
# already prevents framing escape — but cheap and worth having.
#
# Ported from ironclaw#1639.
_TOOL_ERROR_ROLE_TAG_RE = re.compile(
    r'</?(?:tool_call|function_call|result|response|output|input|system|assistant|user)>',
    re.IGNORECASE,
)
_TOOL_ERROR_FENCE_OPEN_RE = re.compile(r'^\s*```(?:json|xml|html|markdown)?\s*', re.MULTILINE)
_TOOL_ERROR_FENCE_CLOSE_RE = re.compile(r'\s*```\s*$', re.MULTILINE)
_TOOL_ERROR_CDATA_RE = re.compile(r'<!\[CDATA\[.*?\]\]>', re.DOTALL)
_TOOL_ERROR_MAX_LEN = 2000


def _sanitize_tool_error(error_msg: str) -> str:
    """Strip structural framing tokens from a tool error before showing it to the model.

    See _TOOL_ERROR_ROLE_TAG_RE docstring above for rationale.
    """
    if not error_msg:
        return "[TOOL_ERROR] "
    sanitized = _TOOL_ERROR_ROLE_TAG_RE.sub("", error_msg)
    sanitized = _TOOL_ERROR_FENCE_OPEN_RE.sub("", sanitized)
    sanitized = _TOOL_ERROR_FENCE_CLOSE_RE.sub("", sanitized)
    sanitized = _TOOL_ERROR_CDATA_RE.sub("", sanitized)
    if len(sanitized) > _TOOL_ERROR_MAX_LEN:
        sanitized = sanitized[:_TOOL_ERROR_MAX_LEN - 3] + "..."
    return f"[TOOL_ERROR] {sanitized}"


# =========================================================================
# Tool argument type coercion
# =========================================================================

def coerce_tool_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce tool call arguments to match their JSON Schema types.

    LLMs frequently return numbers as strings (``"42"`` instead of ``42``)
    and booleans as strings (``"true"`` instead of ``true``).  This compares
    each argument value against the tool's registered JSON Schema and attempts
    safe coercion when the value is a string but the schema expects a different
    type.  Original values are preserved when coercion fails.

    Handles ``"type": "integer"``, ``"type": "number"``, ``"type": "boolean"``,
    and union types (``"type": ["integer", "string"]``).

    Also wraps bare scalar values in a single-element list when the schema
    declares ``"type": "array"``.  Open-weight models (DeepSeek, Qwen, GLM)
    sometimes emit ``{"urls": "https://a.com"}`` when the tool expects
    ``{"urls": ["https://a.com"]}``; wrapping here avoids a confusing tool
    failure on what is otherwise a well-formed call.
    """
    if not args or not isinstance(args, dict):
        return args

    schema = registry.get_schema(tool_name)
    if not schema:
        return args

    properties = (schema.get("parameters") or {}).get("properties")
    if not properties:
        return args

    for key, value in list(args.items()):
        prop_schema = properties.get(key)
        if not prop_schema:
            continue
        expected = prop_schema.get("type")

        # Wrap bare non-list values when the schema declares ``array``.
        # Strings still go through _coerce_value first so JSON-encoded
        # arrays (``'["a","b"]'``) get parsed and nullable ``"null"``
        # becomes ``None`` rather than ``["null"]``.
        # ``None`` itself is preserved — we don't know whether the model
        # meant "omit" or "empty list", and tools with sensible defaults
        # (e.g. read_file's normalize_read_pagination) already handle it.
        if expected == "array" and value is not None and not isinstance(value, (list, tuple)):
            if isinstance(value, str):
                coerced = _coerce_value(value, expected, schema=prop_schema)
                if coerced is not value:
                    # _coerce_value handled it (JSON-parsed list or
                    # nullable "null" → None).
                    args[key] = coerced
                    continue
                # If the string looks like a JSON array but _coerce_value
                # failed to parse it, warn clearly instead of silently wrapping.
                if value.strip().startswith("["):
                    logger.warning(
                        "coerce_tool_args: %s.%s looks like a JSON array string "
                        "but could not be parsed — model may have emitted a "
                        "JSON-encoded string instead of a native array. "
                        "Falling back to single-element list.",
                        tool_name, key,
                    )
                args[key] = [value]
                logger.info(
                    "coerce_tool_args: wrapped bare string in list for %s.%s",
                    tool_name, key,
                )
                continue
            args[key] = [value]
            logger.info(
                "coerce_tool_args: wrapped bare %s in list for %s.%s",
                type(value).__name__, tool_name, key,
            )
            continue

        if not isinstance(value, str):
            continue
        if not expected and not _schema_allows_null(prop_schema):
            continue
        coerced = _coerce_value(value, expected, schema=prop_schema)
        if coerced is not value:
            args[key] = coerced

    return args


def _coerce_value(value: str, expected_type, schema: dict | None = None):
    """Attempt to coerce a string *value* to *expected_type*.

    Returns the original string when coercion is not applicable or fails.
    """
    if _schema_allows_null(schema) and value.strip().lower() == "null":
        return None

    if isinstance(expected_type, list):
        # Union type — try each in order, return first successful coercion
        for t in expected_type:
            result = _coerce_value(value, t, schema=schema)
            if result is not value:
                return result
        return value

    if expected_type in {"integer", "number"}:
        return _coerce_number(value, integer_only=(expected_type == "integer"))
    if expected_type == "boolean":
        return _coerce_boolean(value)
    if expected_type == "array":
        return _coerce_json(value, list)
    if expected_type == "object":
        return _coerce_json(value, dict)
    if expected_type == "null" and value.strip().lower() == "null":
        return None
    return value


def _schema_allows_null(schema: dict | None) -> bool:
    """Return True when a JSON Schema fragment explicitly permits null."""
    if not isinstance(schema, dict):
        return False

    schema_type = schema.get("type")
    if schema_type == "null":
        return True
    if isinstance(schema_type, list) and "null" in schema_type:
        return True
    if schema.get("nullable") is True:
        return True

    for union_key in ("anyOf", "oneOf"):
        variants = schema.get(union_key)
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if isinstance(variant, dict) and variant.get("type") == "null":
                return True

    return False


def _coerce_json(value: str, expected_python_type: type):
    """Parse *value* as JSON when the schema expects an array or object.

    Handles model output drift where a complex oneOf/discriminated-union schema
    causes the LLM to emit the array/object as a JSON string instead of a native
    structure.  Returns the original string if parsing fails or yields the wrong
    Python type.
    """
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError) as exc:
        logger.warning(
            "coerce_tool_args: failed to parse string as JSON for expected type %s: %s",
            expected_python_type.__name__,
            exc,
        )
        return value
    if isinstance(parsed, expected_python_type):
        logger.debug(
            "coerce_tool_args: coerced string to %s via json.loads",
            expected_python_type.__name__,
        )
        return parsed
    logger.warning(
        "coerce_tool_args: JSON-parsed value is %s, expected %s — skipping coercion",
        type(parsed).__name__,
        expected_python_type.__name__,
    )
    return value


def _coerce_number(value: str, integer_only: bool = False):
    """Try to parse *value* as a number.  Returns original string on failure."""
    try:
        f = float(value)
    except (ValueError, OverflowError):
        return value
    # Guard against inf/nan — not JSON-serializable, keep original string
    if f != f or f == float("inf") or f == float("-inf"):
        return value
    # If it looks like an integer (no fractional part), return int
    if f == int(f):
        return int(f)
    if integer_only:
        # Schema wants an integer but value has decimals — keep as string
        return value
    return f


def _coerce_boolean(value: str):
    """Try to parse *value* as a boolean.  Returns original string on failure."""
    low = value.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    return value


def _tool_result_observer_fields(result: Any) -> tuple[str, Optional[str], Optional[str]]:
    try:
        parsed_result = json.loads(result) if isinstance(result, str) else result
        if isinstance(parsed_result, dict) and parsed_result.get("error"):
            return "error", "tool_error", str(parsed_result.get("error"))
    except Exception:
        pass
    return "ok", None, None


def _emit_post_tool_call_hook(
    *,
    function_name: str,
    function_args: Dict[str, Any],
    result: Any,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    api_request_id: Optional[str] = None,
    duration_ms: int = 0,
    status: Optional[str] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    middleware_trace: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Emit the ``post_tool_call`` observer hook.

    No-ops cheaply when no plugin has registered for ``post_tool_call`` —
    the ``has_hook`` gate skips both the result-field derivation and the
    payload dispatch so the no-listener path costs one dict lookup.  When
    ``status`` is not supplied, the ok/error fields are derived from the
    result *after* the gate (parsing the result is only worth it when a
    listener will actually consume it).
    """
    try:
        from hermes_cli.plugins import has_hook, invoke_hook
        if not has_hook("post_tool_call"):
            return
        if status is None:
            status, error_type, error_message = _tool_result_observer_fields(result)
        invoke_hook(
            "post_tool_call",
            tool_name=function_name,
            args=function_args,
            result=result,
            task_id=task_id or "",
            session_id=session_id or "",
            tool_call_id=tool_call_id or "",
            turn_id=turn_id or "",
            api_request_id=api_request_id or "",
            duration_ms=duration_ms,
            status=status,
            error_type=error_type,
            error_message=error_message,
            middleware_trace=list(middleware_trace or []),
        )
    except Exception as _hook_err:
        logger.debug("post_tool_call hook error: %s", _hook_err)


def handle_function_call(
    function_name: str,
    function_args: Dict[str, Any],
    task_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    session_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    api_request_id: Optional[str] = None,
    user_task: Optional[str] = None,
    enabled_tools: Optional[List[str]] = None,
    skip_pre_tool_call_hook: bool = False,
    skip_tool_request_middleware: bool = False,
    tool_request_middleware_trace: Optional[List[Dict[str, Any]]] = None,
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
) -> str:
    """
    Main function call dispatcher that routes calls to the tool registry.

    Args:
        function_name: Name of the function to call.
        function_args: Arguments for the function.
        task_id: Unique identifier for terminal/browser session isolation.
        user_task: The user's original task (for browser_snapshot context).
        enabled_tools: Tool names enabled for this session.  When provided,
                       execute_code uses this list to determine which sandbox
                       tools to generate.  Falls back to the process-global
                       ``_last_resolved_tool_names`` for backward compat.
        enabled_toolsets: The session's enabled toolsets.  Used to scope the
                       Tool Search bridge catalog so ``tool_search`` /
                       ``tool_describe`` / ``tool_call`` only see and invoke
                       tools the session was actually granted.  ``None`` means
                       "no restriction" (the caller scopes to every toolset),
                       matching ``get_tool_definitions`` semantics.
        disabled_toolsets: The session's disabled toolsets, applied as a
                       subtraction when scoping the bridge catalog.

    Returns:
        Function result as a JSON string.
    """
    # Coerce string arguments to their schema-declared types (e.g. "42"→42)
    function_args = coerce_tool_args(function_name, function_args)
    if not isinstance(function_args, dict):
        function_args = {}
    _tool_middleware_trace = list(tool_request_middleware_trace or [])

    # ── Tool Search bridge dispatch ──────────────────────────────────
    # tool_search and tool_describe are pure catalog reads — handle them
    # inline. tool_call is unwrapped to the underlying tool so that every
    # downstream hook (pre/post, edit approval, guardrails) sees the real
    # tool name, not the bridge.
    _ts_mod = None
    try:
        from tools import tool_search as _ts_mod  # noqa: F401
    except Exception:
        _ts_mod = None

    if _ts_mod is not None and _ts_mod.is_bridge_tool(function_name):
        try:
            # Use skip_tool_search_assembly=True so we see the real catalog,
            # not the already-collapsed bridge-only list (the bridge would
            # otherwise be searching only itself).
            #
            # Scope the catalog to the session's toolsets so the bridge can
            # only surface and invoke tools the session was actually granted.
            # Without this, a restricted-toolset session (subagent, kanban
            # worker, curated gateway session) would see and be able to call
            # the entire process registry via the bridge. Passing the same
            # enabled/disabled toolsets the session was assembled with keeps
            # the deferred catalog identical to the deferrable subset of the
            # session's own tool list, and avoids polluting the process-global
            # _last_resolved_tool_names with out-of-scope tools.
            current_defs = get_tool_definitions(
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
                quiet_mode=True, skip_tool_search_assembly=True,
            ) or []
        except Exception:
            current_defs = []
        if function_name == _ts_mod.TOOL_SEARCH_NAME:
            return _ts_mod.dispatch_tool_search(function_args or {},
                                                current_tool_defs=current_defs)
        if function_name == _ts_mod.TOOL_DESCRIBE_NAME:
            return _ts_mod.dispatch_tool_describe(function_args or {},
                                                  current_tool_defs=current_defs)
        if function_name == _ts_mod.TOOL_CALL_NAME:
            underlying_name, underlying_args, err = _ts_mod.resolve_underlying_call(function_args or {})
            if err or not underlying_name:
                return json.dumps({"error": err or "tool_call could not be resolved"},
                                  ensure_ascii=False)
            # Defense in depth: the underlying tool MUST be in the session's
            # scoped deferrable catalog. resolve_underlying_call() only checks
            # that the name is deferrable in the global registry; this gate
            # additionally rejects any tool the session was not granted, so a
            # restricted session can never invoke an out-of-scope tool through
            # the bridge even if the catalog scoping above regressed.
            _scoped_deferrable = _ts_mod.scoped_deferrable_names(current_defs)
            if underlying_name not in _scoped_deferrable:
                return json.dumps({
                    "error": (
                        f"'{underlying_name}' is not available in this session. "
                        "Use tool_search to find tools you can call."
                    ),
                }, ensure_ascii=False)
            # Recurse with the underlying tool. All hooks fire against the
            # real tool name. The bridge is invisible to hooks by design.
            return handle_function_call(
                function_name=underlying_name,
                function_args=underlying_args,
                task_id=task_id,
                tool_call_id=tool_call_id,
                session_id=session_id,
                user_task=user_task,
                enabled_tools=enabled_tools,
                skip_pre_tool_call_hook=skip_pre_tool_call_hook,
                skip_tool_request_middleware=skip_tool_request_middleware,
                tool_request_middleware_trace=list(_tool_middleware_trace),
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
            )

    _tool_original_args = dict(function_args)
    if not skip_tool_request_middleware:
        try:
            from hermes_cli.middleware import apply_tool_request_middleware

            _tool_request_mw = apply_tool_request_middleware(
                function_name,
                function_args,
                task_id=task_id or "",
                session_id=session_id or "",
                tool_call_id=tool_call_id or "",
                turn_id=turn_id or "",
                api_request_id=api_request_id or "",
            )
            function_args = _tool_request_mw.payload
            _tool_original_args = _tool_request_mw.original_payload
            _tool_middleware_trace = _tool_request_mw.trace
        except Exception as _mw_err:
            logger.debug("tool_request middleware error: %s", _mw_err)

    try:
        if function_name in _AGENT_LOOP_TOOLS:
            return json.dumps({"error": f"{function_name} must be handled by the agent loop"})

        # Check plugin hooks for a block directive (unless caller already
        # checked — e.g. run_agent._invoke_tool passes skip=True to
        # avoid double-firing the hook).
        #
        # Single-fire contract: pre_tool_call fires exactly once per tool
        # execution. get_pre_tool_call_block_message() internally calls
        # invoke_hook("pre_tool_call", ...) and returns the first block
        # directive (if any), so observer plugins see the hook on that same
        # pass. When skip=True, the caller already fired it — do nothing
        # here.
        if not skip_pre_tool_call_hook:
            block_message: Optional[str] = None
            try:
                from hermes_cli.plugins import get_pre_tool_call_block_message
                block_message = get_pre_tool_call_block_message(
                    function_name,
                    function_args,
                    task_id=task_id or "",
                    session_id=session_id or "",
                    tool_call_id=tool_call_id or "",
                    turn_id=turn_id or "",
                    api_request_id=api_request_id or "",
                    middleware_trace=list(_tool_middleware_trace),
                )
            except Exception as _hook_err:
                logger.debug("pre_tool_call hook error: %s", _hook_err)

            if block_message is not None:
                result = json.dumps({"error": block_message}, ensure_ascii=False)
                _emit_post_tool_call_hook(
                    function_name=function_name,
                    function_args=function_args,
                    result=result,
                    task_id=task_id,
                    session_id=session_id,
                    tool_call_id=tool_call_id,
                    turn_id=turn_id,
                    api_request_id=api_request_id,
                    status="blocked",
                    error_type="plugin_block",
                    error_message=block_message,
                    middleware_trace=list(_tool_middleware_trace),
                )
                return result

        # ACP/Zed edit approval runs before any file mutation.  The requester
        # is bound via ContextVar only for ACP sessions, so CLI/gateway paths
        # are unaffected when it is unset.
        try:
            from acp_adapter.edit_approval import maybe_require_edit_approval

            edit_block_message = maybe_require_edit_approval(function_name, function_args)
            if edit_block_message is not None:
                return edit_block_message
        except Exception as _edit_approval_err:
            logger.debug("ACP edit approval guard error: %s", _edit_approval_err)
            if function_name in {"write_file", "patch"}:
                return json.dumps({"error": "Edit approval denied: approval guard failed"}, ensure_ascii=False)

        # Notify the read-loop tracker when a non-read/search tool runs,
        # so the *consecutive* counter resets (reads after other work are fine).
        if function_name not in _READ_SEARCH_TOOLS:
            try:
                from tools.file_tools import notify_other_tool_call
                notify_other_tool_call(task_id or "default")
            except Exception:
                pass  # file_tools may not be loaded yet

        # Measure tool dispatch latency so post_tool_call and
        # transform_tool_result hooks can observe per-tool duration.
        # Inspired by Claude Code 2.1.119, which added ``duration_ms`` to
        # PostToolUse hook inputs so plugin authors can build latency
        # dashboards, budget alerts, and regression canaries without having
        # to wrap every tool manually.  We use monotonic() so the value is
        # unaffected by wall-clock adjustments during the call.
        _dispatch_start = time.monotonic()
        _approval_tokens = None
        try:
            from tools.approval import (
                reset_current_observability_context,
                set_current_observability_context,
            )
            _approval_tokens = set_current_observability_context(
                turn_id=turn_id or "",
                tool_call_id=tool_call_id or "",
            )
        except Exception:
            reset_current_observability_context = None
        try:
            if function_name == "execute_code":
                # Prefer the caller-provided list so subagents can't overwrite
                # the parent's tool set via the process-global.
                sandbox_enabled = enabled_tools if enabled_tools is not None else _last_resolved_tool_names
                def _dispatch(next_args: Dict[str, Any]) -> Any:
                    return registry.dispatch(
                        function_name, next_args,
                        task_id=task_id,
                        session_id=session_id,
                        enabled_tools=sandbox_enabled,
                    )
            else:
                def _dispatch(next_args: Dict[str, Any]) -> Any:
                    return registry.dispatch(
                        function_name, next_args,
                        task_id=task_id,
                        session_id=session_id,
                        user_task=user_task,
                    )
            supervisor_block = _maybe_apply_supervisor_gate(
                function_name=function_name,
                function_args=function_args,
                user_task=user_task,
                task_id=task_id,
                session_id=session_id,
                tool_call_id=tool_call_id,
                turn_id=turn_id,
                api_request_id=api_request_id,
            )
            if supervisor_block is not None:
                _emit_post_tool_call_hook(
                    function_name=function_name,
                    function_args=function_args,
                    result=supervisor_block,
                    task_id=task_id,
                    session_id=session_id,
                    tool_call_id=tool_call_id,
                    turn_id=turn_id,
                    api_request_id=api_request_id,
                    status="blocked",
                    error_type="supervisor_gate_block",
                    error_message=supervisor_block,
                    middleware_trace=list(_tool_middleware_trace),
                )
                return supervisor_block

            from hermes_cli.middleware import run_tool_execution_middleware

            result = run_tool_execution_middleware(
                function_name,
                function_args,
                _dispatch,
                original_args=_tool_original_args,
                task_id=task_id or "",
                session_id=session_id or "",
                tool_call_id=tool_call_id or "",
                turn_id=turn_id or "",
                api_request_id=api_request_id or "",
            )
        finally:
            if _approval_tokens is not None and reset_current_observability_context is not None:
                try:
                    reset_current_observability_context(_approval_tokens)
                except Exception:
                    pass
        duration_ms = int((time.monotonic() - _dispatch_start) * 1000)

        _emit_post_tool_call_hook(
            function_name=function_name,
            function_args=function_args,
            result=result,
            task_id=task_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
            turn_id=turn_id,
            api_request_id=api_request_id,
            duration_ms=duration_ms,
            middleware_trace=list(_tool_middleware_trace),
        )

        # Generic tool-result canonicalization seam: plugins receive the
        # final result string (JSON, usually) and may replace it by
        # returning a string from transform_tool_result. Runs after
        # post_tool_call (which stays observational) and before the result
        # is appended back into conversation context. Fail-open; the first
        # valid string return wins; non-string returns are ignored.
        # Gated on has_hook so the no-listener path skips both the result
        # field derivation and the payload dispatch.
        try:
            from hermes_cli.plugins import has_hook, invoke_hook
            if has_hook("transform_tool_result"):
                status, error_type, error_message = _tool_result_observer_fields(result)
                hook_results = invoke_hook(
                    "transform_tool_result",
                    tool_name=function_name,
                    args=function_args,
                    result=result,
                    task_id=task_id or "",
                    session_id=session_id or "",
                    tool_call_id=tool_call_id or "",
                    turn_id=turn_id or "",
                    api_request_id=api_request_id or "",
                    duration_ms=duration_ms,
                    status=status,
                    error_type=error_type,
                    error_message=error_message,
                )
                for hook_result in hook_results:
                    if isinstance(hook_result, str):
                        result = hook_result
                        break
        except Exception as _hook_err:
            logger.debug("transform_tool_result hook error: %s", _hook_err)

        return result

    except Exception as e:
        error_msg = f"Error executing {function_name}: {str(e)}"
        logger.exception(error_msg)
        return json.dumps({"error": _sanitize_tool_error(error_msg)}, ensure_ascii=False)


# =============================================================================
# Backward-compat wrapper functions
# =============================================================================

def get_all_tool_names() -> List[str]:
    """Return all registered tool names."""
    return registry.get_all_tool_names()


def get_toolset_for_tool(tool_name: str) -> Optional[str]:
    """Return the toolset a tool belongs to."""
    return registry.get_toolset_for_tool(tool_name)


def get_available_toolsets() -> Dict[str, dict]:
    """Return toolset availability info for UI display."""
    return registry.get_available_toolsets()


def check_toolset_requirements() -> Dict[str, bool]:
    """Return {toolset: available_bool} for every registered toolset."""
    return registry.check_toolset_requirements()


def check_tool_availability(quiet: bool = False) -> Tuple[List[str], List[dict]]:
    """Return (available_toolsets, unavailable_info)."""
    return registry.check_tool_availability(quiet=quiet)
