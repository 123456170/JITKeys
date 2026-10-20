"""
JITKeys - Just-in-time privileged access broker.

Local-first, synthetic-data-only Streamlit application.
No real credentials are ever created.
No real hosts, domains, accounts, IPs, or systems are touched.
All actions are simulated against local SQLite lab state.
"""

import os
import re
import json
import time
import uuid
import random
import hashlib
import sqlite3
import tempfile
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio


APP_TITLE = "JITKeys - Just-in-time privileged access"
DEFAULT_REQUEST = (
    "I need db-admin on prod-orders for 45 minutes "
    "to fix a failed migration, ticket CHG-2231"
)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
DB_PATH = DATA_DIR / "jitkeys.db"

# Pillow / simulation colors as RGB tuples.
COLORS = {
    "bg": (7, 9, 15),
    "panel": (14, 19, 30),
    "panel2": (18, 26, 40),
    "border": (40, 55, 75),
    "text": (220, 235, 255),
    "muted": (130, 150, 170),
    "cyan": (0, 229, 255),
    "green": (57, 255, 136),
    "amber": (255, 191, 0),
    "red": (255, 64, 64),
}

# Plotly / CSS-safe hex colors.
HEX = {
    "bg": "#07090F",
    "panel": "#0E131E",
    "panel2": "#121A28",
    "border": "#283B4F",
    "text": "#DCEBFF",
    "muted": "#8AA2B7",
    "cyan": "#00E5FF",
    "green": "#39FF88",
    "amber": "#FFBF00",
    "red": "#FF4040",
    "orange": "#FF8C00",
}

LEVEL_COLORS = {
    "info": COLORS["cyan"],
    "ok": COLORS["green"],
    "warn": COLORS["amber"],
    "crit": COLORS["red"],
}

LEVEL_HEX = {
    "info": HEX["cyan"],
    "ok": HEX["green"],
    "warn": HEX["amber"],
    "crit": HEX["red"],
}

_FONT_CACHE: Dict[Any, Any] = {}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def dt_iso(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def parse_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def safe_json_loads(text: Any, default: Any = None) -> Any:
    if default is None:
        default = {}
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


def mask_secret(value: Any) -> str:
    s = str(value or "")
    if len(s) <= 12:
        return "*" * len(s)
    return s[:6] + "..." + s[-4:]


def human_delta(td: Optional[timedelta]) -> str:
    if td is None:
        return "unknown"
    total = int(td.total_seconds())
    if total <= 0:
        return "expired"
    days = total // 86400
    total %= 86400
    hours = total // 3600
    total %= 3600
    minutes = total // 60
    seconds = total % 60
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def plotly_color(value: Any) -> Any:
    """
    Convert Pillow RGB tuples into Plotly-safe color strings.
    Plotly does not accept tuples like (255, 64, 64).
    """
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        return f"rgb({int(value[0])}, {int(value[1])}, {int(value[2])})"
    return value


def fetch_all(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    cur = conn.execute(sql, params)
    return [dict(row) for row in cur.fetchall()]


def fetch_one(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    row = conn.execute(sql, params).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Database schema and synthetic reference data
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS resources (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    environment TEXT NOT NULL,
    sensitivity INTEGER NOT NULL,
    owner_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS roles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    privilege_level INTEGER NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT,
    department TEXT,
    risk_history INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    approved_by TEXT,
    change_type TEXT,
    risk_label TEXT,
    valid_until TEXT
);

CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,
    requester_id TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    role_id TEXT NOT NULL,
    duration_minutes INTEGER NOT NULL,
    justification TEXT,
    ticket_id TEXT,
    status TEXT NOT NULL,
    risk_score INTEGER,
    risk_level TEXT,
    auto_approved INTEGER DEFAULT 0,
    critic_verdict TEXT,
    ticket_valid INTEGER DEFAULT 0,
    created_at TEXT,
    updated_at TEXT,
    raw_text TEXT,
    details TEXT
);

CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    approver_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT,
    ts TEXT NOT NULL,
    gate_status TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    role_id TEXT NOT NULL,
    mock_credential TEXT NOT NULL,
    issued_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL,
    audit_summary TEXT
);

CREATE TABLE IF NOT EXISTS commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    command TEXT NOT NULL,
    flagged INTEGER DEFAULT 0,
    severity TEXT
);

CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    agent TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    kpi TEXT,
    prev_hash TEXT,
    entry_hash TEXT NOT NULL
);
"""


def get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def seed_reference_data(conn: sqlite3.Connection) -> None:
    if fetch_one(conn, "SELECT COUNT(*) AS c FROM resources")["c"] > 0:
        return

    resources = [
        ("prod-orders", "Production Orders DB", "production", 5, "u-owner-ops"),
        ("prod-payments", "Production Payments DB", "production", 5, "u-owner-sec"),
        ("vault-prod", "Production Secret Vault", "production", 5, "u-owner-sec"),
        ("k8s-prod", "Production Kubernetes Cluster", "production", 4, "u-owner-ops"),
        ("stage-analytics", "Staging Analytics Warehouse", "staging", 3, "u-owner-data"),
        ("dev-sandbox", "Development Sandbox", "development", 1, "u-owner-dev"),
    ]

    roles = [
        ("readonly", "Read Only", 1, "Read-only inspection access"),
        ("deployer", "Deployer", 3, "Deployment and rollout access"),
        ("secrets-reader", "Secrets Reader", 4, "Read selected secret material"),
        ("db-admin", "Database Admin", 5, "High-privilege database administration"),
        ("cluster-admin", "Cluster Admin", 5, "Kubernetes cluster administration"),
        ("shell-root", "Host Root Shell", 5, "Root-like host shell access"),
    ]

    users = [
        ("u-alice", "Alice Nguyen", "alice@example.local", "Finance Ops", 0),
        ("u-bob", "Bob Martinez", "bob@example.local", "Platform Engineering", 1),
        ("u-eve", "Eve Chen", "eve@example.local", "External Contractor", 3),
        ("u-owner-ops", "Ops Owner", "ops-owner@example.local", "Operations", 0),
        ("u-owner-sec", "Security Owner", "sec-owner@example.local", "Security", 0),
        ("u-owner-data", "Data Owner", "data-owner@example.local", "Data", 0),
        ("u-owner-dev", "Dev Owner", "dev-owner@example.local", "Development", 0),
        ("u-sec-lead", "Security Lead", "sec-lead@example.local", "Security", 0),
        ("u-ciso", "CISO", "ciso@example.local", "Executive", 0),
        ("u-change", "Change Manager", "change@example.local", "PMO", 0),
    ]

    far_future = "2030-01-01T00:00:00+00:00"
    tickets = [
        ("CHG-2231", "approved", "u-change", "migration", "normal", far_future),
        ("CHG-2232", "draft", None, "schema-change", "normal", far_future),
        ("CHG-2233", "rejected", "u-change", "emergency-access", "high", far_future),
        ("CHG-2240", "approved", "u-change", "routine", "low", far_future),
        ("CHG-2251", "closed", "u-change", "old-change", "normal", far_future),
        ("INC-9001", "approved", "u-sec-lead", "incident", "emergency", far_future),
    ]

    conn.executemany(
        "INSERT OR IGNORE INTO resources (id, name, environment, sensitivity, owner_id) VALUES (?, ?, ?, ?, ?)",
        resources,
    )
    conn.executemany(
        "INSERT OR IGNORE INTO roles (id, name, privilege_level, description) VALUES (?, ?, ?, ?)",
        roles,
    )
    conn.executemany(
        "INSERT OR IGNORE INTO users (id, name, email, department, risk_history) VALUES (?, ?, ?, ?, ?)",
        users,
    )
    conn.executemany(
        "INSERT OR IGNORE INTO tickets (id, status, approved_by, change_type, risk_label, valid_until) VALUES (?, ?, ?, ?, ?, ?)",
        tickets,
    )
    conn.commit()


def get_resource(conn: Optional[sqlite3.Connection], rid: str) -> Optional[Dict[str, Any]]:
    if not conn or not rid:
        return None
    return fetch_one(conn, "SELECT * FROM resources WHERE id = ?", (rid,))


def get_role(conn: Optional[sqlite3.Connection], rid: str) -> Optional[Dict[str, Any]]:
    if not conn or not rid:
        return None
    return fetch_one(conn, "SELECT * FROM roles WHERE id = ?", (rid,))


def get_user(conn: Optional[sqlite3.Connection], uid: str) -> Optional[Dict[str, Any]]:
    if not conn or not uid:
        return None
    return fetch_one(conn, "SELECT * FROM users WHERE id = ?", (uid,))


def get_ticket(conn: Optional[sqlite3.Connection], tid: str) -> Optional[Dict[str, Any]]:
    if not conn or not tid:
        return None
    return fetch_one(conn, "SELECT * FROM tickets WHERE id = ?", (tid,))


# ---------------------------------------------------------------------------
# Hash-chained audit log and shared event stream
# ---------------------------------------------------------------------------

def compute_audit_hash(prev_hash: Optional[str], ts: str, agent: str, level: str, msg: str, kpi: Any) -> str:
    payload = canonical_json({"ts": ts, "agent": agent, "level": level, "msg": msg, "kpi": kpi})
    base = (prev_hash or "GENESIS") + "|" + payload
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def emit_event(
    state: Optional[Dict[str, Any]],
    conn: sqlite3.Connection,
    agent: str,
    level: str,
    msg: str,
    kpi: Optional[Dict[str, Any]] = None,
    ts: Optional[datetime] = None,
) -> Dict[str, Any]:
    ts_dt = ts or utcnow()
    ts_str = dt_iso(ts_dt)
    kpi = kpi or {}

    prev_hash = None
    if state is not None:
        prev_hash = state.get("last_audit_hash")
    if prev_hash is None:
        row = fetch_one(conn, "SELECT entry_hash FROM audit_events ORDER BY id DESC LIMIT 1")
        prev_hash = row["entry_hash"] if row else None

    entry_hash = compute_audit_hash(prev_hash, ts_str, agent, level, msg, kpi)
    kpi_text = canonical_json(kpi)

    conn.execute(
        """
        INSERT INTO audit_events (ts, agent, level, message, kpi, prev_hash, entry_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (ts_str, agent, level, msg, kpi_text, prev_hash, entry_hash),
    )
    conn.commit()

    event = {
        "ts": ts_str,
        "agent": agent,
        "level": level,
        "msg": msg,
        "kpi": kpi,
        "hash": entry_hash,
    }

    if state is not None:
        state.setdefault("event_log", []).append(event)
        state["last_audit_hash"] = entry_hash
        if len(state["event_log"]) > 500:
            state["event_log"] = state["event_log"][-500:]

    return event


def verify_audit_chain(conn: sqlite3.Connection) -> tuple[int, int, bool]:
    rows = fetch_all(conn, "SELECT * FROM audit_events ORDER BY id ASC")
    prev_hash = None
    valid = 0
    for row in rows:
        kpi = safe_json_loads(row.get("kpi"), {})
        expected = compute_audit_hash(
            prev_hash,
            row["ts"],
            row["agent"],
            row["level"],
            row["message"],
            kpi,
        )
        row_prev = row.get("prev_hash") or None
        if expected == row["entry_hash"] and row_prev == prev_hash:
            valid += 1
        prev_hash = row["entry_hash"]
    total = len(rows)
    return valid, total, valid == total


def load_recent_events(conn: sqlite3.Connection, state: Dict[str, Any], limit: int = 200) -> None:
    if state.get("event_log_loaded"):
        return
    rows = fetch_all(conn, "SELECT * FROM audit_events ORDER BY id DESC LIMIT ?", (limit,))
    events = []
    for row in reversed(rows):
        events.append(
            {
                "ts": row["ts"],
                "agent": row["agent"],
                "level": row["level"],
                "msg": row["message"],
                "kpi": safe_json_loads(row.get("kpi"), {}),
                "hash": row["entry_hash"],
            }
        )
    state["event_log"] = events
    state["last_audit_hash"] = rows[0]["entry_hash"] if rows else None
    state["event_log_loaded"] = True


# ---------------------------------------------------------------------------
# Approval gate
# ---------------------------------------------------------------------------

def approval_gate(action: Dict[str, Any], approved: bool = False, dry_run: bool = True) -> Dict[str, Any]:
    """
    All consequential actions must pass through this gate.
    The app never performs real defensive or privileged actions.
    """
    action = dict(action) if isinstance(action, dict) else {"value": action}
    if not approved:
        return {"status": "pending_approval", "action": action}
    if dry_run:
        return {"status": "dry_run", "action": action}
    return {"status": "executed", "action": action}


# ---------------------------------------------------------------------------
# JSON schema validation and optional Gemini support
# ---------------------------------------------------------------------------

def _type_ok(value: Any, typ: Optional[str]) -> bool:
    if typ is None:
        return True
    if typ == "string":
        return isinstance(value, str)
    if typ == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if typ == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if typ == "boolean":
        return isinstance(value, bool)
    if typ == "array":
        return isinstance(value, list)
    if typ == "object":
        return isinstance(value, dict)
    return True


def validate_schema(data: Any, schema: Optional[Dict[str, Any]]) -> bool:
    if schema is None:
        return True
    if not isinstance(data, dict):
        return False
    for key in schema.get("required", []):
        if key not in data:
            return False
    properties = schema.get("properties", {})
    for key, spec in properties.items():
        if key in data and not _type_ok(data[key], spec.get("type")):
            return False
    return True


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise ValueError("Model output did not contain a JSON object")


def _sanitize_context_for_model(ctx: Dict[str, Any]) -> Dict[str, Any]:
    skip_keys = {"conn", "state", "api_key", "password", "secret", "authorization"}
    out: Dict[str, Any] = {}
    for key, value in (ctx or {}).items():
        if key in skip_keys:
            continue
        if isinstance(value, sqlite3.Row):
            value = dict(value)
        if isinstance(value, datetime):
            value = dt_iso(value)
        if isinstance(value, dict):
            value = _sanitize_context_for_model(value)
        elif isinstance(value, list):
            cleaned = []
            for item in value[:25]:
                if isinstance(item, dict):
                    cleaned.append(_sanitize_context_for_model(item))
                elif isinstance(item, sqlite3.Row):
                    cleaned.append(dict(item))
                else:
                    cleaned.append(str(item)[:300])
            value = cleaned
        elif isinstance(value, str):
            value = value[:1000]
        out[key] = value
    return out


def _call_gemini(prompt: str, api_key: str, model: str = "gemini-2.0-flash") -> str:
    import urllib.request
    import urllib.parse

    key = urllib.parse.quote_plus(api_key.strip())
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }
    req = urllib.request.Request(
        url,
        data=canonical_json(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    try:
        return payload["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:
        raise RuntimeError("Unexpected Gemini response shape") from exc


# ---------------------------------------------------------------------------
# Agent and Crew primitives
# ---------------------------------------------------------------------------

class Agent:
    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        demo_fn: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        schema: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None,
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.demo_fn = demo_fn
        self.schema = schema
        self.tools = tools or []

    def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        json_mode: bool = True,
        api_key: Optional[str] = None,
        retries: int = 3,
    ) -> Dict[str, Any]:
        context = context or {}

        if api_key and json_mode:
            schema_text = canonical_json(self.schema or {})
            prompt = (
                f"{self.system_prompt}\n\n"
                f"Return only valid JSON matching this schema:\n{schema_text}\n\n"
                f"Task:\n{task}\n\n"
                f"Context:\n{canonical_json(_sanitize_context_for_model(context))}"
            )
            for attempt in range(retries):
                try:
                    raw = _call_gemini(prompt, api_key)
                    data = _extract_json(raw)
                    if not validate_schema(data, self.schema):
                        raise ValueError("Schema validation failed")
                    return data
                except Exception:
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)

        try:
            data = self.demo_fn(task, context)
            if not validate_schema(data, self.schema):
                return {"error": "invalid_deterministic_output", "detail": canonical_json(data)[:500]}
            return data
        except Exception as exc:
            return {"error": "agent_exception", "detail": str(exc)}


class Crew:
    def __init__(self, agents: Dict[str, Agent], log_fn: Optional[Callable[..., Dict[str, Any]]] = None):
        self.agents = agents
        self.log_fn = log_fn or (lambda agent, level, msg, kpi=None: None)

    def _log(self, agent: str, level: str, msg: str, kpi: Optional[Dict[str, Any]] = None) -> None:
        try:
            self.log_fn(agent, level, msg, kpi or {})
        except Exception:
            pass

    def run(self, steps: List[tuple], state: Dict[str, Any]) -> Dict[str, Any]:
        for key, task, context in steps:
            agent = self.agents.get(key)
            if agent is None:
                state[key] = {"error": "missing_agent"}
                self._log("Crew Orchestrator", "crit", f"Missing agent: {key}")
                continue

            self._log(agent.name, "info", f"Started: {task[:140]}")
            result = agent.run(task, context, api_key=context.get("api_key"))
            state[key] = result
            state[agent.name] = result

            is_error = isinstance(result, dict) and bool(result.get("error"))
            level = "crit" if is_error else "ok"
            kpi = {"agent": agent.name, "result_keys": list(result.keys()) if isinstance(result, dict) else []}
            self._log(agent.name, level, f"Finished: {task[:140]}", kpi)

        return state


# ---------------------------------------------------------------------------
# Agent schemas
# ---------------------------------------------------------------------------

ORCHESTRATOR_SCHEMA = {
    "type": "object",
    "required": ["plan", "mode"],
    "properties": {
        "plan": {"type": "array"},
        "mode": {"type": "string"},
    },
}

INTERPRETER_SCHEMA = {
    "type": "object",
    "required": ["resource", "role", "duration_minutes", "justification", "ticket"],
    "properties": {
        "resource": {"type": "string"},
        "role": {"type": "string"},
        "duration_minutes": {"type": "integer"},
        "justification": {"type": "string"},
        "ticket": {"type": "string"},
        "confidence": {"type": "number"},
    },
}

RISK_SCHEMA = {
    "type": "object",
    "required": ["risk_score", "risk_level", "factors"],
    "properties": {
        "risk_score": {"type": "integer"},
        "risk_level": {"type": "string"},
        "ticket_valid": {"type": "boolean"},
        "factors": {"type": "array"},
    },
}

ROUTER_SCHEMA = {
    "type": "object",
    "required": ["approvers", "auto_approve", "policy_reason"],
    "properties": {
        "approvers": {"type": "array"},
        "auto_approve": {"type": "boolean"},
        "policy_reason": {"type": "string"},
        "escalation_path": {"type": "array"},
    },
}

BROKER_SCHEMA = {
    "type": "object",
    "required": ["mock_credential", "issued_at", "expires_at", "duration_minutes", "status"],
    "properties": {
        "mock_credential": {"type": "string"},
        "issued_at": {"type": "string"},
        "expires_at": {"type": "string"},
        "duration_minutes": {"type": "integer"},
        "status": {"type": "string"},
    },
}

AUDITOR_SCHEMA = {
    "type": "object",
    "required": ["summary", "risky_commands", "risk_score", "recommendation"],
    "properties": {
        "summary": {"type": "string"},
        "risky_commands": {"type": "array"},
        "risk_score": {"type": "integer"},
        "recommendation": {"type": "string"},
    },
}

CRITIC_SCHEMA = {
    "type": "object",
    "required": ["verdict", "concerns", "evidence"],
    "properties": {
        "verdict": {"type": "string"},
        "concerns": {"type": "array"},
        "evidence": {"type": "array"},
    },
}

EXECUTOR_SCHEMA = {
    "type": "object",
    "required": ["action", "approved", "dry_run", "status"],
    "properties": {
        "action": {"type": "string"},
        "approved": {"type": "boolean"},
        "dry_run": {"type": "boolean"},
        "status": {"type": "string"},
    },
}


# ---------------------------------------------------------------------------
# Deterministic demo agent implementations
# ---------------------------------------------------------------------------

ROLE_SYNONYMS = {
    "readonly": ["readonly", "read-only", "read only", "viewer", "select only"],
    "deployer": ["deployer", "deploy", "release", "rollout"],
    "secrets-reader": ["secrets-reader", "secret reader", "vault read", "secrets"],
    "db-admin": ["db-admin", "database admin", "db admin", "dba", "database administrator"],
    "cluster-admin": ["cluster-admin", "cluster admin", "kubectl admin", "kubernetes admin"],
    "shell-root": ["shell-root", "root shell", "host root", "sudo shell", "root"],
}


def parse_duration_minutes(text: str) -> int:
    total = 0
    pattern = r"(\d+)\s*(day|days|d|hour|hours|hr|hrs|h|minute|minutes|min|mins|m)\b"
    for value, unit in re.findall(pattern, text, flags=re.I):
        n = int(value)
        u = unit.lower()
        if u.startswith("d"):
            total += n * 1440
        elif u.startswith("h"):
            total += n * 60
        else:
            total += n
    if total <= 0:
        return 30
    return max(1, min(1440, total))


def parse_ticket(text: str) -> str:
    m = re.search(r"\b(?:CHG|INC|PROB|TCK|REQ)-\d{3,}\b", text, flags=re.I)
    return m.group(0).upper() if m else ""


def parse_justification(text: str, ticket: str, duration_minutes: int) -> str:
    j = text
    if ticket:
        j = re.sub(re.escape(ticket), " ", j, flags=re.I)
    j = re.sub(r"\bfor\s+\d+\s*(?:minutes?|mins?|m|hours?|hrs?|h|days?)\b", " ", j, flags=re.I)
    j = re.sub(r"\bI need\b|\bplease grant\b|\baccess\b", " ", j, flags=re.I)
    j = re.sub(r"\s+", " ", j).strip(" .,:;-")
    m = re.search(r"\bto\s+(.+)$", j, flags=re.I)
    if m:
        j = m.group(1).strip(" .,:;-")
    if not j:
        j = "No justification supplied"
    return j[:500]


def orchestrator_demo(task: str, context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan": ["interpreter", "risk", "router", "critic", "executor"],
        "mode": "dry_run" if context.get("dry_run", True) else "simulated_executed",
        "note": "No real systems are touched.",
    }


def interpreter_demo(task: str, context: Dict[str, Any]) -> Dict[str, Any]:
    text = (task or "").strip()
    conn = context.get("conn")
    resources = fetch_all(conn, "SELECT * FROM resources") if conn else []
    roles = fetch_all(conn, "SELECT * FROM roles") if conn else []

    resource_id = "unknown-resource"
    confidence = 0.45

    lowered = text.lower()
    for r in resources:
        if str(r["id"]).lower() in lowered or str(r["name"]).lower() in lowered:
            resource_id = r["id"]
            confidence = 0.95
            break

    if resource_id == "unknown-resource":
        m = re.search(r"\bon\s+([A-Za-z0-9_-]+)", text, flags=re.I)
        candidate = m.group(1).strip(".,;:") if m else ""
        for r in resources:
            if candidate and (candidate.lower() in str(r["id"]).lower() or str(r["id"]).lower().startswith(candidate.lower())):
                resource_id = r["id"]
                confidence = 0.75
                break

    role_id = "unknown-role"
    for rid, synonyms in ROLE_SYNONYMS.items():
        if any(s in lowered for s in synonyms):
            role_id = rid
            confidence = min(0.99, confidence + 0.05)
            break

    if role_id == "unknown-role":
        for r in roles:
            if str(r["id"]).lower() in lowered or str(r["name"]).lower() in lowered:
                role_id = r["id"]
                confidence = max(confidence, 0.70)
                break

    duration = parse_duration_minutes(text)
    ticket = parse_ticket(text)
    justification = parse_justification(text, ticket, duration)

    return {
        "resource": resource_id,
        "role": role_id,
        "duration_minutes": duration,
        "justification": justification,
        "ticket": ticket,
        "confidence": round(float(confidence), 2),
    }


def risk_assessor_demo(task: str, context: Dict[str, Any]) -> Dict[str, Any]:
    parsed = context.get("parsed") or {}
    conn = context.get("conn")
    now = context.get("now") or utcnow()
    requester_id = context.get("requester_id")

    resource = get_resource(conn, parsed.get("resource"))
    role = get_role(conn, parsed.get("role"))
    user = get_user(conn, requester_id)
    ticket = get_ticket(conn, parsed.get("ticket"))

    ticket_valid = bool(
        ticket
        and str(ticket.get("status", "")).lower() in ("approved", "open")
        and (not ticket.get("valid_until") or (parse_dt(ticket.get("valid_until")) or now) >= now)
    )

    score = 0
    factors: List[str] = []

    if resource:
        score += int(resource.get("sensitivity", 3)) * 6
        factors.append(f"Evidence: resource {resource['id']} sensitivity is {resource['sensitivity']}/5.")
        if str(resource.get("environment", "")).lower() == "production":
            score += 5
            factors.append("Evidence: target resource is in production.")
    else:
        score += 25
        factors.append("Lead: resource could not be matched to the synthetic catalog.")

    if role:
        score += int(role.get("privilege_level", 3)) * 5
        factors.append(f"Evidence: role {role['id']} privilege level is {role['privilege_level']}/5.")
    else:
        score += 20
        factors.append("Lead: role could not be matched to the synthetic role catalog.")

    duration = int(parsed.get("duration_minutes", 30) or 30)
    if duration > 120:
        score += 15
        factors.append(f"Evidence: requested duration {duration} minutes exceeds 2 hours.")
    elif duration > 60:
        score += 8
        factors.append(f"Evidence: requested duration {duration} minutes exceeds 1 hour.")
    elif duration > 30:
        score += 3
        factors.append(f"Evidence: requested duration {duration} minutes exceeds default 30-minute window.")

    if not parsed.get("ticket"):
        score += 20
        factors.append("Evidence: no ticket reference was supplied.")
    elif not ticket_valid:
        score += 25
        factors.append(f"Evidence: ticket {parsed.get('ticket')} is missing, draft, rejected, closed, or expired.")
    else:
        score -= 5
        factors.append(f"Evidence: ticket {parsed.get('ticket')} is valid in the mock ticket table.")

    hour = now.astimezone(timezone.utc).hour
    if hour < 7 or hour >= 19:
        score += 10
        factors.append(f"Evidence: request time {hour:02d}:00 UTC is outside normal business hours.")

    if user:
        history = int(user.get("risk_history", 0) or 0)
        score += history * 5
        factors.append(f"Evidence: requester {user['id']} has synthetic risk-history score {history}.")
        if conn:
            denied = fetch_one(
                conn,
                "SELECT COUNT(*) AS c FROM requests WHERE requester_id = ? AND status IN ('rejected', 'denied')",
                (user["id"],),
            )["c"]
            if denied:
                score += min(15, int(denied) * 4)
                factors.append(f"Evidence: requester has {denied} prior rejected synthetic requests.")
    else:
        score += 10
        factors.append("Lead: requester is not present in the synthetic user table.")

    justification = str(parsed.get("justification", "")).strip()
    if len(justification) < 10 or justification.lower().startswith("no justification"):
        score += 10
        factors.append("Evidence: justification is missing or too short.")

    if resource and role and resource.get("environment") == "production" and int(role.get("privilege_level", 0)) >= 4:
        score += 10
        factors.append("Evidence: production resource combined with high-privilege role.")

    score = max(0, min(100, int(score)))
    if score <= 25:
        level = "low"
    elif score <= 50:
        level = "medium"
    elif score <= 75:
        level = "high"
    else:
        level = "critical"

    return {
        "risk_score": score,
        "risk_level": level,
        "ticket_valid": ticket_valid,
        "factors": factors,
        "resource_sensitivity": int(resource.get("sensitivity", 0)) if resource else 0,
        "role_privilege": int(role.get("privilege_level", 0)) if role else 0,
    }


def approver_router_demo(task: str, context: Dict[str, Any]) -> Dict[str, Any]:
    parsed = context.get("parsed") or {}
    risk = context.get("risk") or {}
    conn = context.get("conn")
    resource = get_resource(conn, parsed.get("resource"))

    score = int(risk.get("risk_score", 100))
    level = str(risk.get("risk_level", "critical"))
    ticket_valid = bool(risk.get("ticket_valid"))
    owner = resource.get("owner_id") if resource else "u-sec-lead"

    approvers: List[str] = []
    escalation: List[str] = []
    auto = False
    reason = ""

    if level == "low" and ticket_valid and score <= 25:
        auto = True
        approvers = ["policy-auto"]
        escalation = ["policy-auto"]
        reason = "Low-risk, valid-ticket request eligible for automatic time-boxed approval."
    elif level == "medium":
        approvers = [owner]
        escalation = [owner, "u-sec-lead"]
        reason = "Medium-risk request requires resource owner approval."
    elif level == "high":
        approvers = [owner, "u-sec-lead"]
        escalation = [owner, "u-sec-lead", "u-ciso"]
        reason = "High-risk request requires resource owner and security lead approval."
    else:
        approvers = [owner, "u-sec-lead", "u-ciso"]
        escalation = [owner, "u-sec-lead", "u-ciso"]
        reason = "Critical-risk request requires owner, security lead, and CISO approval."

    return {
        "approvers": approvers,
        "auto_approve": auto,
        "policy_reason": reason,
        "escalation_path": escalation,
    }


def critic_skeptic_demo(task: str, context: Dict[str, Any]) -> Dict[str, Any]:
    parsed = context.get("parsed") or {}
    risk = context.get("risk") or {}
    router = context.get("router") or {}

    concerns: List[str] = []
    evidence: List[str] = []
    score = int(risk.get("risk_score", 100))
    level = str(risk.get("risk_level", "critical"))
    ticket_valid = bool(risk.get("ticket_valid"))
    auto = bool(router.get("auto_approve"))
    approvers = [str(a) for a in (router.get("approvers") or [])]

    if parsed.get("resource") == "unknown-resource":
        concerns.append("Resource was not resolved from the synthetic catalog.")
    if parsed.get("role") == "unknown-role":
        concerns.append("Role was not resolved from the synthetic catalog.")
    if not parsed.get("ticket"):
        concerns.append("No ticket reference was supplied.")
    elif not ticket_valid:
        concerns.append(f"Ticket {parsed.get('ticket')} is not valid in the mock ticket table.")
        evidence.append(f"Ticket status check failed for {parsed.get('ticket')}.")

    if auto and score > 25:
        concerns.append("Auto-approval was proposed despite risk score above low-risk threshold.")
        evidence.append(f"Risk score is {score}; auto-approval threshold is 25.")

    if level in ("high", "critical") and "u-sec-lead" not in approvers:
        concerns.append("High or critical request is missing security approval in the routing plan.")
    if level == "critical" and "u-ciso" not in approvers:
        concerns.append("Critical request is missing CISO approval in the routing plan.")

    duration = int(parsed.get("duration_minutes", 0) or 0)
    ticket_text = str(parsed.get("ticket", "")).upper()
    emergency = ticket_text.startswith("INC") or "emergency" in str(parsed.get("justification", "")).lower()
    if duration > 120 and not emergency:
        concerns.append("Requested duration exceeds two hours without an emergency ticket marker.")

    if len(str(parsed.get("justification", "")).strip()) < 10:
        concerns.append("Justification is too short for analyst review.")

    for f in (risk.get("factors") or [])[:5]:
        evidence.append(str(f))

    if not concerns:
        verdict = "approve"
    elif any("Auto-approval" in c for c in concerns) or (not ticket_valid and score >= 70):
        verdict = "reject"
    else:
        verdict = "revise"

    return {
        "verdict": verdict,
        "concerns": concerns,
        "evidence": evidence,
        "summary": "Critic reviewed structured request, risk evidence, routing policy, and ticket validity.",
    }


def session_broker_demo(task: str, context: Dict[str, Any]) -> Dict[str, Any]:
    now = context.get("now") or utcnow()
    duration = int(context.get("duration_minutes", 30) or 30)
    request_id = str(context.get("request_id", "MOCK"))
    expires = now + timedelta(minutes=duration)
    seed = canonical_json({"request_id": request_id, "duration": duration, "issued_at": dt_iso(now)})
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    credential = f"MOCK-JIT-{request_id[-8:]}-{digest}"
    return {
        "mock_credential": credential,
        "issued_at": dt_iso(now),
        "expires_at": dt_iso(expires),
        "duration_minutes": duration,
        "status": "mock_only_no_real_secret",
    }


def detect_command_risk(command: str) -> tuple[bool, str]:
    c = (command or "").upper()
    if re.search(r"\bDROP\s+(TABLE|DATABASE|SCHEMA|INDEX)\b", c):
        return True, "crit"
    if "TRUNCATE" in c:
        return True, "crit"
    if "DELETE FROM" in c and "WHERE" not in c:
        return True, "crit"
    if re.search(r"\bCREATE\s+USER\b|\bALTER\s+USER\b|\bGRANT\b|\bREVOKE\b", c):
        return True, "warn"
    if "CHMOD 777" in c:
        return True, "warn"
    if "RM -RF" in c or "RM -FR" in c:
        return True, "crit"
    if "SUDO" in c:
        return True, "warn"
    return False, "info"


def session_auditor_demo(task: str, context: Dict[str, Any]) -> Dict[str, Any]:
    commands = context.get("commands") or []
    risky: List[Dict[str, Any]] = []
    crit = 0
    warn = 0

    for cmd in commands:
        if isinstance(cmd, sqlite3.Row):
            cmd = dict(cmd)
        flagged = bool(cmd.get("flagged"))
        severity = str(cmd.get("severity", "info"))
        if flagged:
            risky.append(
                {
                    "ts": cmd.get("ts", ""),
                    "command": cmd.get("command", ""),
                    "severity": severity,
                    "evidence": "Command matched a synthetic risky-command pattern.",
                }
            )
            if severity == "crit":
                crit += 1
            elif severity == "warn":
                warn += 1

    score = min(100, len(risky) * 12 + crit * 8)
    summary = (
        f"Reviewed {len(commands)} synthetic commands. "
        f"Flagged {len(risky)} commands as leads for analyst review. "
        f"Critical-pattern count: {crit}. Warning-pattern count: {warn}."
    )
    if crit:
        recommendation = "Revoke remaining mock access, review command evidence, and require post-incident notes."
    elif warn:
        recommendation = "Analyst review recommended; no real system action was taken."
    else:
        recommendation = "No risky synthetic commands detected. Retain audit evidence."

    return {
        "summary": summary,
        "risky_commands": risky,
        "risk_score": score,
        "recommendation": recommendation,
    }


def executor_agent_demo(task: str, context: Dict[str, Any]) -> Dict[str, Any]:
    action = context.get("action") or {}
    approved = bool(context.get("approved", False))
    dry_run = bool(context.get("dry_run", True))
    gate = approval_gate(action, approved=approved, dry_run=dry_run)
    return {
        "action": str(action.get("type", "unknown")),
        "approved": approved,
        "dry_run": dry_run,
        "status": gate["status"],
        "note": "Executor can only act through approval_gate and only against local simulated state.",
    }


def build_agents() -> Dict[str, Agent]:
    return {
        "orchestrator": Agent(
            name="Crew Orchestrator",
            role="Plans intake, specialist, critic, and executor steps",
            system_prompt="You coordinate a local JIT access workflow. You never touch real systems.",
            demo_fn=orchestrator_demo,
            schema=ORCHESTRATOR_SCHEMA,
            tools=["plan", "route_steps"],
        ),
        "interpreter": Agent(
            name="Request Interpreter",
            role="Converts plain-English access requests into structured JSON",
            system_prompt="Extract resource, role, duration, justification, and ticket. Return JSON only.",
            demo_fn=interpreter_demo,
            schema=INTERPRETER_SCHEMA,
            tools=["regex_parser", "catalog_matcher"],
        ),
        "risk": Agent(
            name="Risk Assessor",
            role="Scores request risk using local synthetic evidence",
            system_prompt="Score risk from resource sensitivity, role privilege, ticket validity, time of day, and requester history.",
            demo_fn=risk_assessor_demo,
            schema=RISK_SCHEMA,
            tools=["ticket_check", "history_check", "time_check"],
        ),
        "router": Agent(
            name="Approver Router",
            role="Selects approvers and auto-approves only low-risk requests",
            system_prompt="Route according to risk level. Auto-approval is allowed only for low-risk valid-ticket requests.",
            demo_fn=approver_router_demo,
            schema=ROUTER_SCHEMA,
            tools=["policy_engine"],
        ),
        "critic": Agent(
            name="Critic Skeptic",
            role="Tries to disprove the proposed approval path",
            system_prompt="Challenge assumptions. Identify missing evidence, invalid tickets, excessive duration, or weak routing.",
            demo_fn=critic_skeptic_demo,
            schema=CRITIC_SCHEMA,
            tools=["consistency_checker"],
        ),
        "broker": Agent(
            name="Session Broker",
            role="Issues mock time-boxed credentials only",
            system_prompt="Create a mock credential with expiry. Never create real secrets.",
            demo_fn=session_broker_demo,
            schema=BROKER_SCHEMA,
            tools=["mock_token_minter"],
        ),
        "auditor": Agent(
            name="Session Auditor",
            role="Summarizes synthetic command logs and flags risky commands",
            system_prompt="Review synthetic commands. Flag DROP, mass DELETE, user creation, grants, chmod 777, rm -rf, and sudo as leads.",
            demo_fn=session_auditor_demo,
            schema=AUDITOR_SCHEMA,
            tools=["pattern_detector"],
        ),
        "executor": Agent(
            name="Executor",
            role="Prepares gated actions only",
            system_prompt="Do not act directly. Return the approval-gate status for the proposed action.",
            demo_fn=executor_agent_demo,
            schema=EXECUTOR_SCHEMA,
            tools=["approval_gate"],
        ),
    }


def get_agent_roster() -> List[Dict[str, str]]:
    return [{"name": a.name, "role": a.role} for a in build_agents().values()]


# ---------------------------------------------------------------------------
# Synthetic command generation and session lifecycle
# ---------------------------------------------------------------------------

BENIGN_COMMANDS = {
    "readonly": [
        "SELECT 1;",
        "SHOW TABLES;",
        "EXPLAIN SELECT * FROM orders LIMIT 10;",
        "SELECT status, COUNT(*) FROM orders GROUP BY status;",
    ],
    "deployer": [
        "helm list --namespace prod",
        "kubectl rollout status deployment/orders",
        "systemctl status app.service",
        "git rev-parse HEAD",
    ],
    "secrets-reader": [
        "vault kv get secret/data/orders",
        "vault token lookup",
        "vault policy list",
    ],
    "cluster-admin": [
        "kubectl get pods -A",
        "kubectl describe node worker-1",
        "helm history orders",
    ],
    "shell-root": [
        "whoami",
        "id",
        "uptime",
        "cat /etc/hostname",
    ],
    "db-admin": [
        "SELECT COUNT(*) FROM orders;",
        "BEGIN;",
        "UPDATE orders SET status='retry' WHERE id=1001;",
        "COMMIT;",
        "SELECT * FROM migration_log ORDER BY ts DESC LIMIT 20;",
    ],
}

RISKY_COMMANDS = [
    "DROP TABLE migration_backup;",
    "DELETE FROM orders;",
    "TRUNCATE TABLE audit_tmp;",
    "CREATE USER jit_test@'%' IDENTIFIED BY 'mock';",
    "GRANT ALL PRIVILEGES ON *.* TO jit_test;",
    "chmod 777 /var/run/app.sock",
    "rm -rf /tmp/cache/*",
    "sudo su -",
]


def generate_session_commands(
    conn: sqlite3.Connection,
    session_id: str,
    issued_at: str,
    expires_at: str,
    role_id: str,
    resource_id: str,
    force: bool = False,
) -> None:
    gate = approval_gate(
        {"type": "generate_synthetic_commands", "session_id": session_id, "requires_human": False},
        approved=True,
        dry_run=True,
    )
    if gate["status"] == "pending_approval":
        return

    existing = fetch_one(conn, "SELECT COUNT(*) AS c FROM commands WHERE session_id = ?", (session_id,))
    if existing and existing["c"] > 0 and not force:
        return

    start = parse_dt(issued_at) or utcnow()
    end = parse_dt(expires_at) or (start + timedelta(minutes=30))
    if end <= start:
        end = start + timedelta(minutes=1)

    resource = get_resource(conn, resource_id)
    role = get_role(conn, role_id)
    rng = random.Random(session_id)
    count = rng.randint(8, 14)
    span = max(60, int((end - start).total_seconds()))

    pool = BENIGN_COMMANDS.get(role_id, BENIGN_COMMANDS["readonly"])
    risky_enabled = bool(resource and int(resource.get("sensitivity", 0)) >= 4 and role and int(role.get("privilege_level", 0)) >= 4)

    rows = []
    for i in range(count):
        offset = int(span * (i + 1) / (count + 1)) + rng.randint(-3, 3)
        offset = max(0, min(span, offset))
        ts = start + timedelta(seconds=offset)
        if risky_enabled and rng.random() < 0.22:
            command = rng.choice(RISKY_COMMANDS)
        else:
            command = rng.choice(pool)
        flagged, severity = detect_command_risk(command)
        rows.append((session_id, dt_iso(ts), command, 1 if flagged else 0, severity))

    conn.executemany(
        "INSERT INTO commands (session_id, ts, command, flagged, severity) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def audit_session(
    conn: sqlite3.Connection,
    state: Dict[str, Any],
    session_id: str,
    now: Optional[datetime] = None,
    force: bool = False,
) -> Dict[str, Any]:
    session = fetch_one(conn, "SELECT * FROM sessions WHERE id = ?", (session_id,))
    if not session:
        return {"error": "session_not_found"}

    if session.get("audit_summary") and not force:
        return safe_json_loads(session["audit_summary"], {})

    commands = fetch_all(conn, "SELECT * FROM commands WHERE session_id = ? ORDER BY ts", (session_id,))
    auditor = build_agents()["auditor"]
    result = auditor.run("Summarize synthetic command log", {"commands": commands, "session_id": session_id})

    gate = approval_gate(
        {"type": "write_audit_summary", "session_id": session_id, "requires_human": False},
        approved=True,
        dry_run=True,
    )
    if gate["status"] != "pending_approval":
        conn.execute("UPDATE sessions SET audit_summary = ? WHERE id = ?", (canonical_json(result), session_id))
        conn.commit()

    risky_count = len(result.get("risky_commands", [])) if isinstance(result, dict) else 0
    level = "warn" if risky_count else "ok"
    emit_event(
        state,
        conn,
        "Session Auditor",
        level,
        str(result.get("summary", "Audit completed.")),
        {"session_id": session_id, "risky_count": risky_count},
        now,
    )
    return result


def issue_session(
    conn: sqlite3.Connection,
    state: Dict[str, Any],
    request_id: str,
    now: datetime,
    dry_run: bool = True,
    approved_by: Optional[str] = None,
) -> Dict[str, Any]:
    req = fetch_one(conn, "SELECT * FROM requests WHERE id = ?", (request_id,))
    if not req:
        return {"status": "error", "detail": "request_not_found"}

    if req["status"] != "approved":
        return {"status": "pending_approval", "action": {"request_id": request_id}}

    approval = fetch_one(
        conn,
        "SELECT * FROM approvals WHERE request_id = ? AND decision = 'approved' ORDER BY id DESC LIMIT 1",
        (request_id,),
    )
    if not approval:
        return {"status": "pending_approval", "action": {"request_id": request_id}}

    action = {
        "type": "issue_mock_credential",
        "request_id": request_id,
        "resource_id": req["resource_id"],
        "role_id": req["role_id"],
        "duration_minutes": req["duration_minutes"],
        "approved_by": approval["approver_id"],
        "requires_human": True,
    }
    gate = approval_gate(action, approved=True, dry_run=dry_run)
    if gate["status"] == "pending_approval":
        emit_event(state, conn, "Executor", "warn", f"Issue blocked by approval gate for {request_id}.", {}, now)
        return gate

    broker = build_agents()["broker"]
    broker_result = broker.run(
        "Issue mock credential",
        {"now": now, "duration_minutes": req["duration_minutes"], "request_id": request_id, "conn": conn},
    )

    if broker_result.get("error"):
        expires = now + timedelta(minutes=int(req["duration_minutes"]))
        credential = f"MOCK-JIT-{request_id[-8:]}-FALLBACK"
        expires_at = dt_iso(expires)
    else:
        credential = broker_result["mock_credential"]
        expires_at = broker_result["expires_at"]

    session_id = new_id("SES")
    conn.execute(
        """
        INSERT INTO sessions
        (id, request_id, user_id, resource_id, role_id, mock_credential, issued_at, expires_at, status, audit_summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', NULL)
        """,
        (
            session_id,
            request_id,
            req["requester_id"],
            req["resource_id"],
            req["role_id"],
            credential,
            dt_iso(now),
            expires_at,
        ),
    )
    conn.commit()

    generate_session_commands(conn, session_id, dt_iso(now), expires_at, req["role_id"], req["resource_id"])
    emit_event(
        state,
        conn,
        "Session Broker",
        "ok",
        f"Mock credential issued for {session_id}. Countdown starts. No real credential was created.",
        {"session_id": session_id, "expires_at": expires_at, "approved_by": approved_by or approval["approver_id"]},
        now,
    )
    audit_session(conn, state, session_id, now, force=False)

    return {
        "status": gate["status"],
        "session_id": session_id,
        "mock_credential": credential,
        "expires_at": expires_at,
        "approved_by": approval["approver_id"],
    }


def process_new_request(
    conn: sqlite3.Connection,
    state: Dict[str, Any],
    requester_id: str,
    raw_text: str,
    api_key: Optional[str] = None,
    dry_run: bool = True,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    now = now or utcnow()
    if not raw_text or not raw_text.strip():
        return {"error": "Empty request text."}

    log_fn = lambda agent, level, msg, kpi=None: emit_event(state, conn, agent, level, msg, kpi or {}, now)
    crew = Crew(build_agents(), log_fn)
    base_ctx = {
        "conn": conn,
        "requester_id": requester_id,
        "now": now,
        "api_key": api_key,
        "dry_run": dry_run,
    }

    results: Dict[str, Any] = {}
    crew.run(
        [
            ("orchestrator", "Plan request intake", base_ctx),
            ("interpreter", raw_text, base_ctx),
        ],
        results,
    )

    parsed = results.get("interpreter", {})
    if parsed.get("error"):
        emit_event(state, conn, "Request Interpreter", "crit", f"Interpreter failed: {parsed.get('detail', '')}", {}, now)
        return {"error": "Interpreter failed", "detail": parsed}

    risk_ctx = dict(base_ctx)
    risk_ctx["parsed"] = parsed
    crew.run([("risk", "Assess risk", risk_ctx)], results)
    risk = results.get("risk", {})

    router_ctx = dict(risk_ctx)
    router_ctx["risk"] = risk
    crew.run([("router", "Route approvers", router_ctx)], results)
    router = results.get("router", {})

    critic_ctx = dict(router_ctx)
    critic_ctx["router"] = router
    crew.run([("critic", "Challenge result", critic_ctx)], results)
    critic = results.get("critic", {})

    try:
        risk_score = int(risk.get("risk_score", 100))
    except Exception:
        risk_score = 100

    auto = bool(
        router.get("auto_approve")
        and critic.get("verdict") == "approve"
        and risk_score <= 25
        and risk.get("ticket_valid")
    )
    status = "approved" if auto else "pending"
    request_id = new_id("REQ")

    details = {
        "parsed": parsed,
        "risk": risk,
        "router": router,
        "critic": critic,
        "auto_candidate": auto,
    }

    conn.execute(
        """
        INSERT INTO requests
        (id, requester_id, resource_id, role_id, duration_minutes, justification, ticket_id,
         status, risk_score, risk_level, auto_approved, critic_verdict, ticket_valid,
         created_at, updated_at, raw_text, details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            request_id,
            requester_id,
            parsed.get("resource", "unknown-resource"),
            parsed.get("role", "unknown-role"),
            int(parsed.get("duration_minutes", 30) or 30),
            parsed.get("justification", ""),
            parsed.get("ticket", ""),
            status,
            risk_score,
            risk.get("risk_level", "critical"),
            1 if auto else 0,
            critic.get("verdict", "unknown"),
            1 if risk.get("ticket_valid") else 0,
            dt_iso(now),
            dt_iso(now),
            raw_text,
            canonical_json(details),
        ),
    )
    conn.commit()

    emit_event(
        state,
        conn,
        "Crew Orchestrator",
        "ok",
        f"Request {request_id} created with risk {risk_score} ({risk.get('risk_level', 'unknown')}).",
        {"request_id": request_id, "risk_score": risk_score},
        now,
    )

    proposed_action = {
        "type": "issue_mock_credential" if auto else "await_manual_approval",
        "request_id": request_id,
        "requires_human": not auto,
    }
    exec_ctx = {"action": proposed_action, "approved": auto, "dry_run": dry_run}
    crew.run([("executor", "Prepare gated action", exec_ctx)], results)
    executor_result = results.get("executor", {})

    issue_result = None
    if auto and executor_result.get("status") != "pending_approval":
        conn.execute(
            """
            INSERT INTO approvals (request_id, approver_id, decision, reason, ts, gate_status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                "policy-auto",
                "approved",
                router.get("policy_reason", "Low-risk automatic approval."),
                dt_iso(now),
                "dry_run",
            ),
        )
        conn.commit()
        issue_result = issue_session(conn, state, request_id, now, dry_run=dry_run, approved_by="policy-auto")
    else:
        approvers = router.get("approvers") or []
        emit_event(
            state,
            conn,
            "Approver Router",
            "info",
            f"Request {request_id} routed to approvers: {', '.join(approvers)}.",
            {"request_id": request_id},
            now,
        )

    return {
        "request_id": request_id,
        "parsed": parsed,
        "risk": risk,
        "router": router,
        "critic": critic,
        "executor": executor_result,
        "issue_result": issue_result,
        "auto_approved": auto,
    }


def approve_request(
    conn: sqlite3.Connection,
    state: Dict[str, Any],
    request_id: str,
    approver_id: str,
    reason: str,
    now: datetime,
    dry_run: bool = True,
) -> Dict[str, Any]:
    req = fetch_one(conn, "SELECT * FROM requests WHERE id = ?", (request_id,))
    if not req:
        return {"status": "error", "detail": "request_not_found"}
    if req["status"] != "pending":
        return {"status": "noop", "detail": "request_not_pending"}
    if approver_id == req["requester_id"]:
        emit_event(state, conn, "Executor", "crit", f"Blocked self-approval for {request_id}.", {}, now)
        return {"status": "blocked", "detail": "separation_of_duties"}

    action = {"type": "approve_request", "request_id": request_id, "approver_id": approver_id, "requires_human": True}
    gate = approval_gate(action, approved=True, dry_run=dry_run)
    if gate["status"] == "pending_approval":
        return gate

    conn.execute(
        """
        INSERT INTO approvals (request_id, approver_id, decision, reason, ts, gate_status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (request_id, approver_id, "approved", reason or "Approved by human reviewer.", dt_iso(now), gate["status"]),
    )
    conn.execute(
        "UPDATE requests SET status = 'approved', updated_at = ? WHERE id = ?",
        (dt_iso(now), request_id),
    )
    conn.commit()

    emit_event(state, conn, "Executor", "ok", f"Human approval recorded for {request_id}.", {"approver_id": approver_id}, now)
    return issue_session(conn, state, request_id, now, dry_run=dry_run, approved_by=approver_id)


def reject_request(
    conn: sqlite3.Connection,
    state: Dict[str, Any],
    request_id: str,
    approver_id: str,
    reason: str,
    now: datetime,
    dry_run: bool = True,
) -> Dict[str, Any]:
    req = fetch_one(conn, "SELECT * FROM requests WHERE id = ?", (request_id,))
    if not req or req["status"] != "pending":
        return {"status": "noop"}

    action = {"type": "reject_request", "request_id": request_id, "approver_id": approver_id, "requires_human": True}
    gate = approval_gate(action, approved=True, dry_run=dry_run)
    if gate["status"] == "pending_approval":
        return gate

    conn.execute(
        """
        INSERT INTO approvals (request_id, approver_id, decision, reason, ts, gate_status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (request_id, approver_id, "rejected", reason or "Rejected by human reviewer.", dt_iso(now), gate["status"]),
    )
    conn.execute(
        "UPDATE requests SET status = 'rejected', updated_at = ? WHERE id = ?",
        (dt_iso(now), request_id),
    )
    conn.commit()
    emit_event(state, conn, "Executor", "warn", f"Request {request_id} rejected.", {"approver_id": approver_id}, now)
    return {"status": gate["status"], "request_id": request_id}


def revoke_session(
    conn: sqlite3.Connection,
    state: Dict[str, Any],
    session_id: str,
    reason: str,
    now: datetime,
    dry_run: bool = True,
) -> Dict[str, Any]:
    session = fetch_one(conn, "SELECT * FROM sessions WHERE id = ?", (session_id,))
    if not session or session["status"] != "active":
        return {"status": "noop"}

    action = {"type": "revoke_mock_session", "session_id": session_id, "requires_human": True}
    gate = approval_gate(action, approved=True, dry_run=dry_run)
    if gate["status"] == "pending_approval":
        return gate

    conn.execute(
        "UPDATE sessions SET status = 'revoked', ended_at = ? WHERE id = ?",
        (dt_iso(now), session_id),
    )
    conn.commit()
    emit_event(state, conn, "Session Broker", "warn", f"Early revoke recorded for {session_id}. {reason}", {"session_id": session_id}, now)
    audit_session(conn, state, session_id, now, force=True)
    return {"status": gate["status"], "session_id": session_id}


def expire_due_sessions(conn: sqlite3.Connection, state: Dict[str, Any], now: datetime) -> int:
    rows = fetch_all(
        conn,
        "SELECT * FROM sessions WHERE status = 'active' AND expires_at <= ? ORDER BY expires_at",
        (dt_iso(now),),
    )
    count = 0
    for row in rows:
        action = {"type": "expire_mock_session", "session_id": row["id"], "requires_human": False}
        gate = approval_gate(action, approved=True, dry_run=True)
        if gate["status"] == "pending_approval":
            continue
        end_time = parse_dt(row["expires_at"]) or now
        conn.execute(
            "UPDATE sessions SET status = 'expired', ended_at = ? WHERE id = ?",
            (dt_iso(end_time), row["id"]),
        )
        conn.commit()
        emit_event(
            state,
            conn,
            "Session Broker",
            "crit",
            f"Mock access expired automatically for {row['id']}. The key crumbled at zero.",
            {"session_id": row["id"]},
            now,
        )
        audit_session(conn, state, row["id"], now, force=True)
        count += 1
    return count


# ---------------------------------------------------------------------------
# Demo scenario seeding
# ---------------------------------------------------------------------------

def _insert_request_direct(
    conn: sqlite3.Connection,
    rid: str,
    requester: str,
    resource: str,
    role: str,
    duration: int,
    justification: str,
    ticket: str,
    status: str,
    risk_score: int,
    risk_level: str,
    auto: bool,
    critic: str,
    ticket_valid: bool,
    created_at: datetime,
    raw_text: str,
    details: Dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO requests
        (id, requester_id, resource_id, role_id, duration_minutes, justification, ticket_id,
         status, risk_score, risk_level, auto_approved, critic_verdict, ticket_valid,
         created_at, updated_at, raw_text, details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            rid,
            requester,
            resource,
            role,
            duration,
            justification,
            ticket,
            status,
            risk_score,
            risk_level,
            1 if auto else 0,
            critic,
            1 if ticket_valid else 0,
            dt_iso(created_at),
            dt_iso(created_at),
            raw_text,
            canonical_json(details),
        ),
    )


def seed_demo_scenarios(conn: sqlite3.Connection, state: Dict[str, Any], now: datetime) -> None:
    # Active high-risk approved session.
    active_created = now - timedelta(minutes=10)
    active_expires = now + timedelta(minutes=35)
    active_details = {
        "parsed": {
            "resource": "prod-orders",
            "role": "db-admin",
            "duration_minutes": 45,
            "justification": "fix a failed migration",
            "ticket": "CHG-2231",
        },
        "risk": {
            "risk_score": 68,
            "risk_level": "high",
            "ticket_valid": True,
            "factors": [
                "Evidence: resource prod-orders sensitivity is 5/5.",
                "Evidence: role db-admin privilege level is 5/5.",
                "Evidence: ticket CHG-2231 is valid in the mock ticket table.",
            ],
        },
        "router": {
            "approvers": ["u-owner-ops", "u-sec-lead"],
            "auto_approve": False,
            "policy_reason": "High-risk production administrative role requires owner and security approval.",
        },
        "critic": {"verdict": "approve", "concerns": [], "evidence": ["Ticket CHG-2231 is approved."]},
    }
    _insert_request_direct(
        conn,
        "REQ-DEMO-ACTIVE",
        "u-alice",
        "prod-orders",
        "db-admin",
        45,
        "fix a failed migration",
        "CHG-2231",
        "approved",
        68,
        "high",
        False,
        "approve",
        True,
        active_created,
        DEFAULT_REQUEST,
        active_details,
    )
    conn.execute(
        """
        INSERT INTO approvals (request_id, approver_id, decision, reason, ts, gate_status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("REQ-DEMO-ACTIVE", "u-owner-ops", "approved", "Migration window confirmed.", dt_iso(active_created), "dry_run"),
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO sessions
        (id, request_id, user_id, resource_id, role_id, mock_credential, issued_at, expires_at, status, audit_summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', NULL)
        """,
        (
            "SES-DEMO-ACTIVE",
            "REQ-DEMO-ACTIVE",
            "u-alice",
            "prod-orders",
            "db-admin",
            "MOCK-JIT-DEMOACTIVE-1234567890ab",
            dt_iso(active_created),
            dt_iso(active_expires),
        ),
    )
    generate_session_commands(conn, "SES-DEMO-ACTIVE", dt_iso(active_created), dt_iso(active_expires), "db-admin", "prod-orders", force=True)

    # Pending medium-risk request with invalid draft ticket.
    pending_created = now - timedelta(minutes=3)
    pending_details = {
        "parsed": {
            "resource": "stage-analytics",
            "role": "readonly",
            "duration_minutes": 60,
            "justification": "inspect failed ETL output",
            "ticket": "CHG-2232",
        },
        "risk": {
            "risk_score": 51,
            "risk_level": "medium",
            "ticket_valid": False,
            "factors": ["Evidence: ticket CHG-2232 is draft and not valid for access."],
        },
        "router": {
            "approvers": ["u-owner-data"],
            "auto_approve": False,
            "policy_reason": "Medium-risk request requires resource owner approval.",
        },
        "critic": {"verdict": "revise", "concerns": ["Ticket is draft."], "evidence": ["Ticket status is draft."]},
    }
    _insert_request_direct(
        conn,
        "REQ-DEMO-PENDING",
        "u-bob",
        "stage-analytics",
        "readonly",
        60,
        "inspect failed ETL output",
        "CHG-2232",
        "pending",
        51,
        "medium",
        False,
        "revise",
        False,
        pending_created,
        "I need readonly on stage-analytics for 60 minutes to inspect failed ETL output, ticket CHG-2232",
        pending_details,
    )

    # Expired low-risk auto-approved session.
    expired_created = now - timedelta(minutes=90)
    expired_expires = now - timedelta(minutes=45)
    expired_details = {
        "parsed": {
            "resource": "dev-sandbox",
            "role": "readonly",
            "duration_minutes": 15,
            "justification": "check sample table",
            "ticket": "CHG-2240",
        },
        "risk": {
            "risk_score": 6,
            "risk_level": "low",
            "ticket_valid": True,
            "factors": ["Evidence: development sandbox sensitivity is 1/5."],
        },
        "router": {
            "approvers": ["policy-auto"],
            "auto_approve": True,
            "policy_reason": "Low-risk, valid-ticket request eligible for automatic time-boxed approval.",
        },
        "critic": {"verdict": "approve", "concerns": [], "evidence": ["Low sensitivity resource."]},
    }
    _insert_request_direct(
        conn,
        "REQ-DEMO-EXPIRED",
        "u-bob",
        "dev-sandbox",
        "readonly",
        15,
        "check sample table",
        "CHG-2240",
        "approved",
        6,
        "low",
        True,
        "approve",
        True,
        expired_created,
        "I need readonly on dev-sandbox for 15 minutes to check sample table, ticket CHG-2240",
        expired_details,
    )
    conn.execute(
        """
        INSERT INTO approvals (request_id, approver_id, decision, reason, ts, gate_status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("REQ-DEMO-EXPIRED", "policy-auto", "approved", "Low-risk automatic approval.", dt_iso(expired_created), "dry_run"),
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO sessions
        (id, request_id, user_id, resource_id, role_id, mock_credential, issued_at, expires_at, ended_at, status, audit_summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'expired', NULL)
        """,
        (
            "SES-DEMO-EXPIRED",
            "REQ-DEMO-EXPIRED",
            "u-bob",
            "dev-sandbox",
            "readonly",
            "MOCK-JIT-DEMOEXPIRED-abcdef123456",
            dt_iso(expired_created),
            dt_iso(expired_expires),
            dt_iso(expired_expires),
        ),
    )
    generate_session_commands(conn, "SES-DEMO-EXPIRED", dt_iso(expired_created), dt_iso(expired_expires), "readonly", "dev-sandbox", force=True)

    # Rejected critical request.
    rejected_created = now - timedelta(minutes=20)
    rejected_details = {
        "parsed": {
            "resource": "prod-payments",
            "role": "secrets-reader",
            "duration_minutes": 180,
            "justification": "urgent",
            "ticket": "",
        },
        "risk": {
            "risk_score": 100,
            "risk_level": "critical",
            "ticket_valid": False,
            "factors": ["Evidence: no ticket supplied.", "Evidence: production secrets role requested."],
        },
        "router": {
            "approvers": ["u-owner-sec", "u-sec-lead", "u-ciso"],
            "auto_approve": False,
            "policy_reason": "Critical-risk request requires owner, security lead, and CISO approval.",
        },
        "critic": {"verdict": "reject", "concerns": ["No ticket.", "Excessive duration."], "evidence": ["Missing ticket reference."]},
    }
    _insert_request_direct(
        conn,
        "REQ-DEMO-REJECTED",
        "u-eve",
        "prod-payments",
        "secrets-reader",
        180,
        "urgent",
        "",
        "rejected",
        100,
        "critical",
        False,
        "reject",
        False,
        rejected_created,
        "I need secrets-reader on prod-payments for 3 hours urgent",
        rejected_details,
    )
    conn.execute(
        """
        INSERT INTO approvals (request_id, approver_id, decision, reason, ts, gate_status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("REQ-DEMO-REJECTED", "u-sec-lead", "rejected", "No valid ticket and excessive duration.", dt_iso(rejected_created), "dry_run"),
    )

    seed_log = [
        ("Crew Orchestrator", "info", "Loaded synthetic JITKeys demo dataset."),
        ("Request Interpreter", "ok", "Parsed REQ-DEMO-ACTIVE: db-admin on prod-orders for 45 minutes, ticket CHG-2231."),
        ("Risk Assessor", "warn", "REQ-DEMO-ACTIVE scored high risk due to production database admin privileges."),
        ("Approver Router", "info", "REQ-DEMO-ACTIVE routed to resource owner and security lead."),
        ("Critic Skeptic", "ok", "Critic approved REQ-DEMO-ACTIVE because ticket CHG-2231 is valid."),
        ("Executor", "ok", "Approval gate recorded dry-run issuance for REQ-DEMO-ACTIVE."),
        ("Session Broker", "ok", "Mock credential SES-DEMO-ACTIVE minted with countdown ring."),
        ("Session Auditor", "warn", "Synthetic command log contains risky patterns for analyst review."),
        ("Request Interpreter", "ok", "Parsed REQ-DEMO-PENDING: readonly on stage-analytics, ticket CHG-2232."),
        ("Risk Assessor", "warn", "REQ-DEMO-PENDING ticket is draft and not valid."),
        ("Critic Skeptic", "warn", "REQ-DEMO-PENDING requires revision before approval."),
        ("Request Interpreter", "ok", "Parsed REQ-DEMO-EXPIRED: readonly on dev-sandbox for 15 minutes."),
        ("Approver Router", "ok", "REQ-DEMO-EXPIRED auto-approved as low-risk."),
        ("Session Broker", "crit", "SES-DEMO-EXPIRED countdown reached zero and crumbled."),
        ("Session Auditor", "ok", "Expired session audit summary written to hash-chained audit trail."),
        ("Request Interpreter", "crit", "Parsed REQ-DEMO-REJECTED: secrets-reader on prod-payments without ticket."),
        ("Critic Skeptic", "crit", "REQ-DEMO-REJECTED rejected for missing ticket and excessive duration."),
        ("Executor", "warn", "No real credential was ever created; all outcomes are simulated."),
    ]

    for agent, level, msg in seed_log:
        emit_event(state, conn, agent, level, msg, {}, now)

    audit_session(conn, state, "SES-DEMO-ACTIVE", now, force=True)
    audit_session(conn, state, "SES-DEMO-EXPIRED", now, force=True)

    conn.commit()
    state["event_log_loaded"] = True


def ensure_demo_data(conn: sqlite3.Connection, state: Dict[str, Any], now: datetime) -> None:
    init_db(conn)
    seed_reference_data(conn)
    count = fetch_one(conn, "SELECT COUNT(*) AS c FROM requests")["c"]
    if count == 0:
        seed_demo_scenarios(conn, state, now)
    elif not state.get("event_log_loaded"):
        load_recent_events(conn, state)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_metrics(conn: sqlite3.Connection, now: datetime) -> Dict[str, Any]:
    now_str = dt_iso(now)

    total_sessions = fetch_one(conn, "SELECT COUNT(*) AS c FROM sessions")["c"]
    approved_sessions = fetch_one(
        conn,
        """
        SELECT COUNT(*) AS c
        FROM sessions s
        WHERE EXISTS (
            SELECT 1
            FROM approvals a
            WHERE a.request_id = s.request_id
              AND a.decision = 'approved'
        )
        """,
    )["c"]
    policy_enforcement = 100.0 if total_sessions == 0 else (approved_sessions / total_sessions) * 100.0

    auto_total = fetch_one(conn, "SELECT COUNT(*) AS c FROM requests WHERE auto_approved = 1")["c"]
    auto_accurate = fetch_one(
        conn,
        """
        SELECT COUNT(*) AS c
        FROM requests
        WHERE auto_approved = 1
          AND critic_verdict = 'approve'
          AND risk_score <= 25
          AND ticket_valid = 1
        """,
    )["c"]
    auto_accuracy = 100.0 if auto_total == 0 else (auto_accurate / auto_total) * 100.0

    expected_end = fetch_one(
        conn,
        "SELECT COUNT(*) AS c FROM sessions WHERE expires_at <= ? OR status = 'revoked'",
        (now_str,),
    )["c"]
    correct_end = fetch_one(
        conn,
        """
        SELECT COUNT(*) AS c
        FROM sessions
        WHERE (expires_at <= ? OR status = 'revoked')
          AND status IN ('expired', 'revoked')
          AND ended_at IS NOT NULL
        """,
        (now_str,),
    )["c"]
    expiry_correctness = 100.0 if expected_end == 0 else (correct_end / expected_end) * 100.0

    valid_audit, total_audit, chain_ok = verify_audit_chain(conn)
    audit_completeness = 100.0 if total_audit == 0 else (valid_audit / total_audit) * 100.0

    active_sessions = fetch_one(conn, "SELECT COUNT(*) AS c FROM sessions WHERE status = 'active'")["c"]
    pending_requests = fetch_one(conn, "SELECT COUNT(*) AS c FROM requests WHERE status = 'pending'")["c"]

    return {
        "Policy enforcement %": round(policy_enforcement, 1),
        "Auto-approval accuracy %": round(auto_accuracy, 1),
        "Expiry correctness %": round(expiry_correctness, 1),
        "Audit completeness %": round(audit_completeness, 1),
        "Active sessions": int(active_sessions),
        "Pending requests": int(pending_requests),
        "Audit chain valid": chain_ok,
        "Audit events": int(total_audit),
    }


# ---------------------------------------------------------------------------
# Streamlit theme and session state
# ---------------------------------------------------------------------------

def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {HEX['bg']};
            color: {HEX['text']};
            font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
        }}
        header, .stHeader, h1, h2, h3, h4, h5, h6 {{
            color: {HEX['cyan']} !important;
            font-family: 'JetBrains Mono', 'Consolas', monospace;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: {HEX['bg']};
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {HEX['panel']};
            border: 1px solid {HEX['border']};
            border-radius: 8px;
            padding: 8px 14px;
            color: {HEX['cyan']};
        }}
        .stTabs [aria-selected="true"] {{
            background-color: #102A32;
            border-color: {HEX['green']};
            color: {HEX['green']};
        }}
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
            background-color: {HEX['panel']};
            color: {HEX['cyan']};
            border: 1px solid {HEX['cyan']};
            border-radius: 8px;
            font-family: 'JetBrains Mono', 'Consolas', monospace;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            background-color: #102A32;
            color: {HEX['green']};
            border-color: {HEX['green']};
        }}
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div,
        .stMultiselect div[data-baseweb="select"] > div, .stNumberInput input {{
            background-color: {HEX['panel']} !important;
            color: {HEX['text']} !important;
            border: 1px solid {HEX['border']} !important;
            font-family: 'JetBrains Mono', 'Consolas', monospace;
        }}
        [data-testid="stMetricValue"] {{
            color: {HEX['green']};
            font-family: 'JetBrains Mono', 'Consolas', monospace;
        }}
        [data-testid="stMetricLabel"] {{
            color: {HEX['muted']};
            font-family: 'JetBrains Mono', 'Consolas', monospace;
        }}
        div[data-testid="stExpander"] details {{
            background-color: {HEX['panel']};
            border: 1px solid {HEX['border']};
            border-radius: 10px;
        }}
        div[data-testid="stExpander"] summary {{
            color: {HEX['cyan']};
        }}
        .stDataFrame {{
            background-color: {HEX['panel']};
        }}
        code {{
            color: {HEX['green']};
            background-color: {HEX['panel']};
        }}
        .sidebar-content, section[data-testid="stSidebar"] {{
            background-color: #0B0F17 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    st.session_state.setdefault("event_log", [])
    st.session_state.setdefault("last_audit_hash", None)
    st.session_state.setdefault("event_log_loaded", False)
    st.session_state.setdefault("clock_offset", 0)
    st.session_state.setdefault("api_key", "")
    st.session_state.setdefault("mode", "Demo")
    st.session_state.setdefault("dry_run", True)
    st.session_state.setdefault("selected_user", "u-alice")
    st.session_state.setdefault("approver_id", "u-owner-ops")
    st.session_state.setdefault("last_workflow", None)
    st.session_state.setdefault("auto_video_done", False)
    st.session_state.setdefault("auto_video", None)
    st.session_state.setdefault("auto_video_error", None)
    st.session_state.setdefault("sim_video", None)
    st.session_state.setdefault("sim_events", [])


def get_sim_now() -> datetime:
    offset = int(st.session_state.get("clock_offset", 0))
    return utcnow() + timedelta(seconds=offset)


# ---------------------------------------------------------------------------
# Simulation Studio rendering
# ---------------------------------------------------------------------------

def get_font(size: int, bold: bool = False):
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    candidates = []
    if bold:
        candidates.append(r"C:\Windows\Fonts\consolab.ttf")
    candidates.extend(
        [
            r"C:\Windows\Fonts\consola.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/System/Library/Fonts/Menlo.ttc",
        ]
    )

    font = None
    for path in candidates:
        try:
            if os.path.exists(path):
                font = ImageFont.truetype(path, size)
                break
        except Exception:
            pass

    if font is None:
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", size)
        except Exception:
            try:
                font = ImageFont.load_default(size=size)
            except TypeError:
                font = ImageFont.load_default()

    _FONT_CACHE[key] = font
    return font


def fit_text(draw: ImageDraw.Draw, text: str, font: Any, max_width: int) -> str:
    text = str(text)
    try:
        if draw.textlength(text, font=font) <= max_width:
            return text
        while text and draw.textlength(text + "...", font=font) > max_width:
            text = text[:-1]
        return text + "..."
    except Exception:
        return text[: max(8, int(max_width / 10))]


def mix_color(c1: tuple, c2: tuple, t: float) -> tuple:
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def draw_key(draw: ImageDraw.Draw, cx: int, cy: int, color: tuple, scale: float = 1.0) -> None:
    s = max(0.1, float(scale))
    head_r = int(20 * s)
    hx = cx - int(35 * s)
    hy = cy
    draw.ellipse([hx - head_r, hy - head_r, hx + head_r, hy + head_r], outline=color, width=max(2, int(5 * s)))
    draw.rectangle([hx + head_r - 2, cy - int(5 * s), cx + int(45 * s), cy + int(5 * s)], fill=color)
    draw.rectangle([cx + int(25 * s), cy + int(5 * s), cx + int(32 * s), cy + int(18 * s)], fill=color)
    draw.rectangle([cx + int(38 * s), cy + int(5 * s), cx + int(45 * s), cy + int(15 * s)], fill=color)


def draw_countdown_ring(draw: ImageDraw.Draw, cx: int, cy: int, r: int, remaining: float, color: tuple) -> None:
    draw.arc([cx - r, cy - r, cx + r, cy + r], start=0, end=360, fill=COLORS["border"], width=8)
    remaining = max(0.0, min(1.0, remaining))
    if remaining > 0:
        draw.arc([cx - r, cy - r, cx + r, cy + r], start=-90, end=-90 + 360 * remaining, fill=color, width=8)


def draw_key_scene(draw: ImageDraw.Draw, progress: float, events: List[Dict[str, Any]]) -> None:
    cx, cy = 610, 585
    r = 58

    draw.ellipse([cx - r - 12, cy - r - 12, cx + r + 12, cy + r + 12], fill=(10, 15, 24), outline=COLORS["border"])

    if progress < 0.12:
        t = progress / 0.12
        key_color = mix_color(COLORS["panel"], COLORS["cyan"], t)
        draw_key(draw, cx, cy, key_color, 1.0)
        draw_countdown_ring(draw, cx, cy, r, 1.0, COLORS["cyan"])
        draw.text((cx - 72, cy + 76), "Minting mock key", font=get_font(14), fill=COLORS["cyan"])
    elif progress < 0.28:
        t = (progress - 0.12) / 0.16
        remaining = 1.0 - 0.15 * t
        draw_countdown_ring(draw, cx, cy, r, remaining, COLORS["green"])
        draw_key(draw, cx, cy, COLORS["green"], 1.0)
        draw.text((cx - 72, cy + 76), "Credential in use", font=get_font(14), fill=COLORS["green"])
    elif progress < 0.78:
        t = (progress - 0.28) / 0.50
        remaining = 0.85 * (1.0 - t)
        color = COLORS["green"] if remaining > 0.5 else COLORS["amber"] if remaining > 0.2 else COLORS["red"]
        draw_countdown_ring(draw, cx, cy, r, remaining, color)
        draw_key(draw, cx, cy, color, 1.0)
        draw.text((cx - 88, cy + 76), f"Countdown {int(remaining * 100)}%", font=get_font(14), fill=color)
    elif progress < 0.90:
        t = (progress - 0.78) / 0.12
        draw_countdown_ring(draw, cx, cy, r, 0.0, COLORS["red"])
        draw_key(draw, cx, cy, mix_color(COLORS["red"], COLORS["bg"], t), max(0.2, 1.0 - t))
        for i in range(16):
            angle = (i / 16.0) * 6.28
            dist = 18 + t * 70
            x = int(cx + np.cos(angle) * dist)
            y = int(cy + np.sin(angle) * dist)
            size = max(1, int(7 * (1.0 - t)))
            draw.rectangle([x - size, y - size, x + size, y + size], fill=mix_color(COLORS["red"], COLORS["bg"], t))
        draw.text((cx - 82, cy + 76), "Key crumbled at zero", font=get_font(14), fill=COLORS["red"])
    else:
        t = min(1.0, (progress - 0.90) / 0.10)
        draw_countdown_ring(draw, cx, cy, r, 0.0, COLORS["red"])
        card_w, card_h = 360, 155
        final_x = cx - card_w / 2
        start_x = 930
        x = int(start_x + (final_x - start_x) * t)
        y1 = cy - card_h // 2
        y2 = cy + card_h // 2
        draw.rounded_rectangle([x, y1, x + card_w, y2], radius=12, fill=(16, 28, 38), outline=COLORS["green"], width=2)
        draw.text((x + 18, y1 + 16), "AUDIT SUMMARY", font=get_font(18), fill=COLORS["green"])
        draw.text((x + 18, y1 + 46), "Mock credential expired", font=get_font(14), fill=COLORS["text"])
        draw.text((x + 18, y1 + 68), "Risky commands reviewed", font=get_font(14), fill=COLORS["text"])
        draw.text((x + 18, y1 + 90), "No real systems changed", font=get_font(14), fill=COLORS["cyan"])
        draw.text((x + 18, y1 + 112), "Hash-chained audit updated", font=get_font(14), fill=COLORS["muted"])


def render_frame(
    title: str,
    agents: List[Dict[str, str]],
    events: List[Dict[str, Any]],
    kpis: Dict[str, Any],
    progress: float,
) -> Image.Image:
    width, height = 1280, 720
    img = Image.new("RGB", (width, height), COLORS["bg"])
    d = ImageDraw.Draw(img)

    d.rounded_rectangle([20, 20, width - 20, 82], radius=12, fill=COLORS["panel"], outline=COLORS["cyan"], width=1)
    d.text((35, 33), fit_text(d, title, get_font(24), 460), font=get_font(24), fill=COLORS["text"])

    bar_x, bar_y, bar_w, bar_h = 530, 45, 500, 16
    d.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], radius=8, fill=(30, 40, 55))
    fill_w = int(bar_w * max(0.0, min(1.0, progress)))
    d.rounded_rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h], radius=8, fill=COLORS["green"])
    d.text((bar_x + bar_w + 15, 38), f"{int(progress * 100)}%", font=get_font(18), fill=COLORS["muted"])

    left = [20, 100, 280, 700]
    center = [300, 100, 920, 700]
    right = [940, 100, 1260, 700]
    for rect in (left, center, right):
        d.rounded_rectangle(rect, radius=12, fill=COLORS["panel"], outline=COLORS["border"], width=1)

    d.text((35, 112), "AGENT ROSTER", font=get_font(16), fill=COLORS["cyan"])
    d.text((315, 112), "LIVE AGENT LOG", font=get_font(16), fill=COLORS["cyan"])
    d.text((955, 112), "KPI CARDS", font=get_font(16), fill=COLORS["cyan"])

    active_agent = events[-1].get("agent") if events else None

    y = 142
    for agent in agents[:8]:
        name = str(agent.get("name", ""))
        role = str(agent.get("role", ""))
        is_active = name == active_agent
        if is_active:
            d.rounded_rectangle([30, y - 6, 270, y + 46], radius=8, fill=(0, 58, 70))
        d.text((40, y), fit_text(d, name, get_font(15), 220), font=get_font(15), fill=COLORS["green"] if is_active else COLORS["text"])
        d.text((40, y + 20), fit_text(d, role, get_font(12), 220), font=get_font(12), fill=COLORS["muted"])
        y += 56

    y = 138
    visible = events[-15:]
    for ev in visible:
        level = str(ev.get("level", "info"))
        color = LEVEL_COLORS.get(level, COLORS["text"])
        ts = str(ev.get("ts", ""))[11:19]
        agent = str(ev.get("agent", ""))[:13]
        msg = str(ev.get("msg", ""))
        line = f"{ts} {agent:<13} {msg}"
        d.text((315, y), fit_text(d, line, get_font(13), 590), font=get_font(13), fill=color)
        y += 22

    y = 142
    for label, value in list(kpis.items())[:4]:
        d.rounded_rectangle([950, y, 1250, y + 90], radius=10, fill=COLORS["panel2"], outline=COLORS["border"])
        d.text((965, y + 12), fit_text(d, label, get_font(14), 270), font=get_font(14), fill=COLORS["muted"])
        if "%" in label and isinstance(value, (int, float)):
            val_text = f"{float(value):.1f}%"
        else:
            val_text = str(value)
        d.text((965, y + 38), fit_text(d, val_text, get_font(28), 270), font=get_font(28), fill=COLORS["green"])
        y += 105

    draw_key_scene(d, max(0.0, min(1.0, progress)), events)
    return img


def script_events(scenario: str = "Normal day") -> List[Dict[str, Any]]:
    now = utcnow()
    events: List[Dict[str, Any]] = []

    def add(agent: str, level: str, msg: str, kpi: Optional[Dict[str, Any]] = None) -> None:
        events.append(
            {
                "ts": dt_iso(now + timedelta(seconds=len(events) * 3)),
                "agent": agent,
                "level": level,
                "msg": msg,
                "kpi": kpi or {},
            }
        )

    if scenario == "Active attack":
        add("Crew Orchestrator", "info", "Attack simulation started: suspicious privileged access attempt.")
        add("Request Interpreter", "warn", "Parsed request: shell-root on prod-payments for 4 hours, no ticket.")
        add("Risk Assessor", "crit", "Risk score 100 critical: production root access without ticket.")
        add("Approver Router", "warn", "Routed to owner, security lead, and CISO. Auto-approval disabled.")
        add("Critic Skeptic", "crit", "Critic rejects request: missing ticket, off-hours, high-history requester.")
        add("Executor", "warn", "Approval gate returned pending_approval. No credential issued.")
        add("Session Broker", "info", "Existing mock session SES-ATTACK-1 observed with unusual commands.")
        add("Session Auditor", "crit", "Flagged DROP TABLE and mass DELETE as critical synthetic leads.")
        add("Session Broker", "warn", "Early revoke simulated for SES-ATTACK-1 through approval gate.")
        add("Crew Orchestrator", "ok", "Attack path contained in local simulation only.")
        add("Risk Assessor", "warn", "Similar future requests elevated to critical by policy.")
        add("Approver Router", "info", "Escalation path stored for analyst review.")
        add("Session Auditor", "ok", "Audit summary written to hash-chained trail.")
        add("Executor", "ok", "No real IP block, account disable, or host action occurred.")
        add("Crew Orchestrator", "ok", "Active attack simulation complete.")
    elif scenario == "Recovery":
        add("Crew Orchestrator", "info", "Recovery simulation started: cleanup expired mock keys.")
        add("Session Broker", "crit", "Countdown reached zero for SES-RECOVERY-1; mock key crumbled.")
        add("Session Auditor", "ok", "Recovered session audit summary generated.")
        add("Risk Assessor", "info", "Historical risk features updated from synthetic audit.")
        add("Approver Router", "ok", "Policy tuned: low-risk dev sandbox remains auto-eligible.")
        add("Critic Skeptic", "ok", "No unresolved concerns in recovery dataset.")
        add("Executor", "ok", "All recovery actions passed approval gate in dry-run mode.")
        add("Session Broker", "info", "Mock credential registry cleared for expired sessions.")
        add("Crew Orchestrator", "ok", "Recovery simulation complete.")
        add("Session Auditor", "ok", "Audit completeness reported at 100 percent for local chain.")
        add("Risk Assessor", "info", "Evaluation metrics refreshed from SQLite lab state.")
        add("Approver Router", "info", "Approval inbox drained.")
        add("Request Interpreter", "ok", "Natural-language parser ready for next request.")
        add("Crew Orchestrator", "ok", "JITKeys returned to steady state.")
    else:
        add("Crew Orchestrator", "info", "Normal day simulation: JIT key lifecycle.")
        add("Request Interpreter", "ok", "Parsed request: db-admin on prod-orders for 45 minutes, ticket CHG-2231.")
        add("Risk Assessor", "warn", "Risk score 68 high: production database admin privileges.")
        add("Approver Router", "info", "Routed to resource owner and security lead.")
        add("Critic Skeptic", "ok", "Ticket CHG-2231 is valid; request can proceed with approvals.")
        add("Executor", "ok", "Approval gate approved dry-run issuance.")
        add("Session Broker", "ok", "Mock key minted with countdown ring.")
        add("Session Broker", "info", "Credential used in synthetic session.")
        add("Session Auditor", "warn", "Flagged DELETE without WHERE as a risky synthetic command.")
        add("Session Auditor", "crit", "Flagged DROP TABLE as a critical synthetic command lead.")
        add("Session Broker", "warn", "Countdown ring draining.")
        add("Session Broker", "crit", "Countdown reached zero.")
        add("Session Broker", "crit", "Mock key crumbled; no real credential existed.")
        add("Session Auditor", "ok", "Audit summary card generated.")
        add("Crew Orchestrator", "ok", "Hash-chained audit trail updated.")
        add("Risk Assessor", "info", "Policy enforcement measured from local approval records.")
        add("Approver Router", "info", "Auto-approval accuracy measured on synthetic data.")
        add("Executor", "ok", "All state changes were simulated against SQLite lab state.")
        add("Crew Orchestrator", "ok", "Normal day simulation complete.")

    return events


def export_mp4(
    title: str,
    agents: List[Dict[str, str]],
    events: List[Dict[str, Any]],
    kpi_fn: Any,
    fps: int = 10,
    hold: int = 8,
) -> bytes:
    if not events:
        events = script_events("Normal day")

    hold = max(1, int(hold))
    total_frames = max(1, len(events) * hold)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    try:
        try:
            writer = imageio.get_writer(
                tmp.name,
                fps=fps,
                codec="libx264",
                format="FFMPEG",
                macro_block_size=1,
                pixelformat="yuv420p",
            )
        except Exception:
            try:
                writer = imageio.get_writer(tmp.name, fps=fps, codec="libx264")
            except Exception:
                writer = imageio.get_writer(tmp.name, fps=fps)

        for i in range(total_frames):
            progress = i / float(total_frames - 1) if total_frames > 1 else 1.0
            event_idx = min(len(events) - 1, i // hold)
            kpis = kpi_fn(progress) if callable(kpi_fn) else kpi_fn
            frame = render_frame(title, agents, events[: event_idx + 1], kpis, progress)
            writer.append_data(np.asarray(frame, dtype=np.uint8))

        writer.close()
        return Path(tmp.name).read_bytes()
    except Exception as exc:
        raise RuntimeError(
            "Could not encode MP4. Install imageio and imageio-ffmpeg, then retry. "
            f"Detail: {exc}"
        ) from exc
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def get_simulation_events(state: Dict[str, Any], scenario: str, source: str) -> List[Dict[str, Any]]:
    real = state.get("event_log", [])
    if source.startswith("Real") and len(real) >= 15:
        return real[-30:]
    return script_events(scenario)


def play_live_simulation(
    events: List[Dict[str, Any]],
    agents: List[Dict[str, str]],
    kpis: Dict[str, Any],
    speed: float,
) -> None:
    frame_slot = st.empty()
    prog_slot = st.empty()
    status_slot = st.empty()
    total = max(1, len(events))

    for i, ev in enumerate(events, start=1):
        frame = render_frame("JITKeys live simulation", agents, events[:i], kpis, i / total)
        frame_slot.image(frame, use_container_width=True)
        prog_slot.progress(i / total)
        status_slot.markdown(f"**Event {i}/{total}:** {str(ev.get('msg', ''))[:160]}")
        time.sleep(max(0.05, float(speed)))

    status_slot.success("Simulation complete. Use Record video to save the MP4.")


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def risk_gauge(score: int, level: str) -> None:
    color_hex = {
        "low": HEX["green"],
        "medium": HEX["amber"],
        "high": HEX["orange"],
        "critical": HEX["red"],
    }.get(level, HEX["cyan"])

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"color": HEX["text"]}},
            title={"text": f"Risk level: {level}", "font": {"color": HEX["cyan"]}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": HEX["muted"]},
                "bar": {"color": color_hex},
                "bgcolor": HEX["panel"],
                "borderwidth": 1,
                "bordercolor": HEX["border"],
                "threshold": {
                    "line": {"color": HEX["green"], "width": 3},
                    "thickness": 0.75,
                    "value": 25,
                },
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor=HEX["bg"],
        plot_bgcolor=HEX["panel"],
        font=dict(family="JetBrains Mono, Consolas, monospace", color=HEX["text"]),
    )
    st.plotly_chart(fig, use_container_width=True)


def kpi_row(metrics: Dict[str, Any]) -> None:
    cols = st.columns(6)
    cols[0].metric("Policy enforcement", f"{metrics.get('Policy enforcement %', 0):.1f}%")
    cols[1].metric("Auto-approval accuracy", f"{metrics.get('Auto-approval accuracy %', 0):.1f}%")
    cols[2].metric("Expiry correctness", f"{metrics.get('Expiry correctness %', 0):.1f}%")
    cols[3].metric("Audit completeness", f"{metrics.get('Audit completeness %', 0):.1f}%")
    cols[4].metric("Active sessions", metrics.get("Active sessions", 0))
    cols[5].metric("Pending requests", metrics.get("Pending requests", 0))


def show_workflow(out: Dict[str, Any]) -> None:
    if out.get("error"):
        st.error(f"Workflow error: {out.get('error')} {out.get('detail', '')}")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Structured request")
        st.json(out.get("parsed", {}))
    with c2:
        st.markdown("#### Risk assessment")
        risk = out.get("risk", {})
        risk_gauge(int(risk.get("risk_score", 0)), str(risk.get("risk_level", "unknown")))
        for factor in (risk.get("factors") or [])[:8]:
            st.markdown(f"- {factor}")

    st.markdown("#### Approver routing")
    st.json(out.get("router", {}))

    st.markdown("#### Critic / skeptic")
    critic = out.get("critic", {})
    verdict = str(critic.get("verdict", "unknown"))
    color = {"approve": "green", "revise": "orange", "reject": "red"}.get(verdict, "gray")
    st.markdown(f"Verdict: **:{color}[{verdict}]**")
    st.json(critic)

    issue = out.get("issue_result")
    if issue and issue.get("session_id"):
        st.success(
            f"Mock session {issue['session_id']} issued. "
            f"Expires {issue.get('expires_at', 'unknown')}. No real credential was created."
        )
    elif out.get("auto_approved"):
        st.success("Auto-approved low-risk request.")
    else:
        st.info("Request is pending human approval in the Approval Inbox.")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def tab_request_console(conn: sqlite3.Connection, state: Dict[str, Any], now: datetime) -> None:
    st.subheader("Request Console")
    st.caption("Type a plain-English privileged access request. The crew parses, scores, routes, challenges, and gates execution.")

    users = fetch_all(conn, "SELECT id, name FROM users ORDER BY name")
    user_ids = [u["id"] for u in users]
    user_names = {u["id"]: u["name"] for u in users}

    def fmt_user(uid: str) -> str:
        return f"{user_names.get(uid, uid)} ({uid})"

    idx = user_ids.index(state.get("selected_user")) if state.get("selected_user") in user_ids else 0
    selected_id = st.selectbox("Requester", user_ids, index=idx, format_func=fmt_user, key="requester_select")
    state["selected_user"] = selected_id

    raw = st.text_area("Access request", value=DEFAULT_REQUEST, height=110, key="raw_request")
    lab_confirm = st.checkbox(
        "I confirm this request targets only synthetic/local lab resources. No real domain or host is touched.",
        value=True,
        key="lab_confirm",
    )

    c1, c2 = st.columns([1, 3])
    with c1:
        run = st.button("Run JITKeys crew", type="primary", use_container_width=True, disabled=not lab_confirm, key="run_crew")
    with c2:
        st.markdown(
            f"Dry-run: **{'ON' if state.get('dry_run') else 'OFF (still simulated)'}** | "
            f"Mode: **{state.get('mode', 'Demo')}** | "
            f"Simulated clock: `{dt_iso(now)}`"
        )

    if run:
        api_key = state.get("api_key") if state.get("mode") == "Live" else None
        try:
            with st.spinner("Running planner -> interpreter -> risk -> router -> critic -> executor..."):
                out = process_new_request(
                    conn,
                    state,
                    selected_id,
                    raw,
                    api_key=api_key,
                    dry_run=bool(state.get("dry_run", True)),
                    now=now,
                )
            state["last_workflow"] = out
        except Exception as exc:
            st.error(f"Request workflow failed safely: {exc}")
            emit_event(state, conn, "Crew Orchestrator", "crit", f"Workflow error: {exc}", {}, now)

    if state.get("last_workflow"):
        show_workflow(state["last_workflow"])


def tab_approval_inbox(conn: sqlite3.Connection, state: Dict[str, Any], now: datetime) -> None:
    st.subheader("Approval Inbox")
    st.caption("Human approvals are recorded locally. Approval only releases a simulated mock credential through the approval gate.")

    users = fetch_all(conn, "SELECT id, name FROM users ORDER BY name")
    user_ids = [u["id"] for u in users]
    user_names = {u["id"]: u["name"] for u in users}

    def fmt_user(uid: str) -> str:
        return f"{user_names.get(uid, uid)} ({uid})"

    idx = user_ids.index(state.get("approver_id")) if state.get("approver_id") in user_ids else 0
    selected_approver_id = st.selectbox(
        "Acting approver",
        user_ids,
        index=idx,
        format_func=fmt_user,
        key="acting_approver",
    )
    state["approver_id"] = selected_approver_id

    pending = fetch_all(
        conn,
        """
        SELECT r.*, res.name AS resource_name, rol.name AS role_name, u.name AS requester_name
        FROM requests r
        LEFT JOIN resources res ON r.resource_id = res.id
        LEFT JOIN roles rol ON r.role_id = rol.id
        LEFT JOIN users u ON r.requester_id = u.id
        WHERE r.status = 'pending'
        ORDER BY r.created_at DESC
        """,
    )

    if not pending:
        st.success("No pending requests. The inbox is clear.")
        return

    for req in pending:
        details = safe_json_loads(req.get("details"), {})
        risk = details.get("risk", {})
        router = details.get("router", {})
        critic = details.get("critic", {})

        with st.expander(
            f"{req['id']} | {req['requester_name']} | {req['resource_name']} | {req['role_name']} | risk {req['risk_score']}",
            expanded=True,
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Duration", f"{req['duration_minutes']} min")
            c2.metric("Ticket", req["ticket_id"] or "none")
            c3.metric("Critic", critic.get("verdict", "unknown"))

            st.markdown("**Justification**")
            st.write(req["justification"] or "No justification supplied.")

            st.markdown("**Risk evidence**")
            for factor in (risk.get("factors") or [])[:8]:
                st.markdown(f"- {factor}")

            st.markdown("**Routing**")
            st.write(", ".join(router.get("approvers") or []))
            st.caption(router.get("policy_reason", ""))

            st.markdown("**Critic concerns**")
            for concern in (critic.get("concerns") or [])[:8]:
                st.markdown(f"- {concern}")

            reason = st.text_input("Approval / rejection reason", key=f"reason-{req['id']}")
            b1, b2 = st.columns(2)
            if b1.button("Approve", key=f"approve-{req['id']}", type="primary"):
                res = approve_request(
                    conn,
                    state,
                    req["id"],
                    selected_approver_id,
                    reason,
                    now,
                    dry_run=bool(state.get("dry_run", True)),
                )
                if res.get("status") == "blocked":
                    st.warning("Blocked by separation of duties: approver cannot be the requester.")
                elif res.get("session_id"):
                    st.success(f"Approved. Mock session {res['session_id']} issued in dry-run mode.")
                    st.rerun()
                else:
                    st.info(f"Approval recorded. Gate status: {res.get('status', 'unknown')}")
                    st.rerun()

            if b2.button("Reject", key=f"reject-{req['id']}"):
                reject_request(
                    conn,
                    state,
                    req["id"],
                    selected_approver_id,
                    reason,
                    now,
                    dry_run=bool(state.get("dry_run", True)),
                )
                st.success("Request rejected. No credential was issued.")
                st.rerun()


def tab_active_sessions(conn: sqlite3.Connection, state: Dict[str, Any], now: datetime) -> None:
    st.subheader("Active Sessions")
    st.caption("Countdowns use the simulated clock. Advance the clock to watch expiry, or revoke early. All credentials are mock-only.")

    active = fetch_all(
        conn,
        """
        SELECT s.*, res.name AS resource_name, rol.name AS role_name, u.name AS user_name
        FROM sessions s
        LEFT JOIN resources res ON s.resource_id = res.id
        LEFT JOIN roles rol ON s.role_id = rol.id
        LEFT JOIN users u ON s.user_id = u.id
        WHERE s.status = 'active'
        ORDER BY s.expires_at
        """,
    )

    if not active:
        st.info("No active mock sessions. Issue a request or advance the simulated clock.")

    for s in active:
        st.markdown("---")
        expires = parse_dt(s["expires_at"])
        remaining = (expires - now) if expires else timedelta(0)

        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        c1.metric("Session", s["id"])
        c2.metric("Countdown", human_delta(remaining))
        c3.metric("Role", s["role_name"])
        c4.metric("Resource", s["resource_name"])

        st.markdown(
            f"User: **{s['user_name']}** | Mock credential: `{mask_secret(s['mock_credential'])}` | "
            f"Issued: `{s['issued_at']}` | Expires: `{s['expires_at']}`"
        )

        cmds = fetch_all(conn, "SELECT ts, command, flagged, severity FROM commands WHERE session_id = ? ORDER BY ts", (s["id"],))
        with st.expander("Synthetic command log"):
            if cmds:
                st.dataframe(pd.DataFrame(cmds), use_container_width=True, hide_index=True)
            else:
                st.write("No synthetic commands recorded yet.")

        audit = safe_json_loads(s.get("audit_summary"), {})
        if audit:
            with st.expander("Audit summary"):
                st.write(audit.get("summary", "No summary."))
                st.markdown("**Risky command leads**")
                st.json(audit.get("risky_commands", []))
                st.caption(audit.get("recommendation", ""))

        reason = st.text_input("Revoke reason", value="Early revoke requested by analyst.", key=f"revoke-reason-{s['id']}")
        b1, b2, b3 = st.columns(3)
        if b1.button("Advance clock to expiry", key=f"advance-{s['id']}"):
            seconds = int(remaining.total_seconds()) + 1
            if seconds > 0:
                state["clock_offset"] = int(state.get("clock_offset", 0)) + seconds
                st.rerun()
        if b2.button("Revoke now", key=f"revoke-{s['id']}"):
            revoke_session(conn, state, s["id"], reason, now, dry_run=bool(state.get("dry_run", True)))
            st.rerun()
        if b3.button("Audit now", key=f"audit-{s['id']}"):
            res = audit_session(conn, state, s["id"], now, force=True)
            st.json(res)

    ended = fetch_all(
        conn,
        """
        SELECT s.*, res.name AS resource_name, rol.name AS role_name, u.name AS user_name
        FROM sessions s
        LEFT JOIN resources res ON s.resource_id = res.id
        LEFT JOIN roles rol ON s.role_id = rol.id
        LEFT JOIN users u ON s.user_id = u.id
        WHERE s.status IN ('expired', 'revoked')
        ORDER BY s.ended_at DESC
        LIMIT 50
        """,
    )
    if ended:
        st.markdown("### Ended mock sessions")
        df = pd.DataFrame(ended)
        st.dataframe(df[["id", "user_name", "resource_name", "role_name", "status", "issued_at", "expires_at", "ended_at"]], use_container_width=True, hide_index=True)


def tab_audit_trail(conn: sqlite3.Connection, state: Dict[str, Any], now: datetime) -> None:
    st.subheader("Audit Trail")
    st.caption("Every decision and outcome is stored in a local hash-chained SQLite audit table. Results are leads for analyst review.")

    df = pd.read_sql_query(
        """
        SELECT id, ts, agent, level, message, kpi, entry_hash
        FROM audit_events
        ORDER BY id DESC
        LIMIT 1000
        """,
        conn,
    )

    c1, c2, c3 = st.columns(3)
    levels = c1.multiselect("Levels", ["info", "ok", "warn", "crit"], default=[], key="audit_levels")
    agents_list = sorted(df["agent"].unique().tolist()) if not df.empty else []
    agent_filter = c2.selectbox("Agent", ["All"] + agents_list, key="audit_agent")
    search = c3.text_input("Search message", value="", key="audit_search")

    filtered = df.copy()
    if levels:
        filtered = filtered[filtered["level"].isin(levels)]
    if agent_filter != "All":
        filtered = filtered[filtered["agent"] == agent_filter]
    if search:
        filtered = filtered[filtered["message"].str.contains(search, case=False, na=False)]

    st.dataframe(filtered[["ts", "agent", "level", "message"]], use_container_width=True, hide_index=True)

    valid, total, ok = verify_audit_chain(conn)
    if ok:
        st.success(f"Hash chain valid: {valid}/{total} audit entries verified.")
    else:
        st.error(f"Hash chain mismatch: {valid}/{total} entries verified. Treat as tamper lead.")

    if not filtered.empty:
        pc1, pc2 = st.columns(2)
        with pc1:
            fig1 = px.bar(
                filtered,
                x="agent",
                color="level",
                title="Audit events by agent",
                color_discrete_map=LEVEL_HEX,
            )
            fig1.update_layout(template="plotly_dark", paper_bgcolor=HEX["bg"], plot_bgcolor=HEX["panel"], font=dict(color=HEX["text"]))
            st.plotly_chart(fig1, use_container_width=True)
        with pc2:
            fig2 = px.scatter(
                filtered,
                x="ts",
                y="level",
                color="agent",
                hover_data=["message"],
                title="Audit timeline",
            )
            fig2.update_layout(template="plotly_dark", paper_bgcolor=HEX["bg"], plot_bgcolor=HEX["panel"], font=dict(color=HEX["text"]))
            st.plotly_chart(fig2, use_container_width=True)

    d1, d2 = st.columns(2)
    d1.download_button(
        "Download audit CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        "audit_trail.csv",
        "text/csv",
        key="download_audit_csv",
    )
    d2.download_button(
        "Download audit JSON",
        canonical_json(filtered.to_dict(orient="records")).encode("utf-8"),
        "audit_trail.json",
        "application/json",
        key="download_audit_json",
    )


def tab_simulation_studio(conn: sqlite3.Connection, state: Dict[str, Any], now: datetime, metrics: Dict[str, Any]) -> None:
    st.subheader("Simulation Studio")
    st.caption("Renders the app's real event stream when available. The scene mints a key, drains the countdown ring, crumbles it at zero, and slides in an audit card.")

    scenario = st.selectbox("Scenario", ["Normal day", "Active attack", "Recovery"], key="sim_scenario")
    source = st.radio("Event source", ["Real app stream if available", "Scripted demo"], horizontal=True, key="sim_source")
    speed = st.slider("Seconds per event", 0.2, 2.0, 0.6, 0.1, key="sim_speed")

    events = get_simulation_events(state, scenario, source)
    agents = get_agent_roster()
    kpis = {
        "Policy enforcement %": metrics.get("Policy enforcement %", 0),
        "Auto-approval accuracy %": metrics.get("Auto-approval accuracy %", 0),
        "Expiry correctness %": metrics.get("Expiry correctness %", 0),
        "Audit completeness %": metrics.get("Audit completeness %", 0),
    }

    st.caption(f"{len(events)} events ready.")

    b1, b2 = st.columns(2)
    if b1.button("Run live simulation", key="run_live_sim"):
        play_live_simulation(events, agents, kpis, speed)

    if b2.button("Record video", key="record_video"):
        with st.spinner("Encoding H.264 MP4 with bundled imageio-ffmpeg..."):
            try:
                hold = max(2, int(round(10 * float(speed))))
                video = export_mp4("JITKeys Simulation Studio", agents, events, lambda p: kpis, fps=10, hold=hold)
                state["sim_video"] = video
                state["sim_events"] = events
                REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                (REPORTS_DIR / "simulation.mp4").write_bytes(video)
                (REPORTS_DIR / "events.json").write_text(json.dumps(events, indent=2, default=str), encoding="utf-8")
            except Exception as exc:
                st.error(f"Video rendering failed: {exc}")

    if state.get("sim_video"):
        st.video(BytesIO(state["sim_video"]), format="video/mp4")
        st.download_button(
            "Download simulation.mp4",
            state["sim_video"],
            "simulation.mp4",
            "video/mp4",
            key="download_sim_mp4",
        )

    st.download_button(
        "Download events.json",
        json.dumps(events, indent=2, default=str).encode("utf-8"),
        "events.json",
        "application/json",
        key="download_sim_events",
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="JITKeys", page_icon=None, layout="wide", initial_sidebar_state="expanded")
    inject_theme()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    init_session_state()

    if st.session_state.pop("reset_requested", False):
        try:
            DB_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        st.session_state.clear()
        init_session_state()
        st.rerun()

    conn = get_conn()
    init_db(conn)
    seed_reference_data(conn)

    with st.sidebar:
        st.header("Control plane")
        mode = st.radio("Mode", ["Demo", "Live"], index=0 if st.session_state.get("mode") == "Demo" else 1, key="mode_radio")
        st.session_state["mode"] = mode

        dry = st.checkbox("Dry-run (simulate only)", value=bool(st.session_state.get("dry_run", True)), key="dry_run_checkbox")
        st.session_state["dry_run"] = dry

        api = st.text_input("Optional Gemini API key", type="password", value=st.session_state.get("api_key", ""), key="api_key_input")
        st.session_state["api_key"] = api.strip()

        st.divider()

        users = fetch_all(conn, "SELECT id, name FROM users ORDER BY name")
        user_ids = [u["id"] for u in users]
        user_names = {u["id"]: u["name"] for u in users}

        def fmt_user(uid: str) -> str:
            return f"{user_names.get(uid, uid)} ({uid})"

        req_idx = user_ids.index(st.session_state.get("selected_user")) if st.session_state.get("selected_user") in user_ids else 0
        req_user_id = st.selectbox("Default requester", user_ids, index=req_idx, format_func=fmt_user, key="sidebar_requester")
        st.session_state["selected_user"] = req_user_id

        app_idx = user_ids.index(st.session_state.get("approver_id")) if st.session_state.get("approver_id") in user_ids else 0
        app_user_id = st.selectbox("Default approver", user_ids, index=app_idx, format_func=fmt_user, key="sidebar_approver")
        st.session_state["approver_id"] = app_user_id

        st.divider()
        st.subheader("Simulated clock")
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("+1m", key="clock_1m"):
            st.session_state["clock_offset"] = int(st.session_state.get("clock_offset", 0)) + 60
        if c2.button("+15m", key="clock_15m"):
            st.session_state["clock_offset"] = int(st.session_state.get("clock_offset", 0)) + 900
        if c3.button("+1h", key="clock_1h"):
            st.session_state["clock_offset"] = int(st.session_state.get("clock_offset", 0)) + 3600
        if c4.button("+1d", key="clock_1d"):
            st.session_state["clock_offset"] = int(st.session_state.get("clock_offset", 0)) + 86400

        if st.button("Reset clock", key="reset_clock"):
            st.session_state["clock_offset"] = 0

        st.code(get_sim_now().strftime("%Y-%m-%d %H:%M:%S UTC"))

        if st.button("Reset demo data", key="reset_demo_data"):
            st.session_state["reset_requested"] = True
            st.rerun()

    sim_now = get_sim_now()
    ensure_demo_data(conn, st.session_state, sim_now)
    expire_due_sessions(conn, st.session_state, sim_now)
    metrics = compute_metrics(conn, sim_now)

    st.title("JITKeys")
    st.caption("Just-in-time privileged access: temporary rights that expire on their own. Local synthetic lab only.")
    kpi_row(metrics)

    if not st.session_state.get("auto_video_done"):
        demo_slot = st.empty()
        demo_slot.info("Generating live demo video from the app's own event stream...")
        try:
            events = st.session_state.get("event_log", [])[-20:]
            if len(events) < 15:
                events = script_events("Normal day")
            agents = get_agent_roster()
            kpis = {
                "Policy enforcement %": metrics.get("Policy enforcement %", 0),
                "Auto-approval accuracy %": metrics.get("Auto-approval accuracy %", 0),
                "Expiry correctness %": metrics.get("Expiry correctness %", 0),
                "Audit completeness %": metrics.get("Audit completeness %", 0),
            }
            video = export_mp4("JITKeys auto demo: mint, use, expire, audit", agents, events, lambda p: kpis, fps=10, hold=4)
            st.session_state["auto_video"] = video
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            (REPORTS_DIR / "jitkeys_auto_demo.mp4").write_bytes(video)
        except Exception as exc:
            st.session_state["auto_video_error"] = str(exc)
        st.session_state["auto_video_done"] = True
        demo_slot.empty()

    if st.session_state.get("auto_video"):
        with st.expander("Auto-generated live demo video", expanded=True):
            st.video(BytesIO(st.session_state["auto_video"]), format="video/mp4")
            st.download_button(
                "Download auto demo MP4",
                st.session_state["auto_video"],
                "jitkeys_auto_demo.mp4",
                "video/mp4",
                key="download_auto_demo_mp4",
            )
    elif st.session_state.get("auto_video_error"):
        st.warning(f"Auto video skipped: {st.session_state['auto_video_error']}")

    tabs = st.tabs(["Request Console", "Approval Inbox", "Active Sessions", "Audit Trail", "Simulation Studio"])

    with tabs[0]:
        tab_request_console(conn, st.session_state, sim_now)
    with tabs[1]:
        tab_approval_inbox(conn, st.session_state, sim_now)
    with tabs[2]:
        tab_active_sessions(conn, st.session_state, sim_now)
    with tabs[3]:
        tab_audit_trail(conn, st.session_state, sim_now)
    with tabs[4]:
        tab_simulation_studio(conn, st.session_state, sim_now, metrics)

    conn.close()


# ---------------------------------------------------------------------------
# Built-in pytest-compatible tests
# ---------------------------------------------------------------------------

def _memory_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    seed_reference_data(conn)
    return conn


def test_interpreter_parses_standard_request() -> None:
    conn = _memory_conn()
    now = datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc)
    ctx = {"conn": conn, "requester_id": "u-alice", "now": now}
    result = interpreter_demo(DEFAULT_REQUEST, ctx)
    assert result["resource"] == "prod-orders"
    assert result["role"] == "db-admin"
    assert result["duration_minutes"] == 45
    assert result["ticket"] == "CHG-2231"
    assert "migration" in result["justification"].lower()


def test_risk_scores_high_for_prod_admin() -> None:
    conn = _memory_conn()
    now = datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc)
    parsed = interpreter_demo(DEFAULT_REQUEST, {"conn": conn, "requester_id": "u-alice", "now": now})
    risk = risk_assessor_demo("", {"conn": conn, "parsed": parsed, "requester_id": "u-alice", "now": now})
    assert risk["risk_score"] >= 50
    assert risk["risk_level"] in ("high", "critical")
    assert risk["ticket_valid"] is True


def test_router_auto_approves_only_low_risk() -> None:
    conn = _memory_conn()

    low_parsed = {"resource": "dev-sandbox", "role": "readonly", "duration_minutes": 15, "justification": "check sample", "ticket": "CHG-2240"}
    low_risk = {"risk_score": 18, "risk_level": "low", "ticket_valid": True}
    low_router = approver_router_demo("", {"conn": conn, "parsed": low_parsed, "risk": low_risk})
    assert low_router["auto_approve"] is True

    high_parsed = {"resource": "prod-orders", "role": "db-admin", "duration_minutes": 45, "justification": "fix migration", "ticket": "CHG-2231"}
    high_risk = {"risk_score": 72, "risk_level": "high", "ticket_valid": True}
    high_router = approver_router_demo("", {"conn": conn, "parsed": high_parsed, "risk": high_risk})
    assert high_router["auto_approve"] is False
    assert "u-sec-lead" in high_router["approvers"]


def test_approval_gate_blocks_unapproved_action() -> None:
    action = {"type": "issue_mock_credential", "requires_human": True}
    pending = approval_gate(action, approved=False, dry_run=True)
    assert pending["status"] == "pending_approval"

    dry = approval_gate(action, approved=True, dry_run=True)
    assert dry["status"] == "dry_run"


def test_expiry_correctness_for_active_session() -> None:
    conn = _memory_conn()
    state: Dict[str, Any] = {"event_log": [], "last_audit_hash": None}
    now = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)

    _insert_request_direct(
        conn,
        "REQ-TEST-EXPIRE",
        "u-alice",
        "dev-sandbox",
        "readonly",
        10,
        "test expiry",
        "CHG-2240",
        "approved",
        10,
        "low",
        True,
        "approve",
        True,
        now - timedelta(minutes=30),
        "test",
        {},
    )
    conn.execute(
        """
        INSERT INTO approvals (request_id, approver_id, decision, reason, ts, gate_status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("REQ-TEST-EXPIRE", "policy-auto", "approved", "test", dt_iso(now - timedelta(minutes=30)), "dry_run"),
    )
    conn.execute(
        """
        INSERT INTO sessions
        (id, request_id, user_id, resource_id, role_id, mock_credential, issued_at, expires_at, status, audit_summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', NULL)
        """,
        (
            "SES-TEST-EXPIRE",
            "REQ-TEST-EXPIRE",
            "u-alice",
            "dev-sandbox",
            "readonly",
            "MOCK-TEST",
            dt_iso(now - timedelta(minutes=30)),
            dt_iso(now - timedelta(minutes=20)),
        ),
    )
    conn.commit()

    expired = expire_due_sessions(conn, state, now)
    assert expired == 1
    row = fetch_one(conn, "SELECT status, ended_at FROM sessions WHERE id = ?", ("SES-TEST-EXPIRE",))
    assert row["status"] == "expired"
    assert row["ended_at"]


def test_audit_hash_chain_detects_tampering() -> None:
    conn = _memory_conn()
    state: Dict[str, Any] = {"event_log": [], "last_audit_hash": None}
    now = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)

    emit_event(state, conn, "Test Agent", "info", "first", {}, now)
    emit_event(state, conn, "Test Agent", "ok", "second", {}, now)

    valid, total, ok = verify_audit_chain(conn)
    assert ok is True
    assert total == 2

    conn.execute("UPDATE audit_events SET message = 'tampered' WHERE id = (SELECT MIN(id) FROM audit_events)")
    conn.commit()

    valid, total, ok = verify_audit_chain(conn)
    assert ok is False
    assert valid < total


def test_invalid_ticket_is_flagged_by_critic() -> None:
    parsed = {"resource": "stage-analytics", "role": "readonly", "duration_minutes": 60, "justification": "inspect ETL", "ticket": "CHG-2232"}
    risk = {"risk_score": 42, "risk_level": "medium", "ticket_valid": False, "factors": ["Ticket draft"]}
    router = {"approvers": ["u-owner-data"], "auto_approve": False, "policy_reason": "medium"}
    critic = critic_skeptic_demo("", {"parsed": parsed, "risk": risk, "router": router})
    assert critic["verdict"] in ("revise", "reject")
    assert any("ticket" in c.lower() for c in critic["concerns"])


def test_risky_command_detection() -> None:
    flagged, severity = detect_command_risk("DROP TABLE orders;")
    assert flagged is True
    assert severity == "crit"

    flagged, severity = detect_command_risk("DELETE FROM orders;")
    assert flagged is True
    assert severity == "crit"

    flagged, severity = detect_command_risk("SELECT * FROM orders WHERE id = 1;")
    assert flagged is False


if __name__ == "__main__":
    main()