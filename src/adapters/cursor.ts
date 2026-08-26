import fs from "node:fs";
import path from "node:path";
import Database from "better-sqlite3";
import type { NormalizedMessage, NormalizedSession, WatermarkCursorValue, WatermarkSourceAdapter } from "../types.js";

// Cursor persists conversations in a global SQLite db (all projects mixed
// together) plus one per-workspace SQLite db per project. Mapping verified
// against a real, in-use Cursor install — see
// docs-internal/specs/2026-08-26-cursor-adapter-design.md for the full
// investigation (74,691 real bubble rows, static analysis of the packaged
// app bundle to find the actual tool-call field name, etc.):
//
//   User/globalStorage/state.vscdb
//     cursorDiskKV['composerData:<id>']         -> session (name, bubble list)
//     cursorDiskKV['bubbleId:<composerId>:<id>'] -> message (type/text/tools)
//     composerHeaders(composerId, recency, lastUpdatedAt, value) -> newer,
//       faster session index; value JSON has workspaceIdentifier.uri.fsPath
//       (the project path), but only covers composers touched since this
//       table was introduced (~48% of composers on the reference install) —
//       never backfilled for older ones. Watermark on `recency`, not
//       `lastUpdatedAt`: the latter is NULL on ~19% of rows on the reference
//       install (confirmed — chat-mode/single-exchange composers, it looks
//       like), which silently drops them from a `WHERE lastUpdatedAt >= ?`
//       scan forever (NULL >= x is NULL, never true in SQL). `recency` is
//       never NULL and equals `lastUpdatedAt` in every row where both are
//       present — it's the column composerHeaders' own covering index
//       (idx_composerHeaders_1) is built on, so it's the intended field.
//   User/workspaceStorage/<hash>/
//     workspace.json                 -> {"folder": "file:///abs/path"}
//     state.vscdb ItemTable['composer.composerData'].allComposers[].composerId
//       -> which composers belong to this project (the fallback path for
//          composers composerHeaders doesn't cover)
//
// Tool calls: message.toolFormerData (NOT the public-docs-cited
// `toolResults`, which is empty on every real bubble sampled). Its `name`
// field is a human-readable tool name (used as-is for NormalizedMessage.tools,
// same as every other adapter — no cross-tool name normalization, even
// though Cursor itself has at least two different MCP tool-name spellings
// across versions). `result`'s shape varies by tool; some (edit_file_v2 and
// friends) don't inline the diff at all, just a content-hash reference into
// a separate composer.content.<hash> blob cache — deliberately not resolved
// here (see extractToolText below), same "defer the expensive low-value
// join" call the OpenCode adapter already made for sub-sessions.

interface ComposerHeaderRow {
  composerId: string;
  recency: number;
  value: string;
}

interface ConversationHeader {
  bubbleId: string;
  type: number; // 1 = user, 2 = assistant
}

interface ComposerDataValue {
  name?: string;
  createdAt?: number;
  fullConversationHeadersOnly?: ConversationHeader[];
}

interface BubbleValue {
  type?: number;
  text?: string;
  createdAt?: number;
  toolFormerData?: {
    name?: string;
    rawArgs?: string;
    result?: string;
  };
}

// edit-type tool results offload their actual before/after content to a
// composer.content.<hash> blob and leave only this reference behind — a
// value matching this pattern carries no searchable text of its own.
const CONTENT_HASH_REF_RE = /^composer\.content\.[0-9a-f]+$/;

// Tool call params/results have no fixed schema across ~70 distinct tool
// names (confirmed on the reference corpus) and drift between Cursor
// versions (edit_file -> edit_file_v2, read_file -> read_file_v2, ...), so
// rather than a per-tool-name parser this walks whatever JSON shape shows
// up and collects prose, skipping the one identified noise pattern above.
// Bounded in both depth and total size so one huge read_file/grep dump
// doesn't dominate a message's indexed text.
const TOOL_TEXT_DEPTH_LIMIT = 8;
const TOOL_TEXT_CHAR_BUDGET = 6000;

function collectStrings(value: unknown, depth: number, budget: { remaining: number }, out: string[]): void {
  if (budget.remaining <= 0 || depth > TOOL_TEXT_DEPTH_LIMIT) return;
  if (typeof value === "string") {
    if (value && !CONTENT_HASH_REF_RE.test(value)) {
      const take = value.slice(0, budget.remaining);
      out.push(take);
      budget.remaining -= take.length;
    }
    return;
  }
  if (Array.isArray(value)) {
    for (const v of value) {
      if (budget.remaining <= 0) break;
      collectStrings(v, depth + 1, budget, out);
    }
    return;
  }
  if (value && typeof value === "object") {
    for (const v of Object.values(value as Record<string, unknown>)) {
      if (budget.remaining <= 0) break;
      collectStrings(v, depth + 1, budget, out);
    }
  }
}

function extractToolText(rawArgsJson: string | undefined, resultJson: string | undefined): string | undefined {
  const parts: string[] = [];
  const budget = { remaining: TOOL_TEXT_CHAR_BUDGET };
  for (const raw of [rawArgsJson, resultJson]) {
    if (!raw) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      parsed = raw; // not JSON — index the raw string itself
    }
    collectStrings(parsed, 0, budget, parts);
  }
  return parts.length > 0 ? parts.join("\n\n") : undefined;
}

// 32% of bubbles on the reference install have no createdAt of their own
// (confirmed by sampling — not rare). Falling back to 0 (1970-01-01) would
// be actively misleading, so this falls back to the composer's own
// createdAt instead — still approximate, but in the right era. Array.sort
// is stable, so messages sharing a fallback timestamp keep the order they
// were pushed in below, which already follows fullConversationHeadersOnly
// (true conversation order) — a missing timestamp doesn't scramble order.
function parseBubble(raw: string, bubbleId: string, composerCreatedAt: number | undefined): NormalizedMessage | undefined {
  let d: BubbleValue;
  try {
    d = JSON.parse(raw);
  } catch {
    return undefined;
  }
  const role = d.type === 1 ? "user" : d.type === 2 ? "assistant" : undefined;
  if (!role) return undefined;

  const toolFormerData = d.toolFormerData;
  const tools = toolFormerData?.name ? [toolFormerData.name] : [];
  const toolText = extractToolText(toolFormerData?.rawArgs, toolFormerData?.result);

  return {
    uuid: bubbleId,
    role,
    ts: new Date(d.createdAt ?? composerCreatedAt ?? 0).toISOString(),
    text: d.text ?? "",
    toolText,
    tools,
    isSidechain: false,
  };
}

function fileUriToPath(uri: string): string | undefined {
  if (!uri.startsWith("file://")) return undefined;
  try {
    return decodeURIComponent(uri.slice("file://".length));
  } catch {
    return uri.slice("file://".length);
  }
}

// composerHeaders.value is JSON with workspaceIdentifier.uri.fsPath — the
// fast, authoritative project-dir signal, when a header row exists at all.
function projectDirFromHeaderValue(headerValueJson: string): string | undefined {
  try {
    const v = JSON.parse(headerValueJson) as { workspaceIdentifier?: { uri?: { fsPath?: unknown } } };
    const fsPath = v.workspaceIdentifier?.uri?.fsPath;
    return typeof fsPath === "string" && fsPath ? fsPath : undefined;
  } catch {
    return undefined;
  }
}

// Fallback for composers composerHeaders doesn't cover: join every
// workspaceStorage/<hash>/workspace.json (folder path) against that same
// workspace's own state.vscdb, which lists which composerIds belong to it.
// Expensive (opens every per-workspace db) — only called when there's at
// least one composer that actually needs it (see parseSince below), so it's
// a one-time cost on first index and rarely touched again afterward.
function buildLegacyProjectDirMap(workspaceStorageDir: string): Map<string, string> {
  const map = new Map<string, string>();
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(workspaceStorageDir, { withFileTypes: true });
  } catch {
    return map;
  }
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const wsDir = path.join(workspaceStorageDir, entry.name);
    let folder: string | undefined;
    try {
      const wsJson = JSON.parse(fs.readFileSync(path.join(wsDir, "workspace.json"), "utf8")) as { folder?: unknown };
      folder = typeof wsJson.folder === "string" ? fileUriToPath(wsJson.folder) : undefined;
    } catch {
      continue; // no workspace.json (e.g. an empty/orphaned storage dir) — skip
    }
    if (!folder) continue;

    let wsDb: Database.Database;
    try {
      wsDb = new Database(path.join(wsDir, "state.vscdb"), { readonly: true, fileMustExist: true, timeout: 2000 });
    } catch {
      continue;
    }
    try {
      const row = wsDb.prepare("SELECT value FROM ItemTable WHERE key = 'composer.composerData'").get() as
        | { value: string }
        | undefined;
      if (!row) continue;
      const parsed = JSON.parse(row.value) as { allComposers?: unknown };
      const composers = Array.isArray(parsed.allComposers) ? parsed.allComposers : [];
      for (const c of composers) {
        const composerId = (c as { composerId?: unknown } | null)?.composerId;
        if (typeof composerId === "string" && !map.has(composerId)) {
          map.set(composerId, folder);
        }
      }
    } catch {
      // malformed/locked per-workspace db — that one workspace's composers
      // stay unresolved this run, doesn't abort the others.
    } finally {
      wsDb.close();
    }
  }
  return map;
}

function tableExists(db: Database.Database, name: string): boolean {
  const row = db.prepare("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?").get(name);
  return row !== undefined;
}

export class CursorAdapter implements WatermarkSourceAdapter {
  id = "cursor";
  cursorKind = "watermark" as const;

  discover(roots: string[]): string[] {
    const found: string[] = [];
    for (const root of roots) {
      // Cursor's layout is fixed (unlike OpenCode's, which can nest an
      // opencode.db arbitrarily) — root is either the "Cursor" app-support
      // dir or, as DEFAULT_CURSOR_ROOTS points at, the "User" dir directly.
      const candidates = [
        path.join(root, "User", "globalStorage", "state.vscdb"),
        path.join(root, "globalStorage", "state.vscdb"),
      ];
      for (const candidate of candidates) {
        try {
          if (fs.statSync(candidate).isFile()) {
            found.push(candidate);
            break;
          }
        } catch {
          // try the next candidate
        }
      }
    }
    return found;
  }

  parseSince(dbPath: string, cursor?: WatermarkCursorValue): { sessions: NormalizedSession[]; cursor: WatermarkCursorValue } {
    const persistedWatermark = cursor?.value ?? 0;
    // Two independent id namespaces packed into one tieBreakIds array: "hdr:"
    // entries are composerHeaders rows tied at the current watermark
    // boundary (same tie-break need as OpenCode's — two composers CAN share
    // a recency millisecond); "legacy:" entries are every headerless
    // composer ever indexed, tracked by pure id membership since
    // cursorDiskKV has no per-row timestamp column to watermark against at
    // all. A legacy composer is treated as static once indexed — the one
    // documented gap is a previously-empty draft that gets a real message
    // without ever gaining a composerHeaders row, which would go unnoticed
    // until a full reindex. Reference data suggests header rows appear on
    // real activity, so this should be rare in practice.
    const prevHdrTies = new Set<string>();
    const prevLegacySeen = new Set<string>();
    for (const id of cursor?.tieBreakIds ?? []) {
      if (id.startsWith("hdr:")) prevHdrTies.add(id.slice(4));
      else if (id.startsWith("legacy:")) prevLegacySeen.add(id.slice(7));
    }

    const db = new Database(dbPath, { readonly: true, fileMustExist: true, timeout: 5000 });
    try {
      const hasHeaders = tableExists(db, "composerHeaders");

      const overallMax = hasHeaders
        ? ((db.prepare("SELECT MAX(recency) as m FROM composerHeaders").get() as { m: number | null }).m ?? 0)
        : 0;
      const rolledBack = overallMax < persistedWatermark;
      const watermark = rolledBack ? 0 : persistedWatermark;
      const hdrTies = rolledBack ? new Set<string>() : prevHdrTies;

      const headerRows = hasHeaders
        ? (db
            .prepare("SELECT composerId, recency, value FROM composerHeaders WHERE recency >= ?")
            .all(watermark) as ComposerHeaderRow[])
        : [];

      let nextWatermark = watermark;
      for (const r of headerRows) if (r.recency > nextWatermark) nextWatermark = r.recency;
      const nextHdrTies = new Set(headerRows.filter((r) => r.recency === nextWatermark).map((r) => r.composerId));

      const isNewHeaderRow = (r: ComposerHeaderRow) =>
        r.recency > watermark || (r.recency === watermark && !hdrTies.has(r.composerId));
      const touchedHeaderedIds = new Set(headerRows.filter(isNewHeaderRow).map((r) => r.composerId));

      const headeredIds = hasHeaders
        ? new Set((db.prepare("SELECT composerId FROM composerHeaders").all() as { composerId: string }[]).map((r) => r.composerId))
        : new Set<string>();

      const allComposerKeys = db.prepare("SELECT key FROM cursorDiskKV WHERE key LIKE 'composerData:%'").all() as {
        key: string;
      }[];
      const newLegacyIds: string[] = [];
      for (const { key } of allComposerKeys) {
        const id = key.slice("composerData:".length);
        if (headeredIds.has(id)) continue; // covered by the headered path above
        if (prevLegacySeen.has(id)) continue; // already indexed once, treated as static
        newLegacyIds.push(id);
      }
      const nextLegacySeen = new Set(prevLegacySeen);
      for (const id of newLegacyIds) nextLegacySeen.add(id);

      const nextCursor: WatermarkCursorValue = {
        value: nextWatermark,
        tieBreakIds: [...[...nextHdrTies].map((id) => `hdr:${id}`), ...[...nextLegacySeen].map((id) => `legacy:${id}`)],
      };

      const touchedIds = new Set<string>([...touchedHeaderedIds, ...newLegacyIds]);
      if (touchedIds.size === 0) {
        return { sessions: [], cursor: nextCursor };
      }

      const headerValueById = new Map(headerRows.map((r) => [r.composerId, r.value]));
      const workspaceStorageDir = path.join(path.dirname(path.dirname(dbPath)), "workspaceStorage");
      let legacyProjectDirMap: Map<string, string> | undefined;

      const getComposerData = db.prepare("SELECT value FROM cursorDiskKV WHERE key = ?");
      const getBubble = db.prepare("SELECT value FROM cursorDiskKV WHERE key = ?");

      const sessions: NormalizedSession[] = [];
      for (const composerId of touchedIds) {
        const composerRow = getComposerData.get(`composerData:${composerId}`) as { value: string } | undefined;
        if (!composerRow) continue; // deleted between discover and parse

        let composer: ComposerDataValue;
        try {
          composer = JSON.parse(composerRow.value);
        } catch {
          continue;
        }

        let projectDir: string | undefined;
        let projectDirSource: "cwd" | "fallback" | undefined;
        const headerValue = headerValueById.get(composerId);
        if (headerValue) {
          projectDir = projectDirFromHeaderValue(headerValue);
          if (projectDir) projectDirSource = "cwd";
        }
        if (!projectDir) {
          if (!legacyProjectDirMap) legacyProjectDirMap = buildLegacyProjectDirMap(workspaceStorageDir);
          const fallbackDir = legacyProjectDirMap.get(composerId);
          if (fallbackDir) {
            projectDir = fallbackDir;
            projectDirSource = "fallback";
          }
        }
        // No directory-name-encodes-path convention here (composerId is a
        // bare UUID, unlike Claude Code's dash-decoded transcript dirnames),
        // so an unresolved project dir has no further fallback — skip this
        // composer rather than index it under a guessed/empty path.
        if (!projectDir) continue;

        const headers = composer.fullConversationHeadersOnly ?? [];
        const messages: NormalizedMessage[] = [];
        let parseErrors = 0;
        for (const h of headers) {
          const bubbleRow = getBubble.get(`bubbleId:${composerId}:${h.bubbleId}`) as { value: string } | undefined;
          if (!bubbleRow) continue; // pruned/expired bubble — not an error
          const msg = parseBubble(bubbleRow.value, h.bubbleId, composer.createdAt);
          if (!msg) {
            parseErrors++;
            continue;
          }
          messages.push(msg);
        }
        messages.sort((a, b) => a.ts.localeCompare(b.ts));

        sessions.push({
          id: composerId,
          source: "cursor",
          projectDir,
          projectDirSource,
          filePath: dbPath,
          title: composer.name,
          startedAt: messages[0]?.ts,
          endedAt: messages[messages.length - 1]?.ts,
          messages,
          parseErrors,
          bytesConsumed: 0, // unused: watermark-cursor sources resume via parseSince's returned cursor
        });
      }

      return { sessions, cursor: nextCursor };
    } finally {
      db.close();
    }
  }
}
