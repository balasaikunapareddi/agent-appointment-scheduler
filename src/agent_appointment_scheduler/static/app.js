// Appointment Scheduler UI: chat + session picker + booking card.
// Talks to the same FastAPI app over POST /chat and GET /sessions/*.
// No build step, no framework — plain fetch.
"use strict";

let currentSessionId = null;

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

function newSessionId() {
  return "SES-" + Math.random().toString(16).slice(2, 10).toUpperCase();
}

function renderMessages(messages) {
  const log = $("chat-log");
  log.innerHTML = "";
  for (const m of messages || []) {
    const div = document.createElement("div");
    div.className = "msg " + (m.role === "user" ? "user" : "agent");
    div.innerHTML = `<span class="who">${esc(m.role)}</span>${esc(m.content)}`;
    log.appendChild(div);
  }
  log.scrollTop = log.scrollHeight;
}

function renderBookingCard(state) {
  const card = $("booking-card");
  if (!state || !state.booking_state) {
    card.innerHTML = '<p class="hint">No booking in progress yet.</p>';
    return;
  }
  const b = state.booking_state;
  const rows = [
    ["Stage", b.stage || "—"],
    ["Provider", b.provider_name || "—"],
    ["Slot", b.slot || "—"],
    ["Patient ref", b.patient_ref || "—"],
    ["Phone", b.phone_masked || "—"],
    ["Reason", b.reason || "—"],
    ["Contact", b.contact_method || "—"],
    ["Appointment", b.appointment_id || "—"],
  ];
  const filled = ["patient_ref", "provider", "slot", "reason", "contact_method"]
    .map((k) => {
      const ok = k === "provider" ? !!b.provider_npi : k === "slot" ? !!b.slot : !!b[k];
      return `<span class="${ok ? "" : "missing"}">${ok ? "✓" : "○"} ${esc(k)}</span>`;
    }).join(" · ");
  card.innerHTML = rows.map(([k, v]) =>
    `<div class="field"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`
  ).join("") + `<div style="margin-top:8px;font-size:13px">${filled}</div>` +
    (state.readable_summary || b.slot ? `<div id="booking-summary">${esc(state.readable_summary || "")}</div>` : "");
}

function renderModelStatus(reply) {
  const cache = reply.cache || {};
  $("m-prompt").textContent = cache.prompt_tokens ?? "—";
  $("m-cached").textContent = cache.cached_tokens ?? "—";
  $("m-hit").textContent = cache.cache_hit_rate != null ? `${Math.round(cache.cache_hit_rate * 100)}%` : "—";
  $("m-saved").textContent = cache.est_cost_saved_usd != null ? `$${cache.est_cost_saved_usd}` : "—";
  $("m-skills").textContent = (reply.skills_activated || []).join(", ") || "—";
}

async function loadSessions() {
  const sessions = await api("/sessions");
  const list = $("sessions");
  list.innerHTML = "";
  for (const s of sessions) {
    const btn = document.createElement("button");
    if (s.session_id === currentSessionId) btn.className = "active";
    const when = s.updated_at ? new Date(s.updated_at * 1000).toLocaleString() : "";
    btn.innerHTML = `${esc(s.session_id)}<span class="meta">${esc(s.label || "")} ${esc(when)}</span>`;
    btn.onclick = () => selectSession(s.session_id);
    list.appendChild(btn);
  }
}

async function selectSession(sessionId) {
  currentSessionId = sessionId;
  const state = await api(`/sessions/${encodeURIComponent(sessionId)}`);
  renderMessages(state.messages);
  renderBookingCard(state);
  loadSessions();
}

async function sendMessage(text) {
  const reply = await api("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: currentSessionId, message: text }),
  });
  currentSessionId = reply.session_id;
  renderModelStatus(reply);
  const state = await api(`/sessions/${encodeURIComponent(currentSessionId)}`);
  renderMessages(state.messages);
  renderBookingCard(state);
  loadSessions();
}

$("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("chat-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  input.disabled = true;
  try {
    await sendMessage(text);
  } catch (err) {
    const log = $("chat-log");
    const div = document.createElement("div");
    div.className = "msg agent";
    div.textContent = "Something went wrong talking to the agent: " + err.message;
    log.appendChild(div);
  } finally {
    input.disabled = false;
    input.focus();
  }
});

$("new-session").addEventListener("click", () => {
  currentSessionId = newSessionId();
  renderMessages([]);
  renderBookingCard(null);
  loadSessions();
  $("chat-input").focus();
});

loadSessions();
