// Drives the test-mode app in headless Chrome through the guide's steps, checks each one, and saves screenshots.
// node guide_walkthrough.mjs <app url> <out dir> <test pack dir>   (make_guide.py explains the order)
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync, readFileSync } from "node:fs";
import { join } from "node:path";

const APP = process.argv[2] || "http://127.0.0.1:8010";
const OUT = process.argv[3] || "shots";
const PACK = process.argv[4];                       // test pack folder (pictures/)
mkdirSync(OUT, { recursive: true });
const CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const PORT = 9333;
const profile = join(OUT, "..", "chrome_profile");
const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  "--no-first-run", "--disable-gpu", "--hide-scrollbars", "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = [];
const check = (name, ok, detail = "") => { log.push(`${ok ? "PASS" : "FAIL"}  ${name}${ok ? "" : "  " + detail}`); console.log(log.at(-1)); };

let ws, id = 0;
const pending = new Map();
async function connect() {
  for (let i = 0; i < 50; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const page = list.find((t) => t.type === "page");
      if (page) { ws = new WebSocket(page.webSocketDebuggerUrl); break; }
    } catch { /* not up yet */ }
    await sleep(300);
  }
  await new Promise((r) => ws.addEventListener("open", r, { once: true }));
  ws.addEventListener("message", (e) => {
    const m = JSON.parse(e.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  });
}
const send = (method, params = {}) => new Promise((res, rej) => {
  const n = ++id;
  pending.set(n, (m) => (m.error ? rej(new Error(`${method}: ${m.error.message}`)) : res(m.result)));
  ws.send(JSON.stringify({ id: n, method, params }));
});
const js = async (expr) => {
  const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || "js error");
  return r.result.value;
};
const W = 1360;
async function view(h = 860) { await send("Emulation.setDeviceMetricsOverride", { width: W, height: h, deviceScaleFactor: 2, mobile: false }); }
async function go(hash) { await send("Page.navigate", { url: `${APP}/${hash}` }); await sleep(300); await send("Page.reload"); await sleep(1800); }
async function shot(name, { full = false, clip = null } = {}) {
  let h = 860;
  if (full) { h = Math.min(4000, Math.ceil(await js("document.documentElement.scrollHeight"))); await view(h); await sleep(400); }
  const r = await send("Page.captureScreenshot", { format: "png", ...(clip ? { clip: { ...clip, scale: 1 } } : {}) });
  writeFileSync(join(OUT, `${name}.png`), Buffer.from(r.data, "base64"));
  if (full) await view();
  console.log("shot", name);
}
// React-controlled inputs need the native setter + an input event
const typeInto = (sel, text, index = 0) => js(`(() => { const el = document.querySelectorAll(${JSON.stringify(sel)})[${index}];
  const proto = el.tagName === "TEXTAREA" ? HTMLTextAreaElement.prototype : el.tagName === "SELECT" ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto, "value").set.call(el, ${JSON.stringify(text)});
  el.dispatchEvent(new Event(el.tagName === "SELECT" ? "change" : "input", { bubbles: true })); return true; })()`);
const clickText = (sel, text) => js(`(() => { const b = [...document.querySelectorAll(${JSON.stringify(sel)})].find((x) => x.textContent.includes(${JSON.stringify(text)}));
  if (!b) return false; b.click(); return true; })()`);
const api = async (path, opts) => (await fetch(APP + path, opts)).json();
async function waitDone(pid) {
  for (let i = 0; i < 240; i++) { const p = await api(`/api/patients/${pid}`); if (p.status !== "working" && !(p.pending || []).length) return p; await sleep(500); }
  throw new Error("AI did not finish");
}
async function newPatient(text, pics = []) {
  const fd = new FormData(); fd.append("text", text); fd.append("who", "Nurse Tan");
  for (const n of pics) fd.append("files", new Blob([readFileSync(join(PACK, "pictures", n))]), n);
  const r = await api("/api/patients", { method: "POST", body: fd });
  return waitDone(r.id);
}

await connect();
await send("Page.enable"); await send("Runtime.enable");
await view();
await go("");
await js(`localStorage.setItem("ed-me", JSON.stringify({ who: "Nurse Tan", role: "Triage nurse" })); true`);

// a few patients already in the ED, so the whiteboard looks like a shift
const cases = Object.fromEntries(readFileSync(join(PACK, "expected_results.csv"), "utf8").replace(/^\uFEFF/, "").split(/\r?\n/).slice(1)
  .filter(Boolean).map((l) => [l.split(",")[0], l]));
const note = (c) => readFileSync(join(PACK, "cases", `${c}.txt`), "utf8").trim();
const bg = {};
for (const c of ["T03", "T04", "T06", "R02", "R05"]) bg[c] = await newPatient(note(c));
bg.P06 = await newPatient(note("P06"), ["cxr_pneumothorax.jpg"]);
bg.P01 = await newPatient(note("P01"), ["adult_wrist_fracture.jpg"]);
bg.P11 = await newPatient(note("P11"), ["ct_head_bleed_epidural.jpg"]);
check("background patients created", Object.values(bg).every((p) => p.decision));

// A. charting screen
await go("#/");
await shot("01_charting_empty");
await typeInto("textarea.note-box", note("T01"));
await sleep(300);
await shot("02_charting_typed");
// stop the new window that Submit opens: headless can't show it, and the case file is opened directly below
await js(`window.open = () => ({ close() {}, set location(v) { window.__opened = v; } }); true`);
await clickText("button", "Submit");
await sleep(1500);
const opened = await js("window.__opened || ''");
check("Submit opens the case file", opened.includes("#/case/"), opened);
const pid = opened.split("#/case/")[1];
await waitDone(pid);
await sleep(4500);                                    // the charting screen polls every few seconds
await shot("03_charting_questions");
check("charting screen shows the patient, case-file link and answer box", await js(`document.body.textContent.includes("AI questions for") && !!document.querySelector('input[placeholder="Answer or add information"]')`));

// B. case file
await go(`#/case/${pid}`);
await sleep(1000);
await shot("04_case_file", { full: true });
const tiles = await js(`[...document.querySelectorAll(".vital .v")].map((x) => x.textContent).join("|")`);
check("tiles show the numbers", tiles.includes("112") && tiles.includes("90/60"), tiles);
check("urgent flag shown", await js(`document.body.textContent.includes("Urgent: flagged")`));
// add information
await typeInto('input[aria-label="Add information"]', "Trop 45 ng/L. ECG ST depression V4-V6. Aspirin 300mg given, cardiology called");
await clickText("button.primary", "Add");
await sleep(1000); await waitDone(pid); await sleep(2500);
await shot("05_case_added", { full: true });
check("added note filed into the case file", await js(`document.querySelector(".report").textContent.includes("Trop 45")`));
// fix numbers
await clickText("button", "Fix numbers");
await sleep(500);
await shot("06_fix_numbers");
const keys = await js(`[...document.querySelectorAll(".vital .k")].map((x) => x.textContent)`);
const sbpIndex = keys.findIndex((k) => k.startsWith("BP sys"));
await typeInto(".vital input", "82", sbpIndex);
await clickText("button.primary", "Save and re-score");
await sleep(2500);
await shot("07_after_fix", { clip: { x: 0, y: 0, width: W, height: 330 } });
const pf = await api(`/api/patients/${pid}`);
check("fixed number re-scored and logged", pf.fields.sbp === 82 && pf.timeline.some((e) => /Numbers corrected/.test(e.text)));
// chat
await clickText(".tabs button", "Chat");
await typeInto('input[placeholder="What else should we rule out?"]', "What else should we rule out?");
await clickText("button", "Ask");
await sleep(2500);
await shot("08_chat", { full: true });
check("chat answered", (await api(`/api/patients/${pid}`)).chat.length >= 2);
// timeline
await clickText(".tabs button", "Timeline");
await sleep(500);
await shot("09_timeline", { full: true });

// C. pictures
for (const [c, name] of [["P01", "10_pictures_fracture"], ["P06", "11_pictures_chest"], ["P11", "12_pictures_head_ct"]]) {
  await go(`#/case/${bg[c].id}`);
  await sleep(800);
  await clickText(".tabs button", "Files");
  await sleep(1500);
  await shot(name, { full: true });
  check(`${c}: picture, marks and model results shown`, await js(`document.querySelectorAll(".file img").length >= 2 && document.querySelectorAll(".checklist .check").length > 0`));
}

// D. whiteboard
await go("#/board");
await sleep(2500);
await shot("13_board", { full: true });
check("board lists every patient", await js(`document.querySelectorAll("table.board tbody tr").length`) === 9);
// move T04 (finger cut) to Discharged home -> outcome boxes -> clear
const t4 = bg.T04.id;
await api(`/api/patients/${t4}/board`, { method: "PATCH", headers: { "content-type": "application/json" },
  body: JSON.stringify({ who: "Nurse Tan", role: "Triage nurse", stage: "Discharged home" }) });
await sleep(3500);
const row = await js(`(() => { const rows = [...document.querySelectorAll("table.board tbody tr")];
  const i = rows.findIndex((r) => r.querySelector('input[aria-label="Final diagnosis"]')); return i; })()`);
check("leaving patient shows outcome boxes", row >= 0);
await typeInto('input[aria-label="Final diagnosis"]', "Simple finger laceration, steri-strips");
await typeInto('select[aria-label="Level that was right"]', "4");
await sleep(400);
const rect = await js(`(() => { const r = document.querySelector('input[aria-label="Final diagnosis"]').closest("tr").getBoundingClientRect();
  return { x: 0, y: Math.max(0, r.top - 60), width: ${W}, height: r.height + 90 }; })()`);
await shot("14_board_outcome", { clip: rect });
await clickText("button", "Clear from board");
await sleep(2000);
const cleared = await api(`/api/patients/${t4}`);
check("cleared with outcome logged", cleared.cleared && cleared.timeline.some((e) => /Outcome/.test(e.what + e.text)));
// view-only role
await js(`localStorage.setItem("ed-me", JSON.stringify({ who: "Dr Wong", role: "Doctor" })); true`);
await go("#/board");
await sleep(2500);
await shot("15_board_doctor", { clip: { x: 0, y: 0, width: W, height: 520 } });
check("doctor sees the board read-only", await js(`document.querySelectorAll("table.board select").length === 0`));

// E. the AI's questions (stroke case), and answering one from the charting screen
await js(`localStorage.setItem("ed-me", JSON.stringify({ who: "Nurse Tan", role: "Triage nurse" })); true`);
await go("#/");
await js(`window.open = () => ({ close() {}, set location(v) { window.__opened = v; } }); true`);
await typeInto("textarea.note-box", note("T02"));
await clickText("button", "Submit");
await sleep(1500);
const pid2 = (await js("window.__opened || ''")).split("#/case/")[1];
await waitDone(pid2);
await sleep(4500);
check("stroke case: questions shown on the charting screen", await js(`document.querySelectorAll(".q").length > 0`));
await js(`document.querySelector(".q").click(); true`);
await sleep(300);
const filled = await js(`document.querySelector('input[placeholder="Answer or add information"]').value`);
check("clicking a question puts it in the answer box", filled.includes("→"), filled);
await typeInto('input[placeholder="Answer or add information"]', filled + " 0830 today, seen by husband");
await shot("03b_charting_stroke_questions");
await clickText("button", "Add");
await sleep(1500); await waitDone(pid2); await sleep(1500);
check("answer added to the case file", (await api(`/api/patients/${pid2}`)).report.includes("0830 today"));

writeFileSync(join(OUT, "walkthrough_checks.txt"), log.join("\n") + "\n");
chrome.kill();
process.exit(log.some((l) => l.startsWith("FAIL")) ? 1 : 0);
