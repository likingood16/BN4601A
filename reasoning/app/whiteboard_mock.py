"""Whiteboard MOCK (Gradio): what the organisation whiteboard will look like, with made-up patients. No AI.

    python whiteboard_mock.py            # this laptop: http://127.0.0.1:7870
    python whiteboard_mock.py --lan      # also reachable from a second laptop on the same Wi-Fi

The board is shared: every window sees the same patients, so open it twice (or on two laptops) and change one.
Only the organising doctor and the triage nurse can change it; everyone else can only look.
Design rules (APP_DESIGN.md): big text readable from about 3 m, few columns, colour for the level.
"""
import argparse
import html
import time

import gradio as gr

LEVEL_COLOUR = {1: "#d32f2f", 2: "#f57c00", 3: "#fbc02d", 4: "#388e3c", 5: "#1976d2"}
STAGES = ["Waiting", "Being seen", "Results pending", "Admitted / ward", "Discharged"]
EDITORS = ["Dr Lim (organising doctor)", "Nurse Tan (triage nurse)"]
VIEWERS = ["Dr Wong", "Nurse Aisha", "Ward 7 clerk"]
DOCTORS = ["unassigned", "Dr Lim", "Dr Wong", "Dr Patel", "Ward 7 (cardiology)"]

now = time.time()
# shared by every window. todo: [text, done, suggested_by_ai]
BOARD = [
    {"bed": "R1", "name": "M, 67", "complaint": "Chest pain, sweaty", "level": 2, "stage": "Being seen", "doctor": "Dr Lim",
     "arrived": now - 38 * 60, "todo": [["ECG within 10 min", True, True], ["Troponin", False, True], ["Aspirin 300 mg?", False, True]]},
    {"bed": "R2", "name": "F, 24", "complaint": "Short of breath, asthma", "level": 2, "stage": "Waiting", "doctor": "unassigned",
     "arrived": now - 6 * 60, "todo": [["Peak flow", False, True], ["Nebuliser", False, False]]},
    {"bed": "C4", "name": "M, 45", "complaint": "Fall, wrist pain", "level": 4, "stage": "Results pending", "doctor": "Dr Wong",
     "arrived": now - 95 * 60, "todo": [["Wrist X-ray", True, True], ["Read X-ray", False, True]]},
    {"bed": "C7", "name": "F, 81", "complaint": "Confused since morning", "level": 3, "stage": "Waiting", "doctor": "unassigned",
     "arrived": now - 52 * 60, "todo": [["Glucose", False, True], ["Urine dip", False, True], ["Collateral history", False, True]]},
    {"bed": "W2", "name": "M, 30", "complaint": "Cut finger", "level": 5, "stage": "Discharged", "doctor": "Dr Patel",
     "arrived": now - 140 * 60, "todo": [["Tetanus status", True, True]]},
]

CSS = """
body, .gradio-container {background:#0f1720 !important;}
#board table {width:100%; border-collapse:collapse; font-size:26px; color:#f5f5f5;}
#board th {text-align:left; font-size:18px; color:#9fb3c8; padding:8px 12px; border-bottom:2px solid #33475b;}
#board td {padding:14px 12px; border-bottom:1px solid #22313f; vertical-align:top;}
#board .lvl {display:inline-block; width:56px; height:56px; line-height:56px; text-align:center; border-radius:10px;
             font-weight:800; font-size:30px; color:#111;}
#board td:nth-child(6) {white-space:nowrap;}
#board .late {color:#ff8a80; font-weight:700;}
#board .chip {display:inline-block; margin:2px 6px 2px 0; padding:2px 10px; border-radius:14px; font-size:18px;
              background:#2b3e50; color:#e3f2fd;}
#board .chip.done {text-decoration:line-through; opacity:.45;}
#board .done-row td {opacity:.4;}
#board .clock {font-size:18px; color:#9fb3c8;}
"""


def board_html():
    rows = sorted(BOARD, key=lambda p: (p["stage"] == "Discharged", p["level"], p["arrived"]))
    out = ["<div class='clock'>Updated " + time.strftime("%H:%M:%S") + " · ✦ = suggested by AI, a person ticks it off</div>",
           "<table><tr><th>Level</th><th>Bed</th><th>Patient</th><th>Stage</th><th>Doctor</th><th>Waiting</th><th>To do</th></tr>"]
    for p in rows:
        mins = int((time.time() - p["arrived"]) // 60)
        late = "late" if mins > 60 and p["stage"] in ("Waiting", "Results pending") else ""
        chips = "".join(f"<span class='chip {'done' if d else ''}'>{'✦ ' if ai else ''}{html.escape(t)}</span>"
                        for t, d, ai in p["todo"])
        out.append(f"<tr class='{'done-row' if p['stage'] == 'Discharged' else ''}'>"
                   f"<td><span class='lvl' style='background:{LEVEL_COLOUR[p['level']]}'>{p['level']}</span></td>"
                   f"<td>{p['bed']}</td><td><b>{html.escape(p['name'])}</b><br>"
                   f"<span style='font-size:20px;color:#b0bec5'>{html.escape(p['complaint'])}</span></td>"
                   f"<td>{p['stage']}</td><td>{html.escape(p['doctor'])}</td>"
                   f"<td class='{late}'>{mins // 60}h {mins % 60:02d}m</td><td>{chips}</td></tr>")
    return "".join(out) + "</table>"


def labels():
    return [f"{p['bed']} · {p['name']} · {p['complaint']}" for p in BOARD]


def find(label):
    return next((p for p in BOARD if label and label.startswith(p["bed"] + " ·")), None)


def can_edit(who):
    return who in EDITORS


def pick(label):
    p = find(label)
    if not p:
        return gr.update(), gr.update(), gr.update(choices=[], value=[])
    todos = [t for t, _, _ in p["todo"]]
    return p["stage"], p["doctor"], gr.update(choices=todos, value=[t for t, d, _ in p["todo"] if d])


def save(who, label, stage, doctor, ticked, new_todo):
    p = find(label)
    if not can_edit(who):
        return board_html(), "Only the organising doctor or the triage nurse can change the board.", new_todo
    if not p:
        return board_html(), "Pick a patient first.", new_todo
    p["stage"], p["doctor"] = stage, doctor
    for t in p["todo"]:
        t[1] = t[0] in (ticked or [])
    if new_todo and new_todo.strip():
        p["todo"].append([new_todo.strip(), False, False])
    return board_html(), f"Saved by {who} at {time.strftime('%H:%M')}.", ""


def who_changed(who):
    msg = "You can change the board." if can_edit(who) else "View only: you can look but not change the board."
    return gr.update(visible=can_edit(who)), msg


with gr.Blocks(title="ED Whiteboard (mock)") as demo:
    gr.Markdown("## <span style='color:#f5f5f5'>ED Whiteboard</span> <span style='color:#9fb3c8;font-size:16px'>"
                "MOCK with made-up patients · no AI here</span>")
    board = gr.HTML(board_html(), elem_id="board")
    gr.Timer(5).tick(board_html, outputs=board)            # waiting times tick and other windows' changes appear
    with gr.Accordion("Change the board", open=False):
        who = gr.Dropdown(EDITORS + VIEWERS, value=EDITORS[0], label="You are")
        status = gr.Markdown("You can change the board.")
        with gr.Group() as editor:
            patient = gr.Dropdown(labels(), label="Patient")
            with gr.Row():
                stage = gr.Dropdown(STAGES, label="Stage")
                doctor = gr.Dropdown(DOCTORS, label="Doctor", allow_custom_value=True)
            ticked = gr.CheckboxGroup([], label="Done (tick when finished)")
            new_todo = gr.Textbox(label="Add a to-do", placeholder="e.g. Repeat troponin at 3 h")
            save_btn = gr.Button("Save to board", variant="primary")
    who.change(who_changed, who, [editor, status])
    patient.change(pick, patient, [stage, doctor, ticked])
    save_btn.click(save, [who, patient, stage, doctor, ticked, new_todo], [board, status, new_todo]).then(
        pick, patient, [stage, doctor, ticked])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lan", action="store_true", help="listen on the Wi-Fi network so a second laptop can open it")
    ap.add_argument("--port", type=int, default=7870)
    a = ap.parse_args()
    demo.launch(server_name="0.0.0.0" if a.lan else "127.0.0.1", server_port=a.port, css=CSS)
