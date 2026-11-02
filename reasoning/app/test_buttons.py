"""Press every button through the running app (python app.py first), right order and wrong order.

    python test_buttons.py [image.jpg]

Every action returns the screen in this order; the helper names the parts.
"""
import sys

from gradio_client import Client, handle_file

IMAGE = sys.argv[1] if len(sys.argv) > 1 else None
NAMES = ["queue", "header", "ai", "nurse", "why", "record", "chat", "overlay", "image_text", "sbar", "file"]
BLANK = ["", "not recorded", "not recorded", None, "", "", "", "", "", "", "", "not recorded", "", "not recorded"]
c = Client("http://127.0.0.1:7860", verbose=False)
failures = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        failures.append(name)


def press(api, *args):
    return dict(zip(NAMES, c.predict(*args, api_name=api)))


def form(age="", sex="not recorded", arrival="not recorded", pick=None, details="", hr="", sbp="", dbp="", rr="", spo2="",
         temp="", mental="not recorded", pain="", injured="not recorded"):
    return [age, sex, arrival, pick, details, hr, sbp, dbp, rr, spo2, temp, mental, pain, injured]


def queue_ids(s):
    return [x[1] for x in s["queue"]["choices"]]


def queue_labels(s):
    return [x[0] for x in s["queue"]["choices"]]


# wrong order: nobody selected, nothing may crash or change
for api, args in [("/accept", []), ("/change_level", ["2", "because"]), ("/recheck", ["80", "", "", "", "", "", "not recorded", ""]),
                  ("/explain", []), ("/add_result", ["bloods", "x"]), ("/send", ["hello"]), ("/sbar", []), ("/remove", [])]:
    c.predict(*args, api_name=api)
check("every button with nobody selected: no crash", True)

s = press("/triage", *form())
check("triage with no complaint is refused", s["record"] == "" and not queue_ids(s))
s = press("/triage", *form(pick="Headache", hr="abc", temp="20", pain="99"))
check("bad numbers are refused", s["record"] == "" and not queue_ids(s))

# made-up waiting list, most urgent first
s = press("/load_demo")
ids, labels = queue_ids(s), queue_labels(s)
check("waiting list has 3 made-up patients", len(ids) == 3)
check("most urgent first", "Chest pain" in labels[0] and "L2" in labels[0] and "unconfirmed" in labels[0])
check("waiting times shown", "6 min" in labels[0] and "50 min" in " ".join(labels))

# select: the screen and form follow the patient
out = c.predict(ids[0], api_name="/select")
s = dict(zip(NAMES, out))
check("selecting shows the header", all(x in s["header"] for x in ("LEVEL 2", "67 yrs", "Chest pain", "waited 6 min", "See within 15 min", "rough PACS P2")))
check("header says AI suggestion until a nurse confirms", "not yet confirmed" in s["header"])
check("selecting fills the form", "67" in out[len(NAMES):] and "Chest pain" in out[len(NAMES):])
check("selecting clears the Change level boxes", out[-2] is None and out[-1] == "")
check("AI box shows the suggestion", "AI suggests" in s["ai"] and "Urgency score" in s["ai"] and "FLAGGED URGENT" in s["ai"])
check("no explanation until asked", "No explanation written yet" in s["why"])
check("record has card, ED assessment and decision", all(x in s["record"] for x in ("PRESENTING COMPLAINT", "ED ASSESSMENT", "URGENCY DECISION")))
s = press("/explain")
check("explain button fills in the reasons", "Why:" in s["why"] and "Next steps" in s["why"])

# a triaged patient can't be triaged again: the first level stays
s = press("/triage", *form(pick="Headache"))
check("re-triage of a selected patient is refused", "Chest pain" in s["header"] and len(queue_ids(s)) == 3)

# nurse decides
s = press("/accept")
check("accept is recorded", "Nurse accepted the AI's level 2" in s["nurse"] and "Nurse accepted" in s["record"] and "nurse confirmed" in s["header"])
check("accept clears unconfirmed in the list", "unconfirmed" not in queue_labels(s)[0])
out = c.predict("3", "   ", api_name="/change_level")
check("change without a reason is refused", "Nurse set level" not in out[3])
out = c.predict("3", "Pain settled, normal ECG", api_name="/change_level")
s = dict(zip(NAMES, out))
check("change level is recorded with the reason", "Nurse set level 3 (AI suggested 2)" in s["nurse"] and "Pain settled" in s["record"] and out[-1] == "" and out[-2] is None)
check("header follows the nurse", "LEVEL 3" in s["header"] and "See within 30 min" in s["header"])
check("AI suggestion still shown", "Level 2" in s["ai"])
check("list re-sorts by the nurse's level", any("L3" in x and "Chest pain" in x for x in queue_labels(s)))
out = c.predict("2", "Back to urgent", api_name="/change_level")
s = dict(zip(NAMES, out))

# repeat vitals: first level kept, rising score warned
s = press("/recheck", "130", "85", "", "28", "88", "", "alert", "")
check("repeat vitals are recorded", "Repeat vitals at" in s["record"] and "HR 130" in s["record"])
check("first level stays after repeat vitals", "LEVEL 2" in s["header"] and "URGENCY DECISION" in s["record"])
before = s["record"]
s = press("/recheck", "", "", "", "", "", "", "not recorded", "")
check("empty repeat vitals are refused", s["record"] == before)
s = press("/recheck", "abc", "", "", "", "", "", "not recorded", "")
check("bad repeat vitals are refused", s["record"] == before)

# results
out = c.predict("bloods", "Troponin 2.1 ng/mL (high)", api_name="/add_result")
s = dict(zip(NAMES, out))
check("result reaches the record", "BLOOD TESTS: Troponin 2.1" in s["record"] and out[-1] == "")
out = c.predict("bloods", "   ", api_name="/add_result")
check("empty result is refused", dict(zip(NAMES, out))["record"].count("BLOOD TESTS: Troponin 2.1 ng/mL (high)\n") == 1)
for key in ("ecg", "urine_swabs", "history", "past_history", "medications", "allergies", "imaging", "treatment", "disposition"):
    s = dict(zip(NAMES, c.predict(key, f"test {key}", api_name="/add_result")))
    check(f"section {key} reaches the record", f"test {key}" in s["record"])
check("arrival card lines not repeated", s["record"].count("ALLERGIES:") == 1 and s["record"].count("IMAGING:") == 1)

# images
if IMAGE:
    s = press("/read_image", handle_file(IMAGE), "bone_xray")
    check("bone X-ray: overlay + description + detector", bool(s["overlay"]) and "Fracture detector" in s["image_text"] and "AI description" in s["image_text"])
    check("image reaches the record", "bone X-ray, AI description" in s["record"])
    s = press("/read_image", handle_file(IMAGE), "chest_xray")
    check("chest X-ray: no fracture detector", bool(s["overlay"]) and "Fracture detector" not in s["image_text"])
    s = press("/read_image", handle_file(IMAGE), "ecg")
    check("ECG picture goes to the ECG section", "12-lead ECG, AI description" in s["record"])
else:
    print("SKIP image tests: no image given")

# chat
out = c.predict("What worries you most?", api_name="/send")
s = dict(zip(NAMES, out))
check("chat answers", len(s["chat"]) == 2 and out[-1] == "")
check("chat is told about new results", "Troponin" in str(s["chat"]))
s = dict(zip(NAMES, c.predict("And now?", api_name="/send")))
check("chat remembers earlier turns", len(s["chat"]) == 4 and "remembered: 1" in str(s["chat"]))
s = dict(zip(NAMES, c.predict("   ", api_name="/send")))
check("empty question is refused", len(s["chat"]) == 4)

# SBAR
s = press("/sbar")
check("SBAR has all 4 headings", all(h in s["sbar"] for h in ("Situation", "Background", "Assessment", "Recommendation")))
check("SBAR uses accumulated results", "Troponin" in s["sbar"])
check("SBAR download file made", bool(s["file"]))
check("SBAR added to the chat", len(s["chat"]) == 6)

# switching patients keeps each one's own work
s = dict(zip(NAMES, c.predict(ids[1], api_name="/select")))
check("another patient has their own empty chat and record", s["chat"] == [] and "Troponin" not in s["record"] and s["sbar"] == "")
s = dict(zip(NAMES, c.predict(ids[0], api_name="/select")))
check("back to the first: their work is still there", len(s["chat"]) == 6 and "Troponin" in s["record"])

# critical look
out = c.predict(api_name="/new_patient")
s = dict(zip(NAMES, out))
check("new patient clears the screen and form", s["header"] == "" and s["record"] == "" and list(out[len(NAMES):len(NAMES) + 14]) == BLANK and out[-2] is None and out[-1] == "")
s = press("/critical", *form())
check("critically ill with no typing: level 1 at the top", "LEVEL 1" in s["header"] and "See immediately" in s["header"] and "L1" in queue_labels(s)[0])
check("critical look is recorded", "Nurse critical look" in s["record"] and "Critically ill on arrival" in s["record"])
s = press("/accept")
check("accept can't quietly lower a critically ill patient", "LEVEL 1" in s["header"] and "accepted" not in s["record"])
press("/new_patient")
s = press("/triage", *form(age="58", arrival="ambulance", pick="Collapse or fainting", mental="unresponsive"))
check("unresponsive with no vitals is flagged", "FLAGGED URGENT" in s["ai"])
check("blank vitals are not recorded, never 0", "HR not recorded" in s["record"] and "HR 0" not in s["record"] and "Temp not recorded" in s["record"])
check("explanation fills in after triage", "Why:" in s["why"])
s = press("/critical", *form())
check("critical button on a selected patient raises them to level 1", "LEVEL 1" in s["header"] and "Nurse critical look" in s["nurse"])
press("/new_patient")
s = press("/triage", *form(age="50", sex="female", arrival="walk-in", pick="Headache", details="typed extra", hr="80.5", sbp="120", dbp="80", rr="16", spo2="98", temp="37.5", mental="confused", pain="0", injured="no"))
check("typed details, decimals, pain 0 and 'confused' work", all(x in s["record"] for x in ("Headache: typed extra", "Temp 37.5 C", "Pain 0/10")))
press("/new_patient")
s = press("/triage", *form(age="30", pick="Dizzy after a long run", hr="80", sbp="120", dbp="80", rr="16", spo2="98", temp="36.8", mental="alert"))
check("a complaint not on the list can be typed", "Dizzy after a long run" in s["record"])

# remove
n = len(queue_ids(s))
out = c.predict(api_name="/remove")
s = dict(zip(NAMES, out))
check("seen by doctor removes the patient", len(queue_ids(s)) == n - 1 and s["header"] == "")

print(f"\n{len(failures)} failure(s)", failures)
sys.exit(bool(failures))
