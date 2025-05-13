## README.md – Webcam‑Gesture‑to‑Keyboard Bridge

A tiny Python utility that turns simple body motions picked up by your webcam into key‑presses.
Great for hacking together motion‑controlled games, interactive art projects, or hands‑free presentations.

---

###  What it does

| Action you do    | What the script detects | Key sent (default)               |
| ---------------- | ----------------------- | -------------------------------- |
| Right‑hand punch | `PUNCH_RIGHT`           | <kbd>Space</kbd>                 |
| Left‑hand punch  | `PUNCH_LEFT`            | <kbd>Space</kbd>                 |
| Lean forward     | `LEAN_FWD`              | _(none – sample code shows how)_ |
| Lean back        | `LEAN_BACK`             | _(none – sample code shows how)_ |

The script uses **MediaPipe Pose** to track landmarks, looks for quick wrist‑to‑shoulder distance changes (punches) and nose depth shifts (leans), and fires virtual key‑presses through **pynput** after a short cooldown.

---

###  Quick start

```bash
# 1) Create & activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Plug in / enable your webcam
#    The script defaults to camera index 1 – change to 0 if needed.
python main.py
```

> **Tip:** If nothing shows up, press <kbd>q</kbd> or <kbd>Esc</kbd> to quit, edit
> `video = cv2.VideoCapture(1)` to `VideoCapture(0)`, and run again.

---

### Customising

- **Change key bindings** – edit the `ACTION_KEY_MAP` dictionary.
- **Tweak sensitivity** – play with `SLOPE_THRESH` (punch), `LEAN_FWD_THR`, `LEAN_BACK_THR`, and `frames_per_action`.
- **Add new gestures** – create a detector that returns a new ID in `possible_actions`, then map it in `ACTION_KEY_MAP`.

---

### 🩺 Troubleshooting

| Symptom                       | Likely cause / fix                                                                                                |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Window opens but no landmarks | Low lighting or camera index wrong – brighten area / swap index. You might want to use an index of 0 for windows. |
| Frequent false punches        | Raise `SLOPE_THRESH` or `frames_per_action`                                                                       |
| Lean triggers stuck           | Delete `neutral_depth` logic or reduce `RECENTER_RATE`                                                            |

---

###  License

MIT – do what you like, no warranty. Pull requests welcome!
