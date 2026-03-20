"""
main.py
=======
PsychoPy Coder experiment for an EEG study on moral-dilemma decision-making.

Each trial presents a daily-life moral dilemma, records the participant's
binary choice, then shows whether an **AI** (GPT-4) or **Human** source
agrees or disagrees — the critical ERP-locked event.

Architecture
------------
* Class-based  (``DilemmaExp``)
* All timing parameters, texts, and EEG marker codes live in
  ``settings.json`` (single source of truth).
* ``keyboard.Keyboard`` for low-latency response polling.
* Frame-counted stimulus presentation for the timing-critical trigger
  phase; ``win.callOnFlip()`` for sub-frame EEG-marker accuracy.
* All behavioural data written trial-by-trial to a CSV in ``/data``.

Usage
-----
    python main.py
    python main.py --settings path/to/settings.json

Prerequisites
-------------
    pip install psychopy numpy
    python preprocess_data.py          # generates stimuli/stimulus_list.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from psychopy import core, data, event, logging, visual


# ====================================================================== #
#  Experiment class
# ====================================================================== #

class DilemmaExp:
    """High-precision PsychoPy experiment for EEG moral-dilemma paradigm."""

    # Column order for the output CSV (defined once, used everywhere).
    CSV_FIELDS = [
        "participant_id", "session", "mode",
        "trial_num", "dilemma_idx", "topic_group",
        "scenario",
        "action_left", "action_right",
        "action_left_type", "action_right_type",
        "chosen_side", "chosen_action_type", "chosen_action_text",
        "reconsider_decision", "did_change_choice",
        "final_chosen_side", "final_chosen_action_type", "final_chosen_action_text",
        "choice_rt_sec", "reading_time_sec",
        "reconsider_rt_sec",
        "source", "source_label",
        "is_congruent", "congruency_label",
        "fixation_onset", "fixation_duration_sec",
        "scenario_onset", "choice_onset",
        "anticipation_onset", "trigger_onset",
        "blank_onset",
        "reconsider_onset",
        "trial_start_global", "trial_end_global",
    ]

    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #

    def __init__(self, settings_path: str = "settings.json"):
        self.settings = self._load_settings(settings_path)
        self.participant_id = uuid.uuid4().hex[:8]
        self.session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Clocks
        self.global_clock = core.Clock()
        self.rt_clock = core.Clock()

        # Bookkeeping
        self.marker_log: list[dict] = []
        self.trial_results: list[dict] = []
        self.mode: str = "experiment"  # set later via menu

        # Build components
        self._init_paths()
        self._init_window()
        self._init_stimuli()
        self._init_logging()

    # ---- helpers ----

    @staticmethod
    def _load_settings(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _init_paths(self):
        self.data_dir = Path(self.settings["paths"]["data_dir"])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.participant_id}_{self.session_ts}"
        self.csv_path = self.data_dir / f"{stem}.csv"
        self.marker_path = self.data_dir / f"{stem}_markers.csv"
        self.log_path = self.data_dir / f"{stem}.log"

    def _init_window(self):
        d = self.settings["display"]
        self.win = visual.Window(
            fullscr=d["fullscr"],
            monitor=d["monitor"],
            screen=d["screen"],
            color=d["background_color"],
            units=d["units"],
            waitBlanking=d["waitBlanking"],
            allowGUI=False,
        )
        # Measure actual frame duration for frame-counted waits.
        measured = self.win.getActualFrameRate(
            nIdentical=20, nMaxFrames=200, nWarmUpFrames=10
        )
        if measured is not None:
            self.frame_dur = 1.0 / measured
        else:
            self.frame_dur = 1.0 / 60.0
            logging.warning(
                "Could not measure refresh rate — assuming 60 Hz."
            )
        logging.info(
            f"Frame duration: {self.frame_dur * 1000:.2f} ms "
            f"({1 / self.frame_dur:.1f} Hz)"
        )

    def _init_stimuli(self):
        """Pre-build every ``visual.TextStim`` used in the experiment."""
        t = self.settings["text"]

        # -- fixation cross --
        self.stim_fixation = visual.TextStim(
            self.win, text="+",
            font=t["font"], height=t["fixation_height"],
            color=t["color"], wrapWidth=t["wrap_width"],
        )

        # -- scenario body --
        self.stim_scenario = visual.TextStim(
            self.win, text="",
            font=t["font"], height=t["scenario_height"],
            color=t["color"], wrapWidth=t["wrap_width"],
            alignText="left", anchorHoriz="center",
        )
        self.stim_scenario_prompt = visual.TextStim(
            self.win, text="Press SPACE when you have finished reading.",
            font=t["font"], height=t["height"] * 0.75,
            color=[0.5, 0.5, 0.5], pos=(0, -0.42),
            wrapWidth=t["wrap_width"],
        )

        # -- binary-choice screen --
        self.stim_choice_header = visual.TextStim(
            self.win, text="What would you do?",
            font=t["font"], height=t["height"],
            color=[0.7, 0.7, 0.7], pos=(0, 0.35),
        )
        self.stim_choice_left = visual.TextStim(
            self.win, text="",
            font=t["font"], height=t["choice_height"],
            color=t["color"], pos=(-0.25, 0.05),
            wrapWidth=0.35, alignText="center",
        )
        self.stim_choice_right = visual.TextStim(
            self.win, text="",
            font=t["font"], height=t["choice_height"],
            color=t["color"], pos=(0.25, 0.05),
            wrapWidth=0.35, alignText="center",
        )
        kl = self.settings["keys"]["action_left"].upper()
        kr = self.settings["keys"]["action_right"].upper()
        self.stim_label_left = visual.TextStim(
            self.win, text=f"[{kl}]",
            font=t["font"], height=t["height"] * 0.85,
            color=[0.5, 0.5, 0.5], pos=(-0.25, -0.20),
        )
        self.stim_label_right = visual.TextStim(
            self.win, text=f"[{kr}]",
            font=t["font"], height=t["height"] * 0.85,
            color=[0.5, 0.5, 0.5], pos=(0.25, -0.20),
        )
        self.stim_or = visual.TextStim(
            self.win, text="OR",
            font=t["font"], height=t["height"] * 0.85,
            color=[0.35, 0.35, 0.35], pos=(0, 0.05),
        )
        # thin vertical divider (drawn as a narrow rect)
        self.stim_divider = visual.Rect(
            self.win, width=0.002, height=0.35,
            pos=(0, 0.0), lineColor=None,
            fillColor=[0.25, 0.25, 0.25],
        )

        # -- anticipation cue --
        self.stim_anticipation = visual.TextStim(
            self.win, text="",
            font=t["font"], height=t["height"],
            color=[0.85, 0.85, 0.0],
            pos=(0, 0.16),
            wrapWidth=t["wrap_width"],
        )

        # -- trigger stimulus (AGREE / DISAGREE) --
        self.stim_trigger = visual.TextStim(
            self.win, text="",
            font=t["font"], height=t["trigger_height"],
            color=t["color"], bold=True,
            wrapWidth=t["wrap_width"],
        )

        # -- general-purpose info screen (consent, instructions, etc.) --
        self.stim_info = visual.TextStim(
            self.win, text="",
            font=t["font"], height=t["height"],
            color=t["color"], wrapWidth=t["wrap_width"],
            alignText="left", anchorHoriz="center",
        )

        # -- subtle progress counter --
        self.stim_progress = visual.TextStim(
            self.win, text="",
            font=t["font"], height=t["height"] * 0.65,
            color=[0.25, 0.25, 0.25], pos=(0, 0.46),
        )

        # -- post-feedback reconsideration screen (behavior only) --
        kl = self.settings["keys"]["action_left"].upper()
        kr = self.settings["keys"]["action_right"].upper()
        self.stim_reconsider_prompt = visual.TextStim(
            self.win,
            text="Would you like to change your choice?",
            font=t["font"],
            height=t["height"],
            color=t["color"],
            pos=(0, 0.20),
            wrapWidth=t["wrap_width"],
            alignText="center",
        )
        self.stim_reconsider_initial = visual.TextStim(
            self.win,
            text="",
            font=t["font"],
            height=t["height"] * 0.85,
            color=[0.8, 0.8, 0.8],
            pos=(0, 0.10),
            wrapWidth=t["wrap_width"],
            alignText="center",
        )
        self.stim_reconsider_keep = visual.TextStim(
            self.win,
            text=f"[{kl}] Keep previous choice",
            font=t["font"],
            height=t["height"] * 0.9,
            color=[0.7, 0.7, 0.7],
            pos=(0, 0.02),
            wrapWidth=t["wrap_width"],
            alignText="center",
        )
        self._reconsider_change_template = f"[{kr}] Change to: {{}}"
        self.stim_reconsider_change = visual.TextStim(
            self.win,
            text=self._reconsider_change_template.format(""),
            font=t["font"],
            height=t["height"] * 0.9,
            color=[0.7, 0.7, 0.7],
            pos=(0, -0.10),
            wrapWidth=t["wrap_width"],
            alignText="center",
        )

    def _init_logging(self):
        logging.LogFile(str(self.log_path), level=logging.INFO, filemode="w")
        logging.console.setLevel(logging.WARNING)
        logging.info(f"Participant : {self.participant_id}")
        logging.info(f"Session     : {self.session_ts}")

    # ------------------------------------------------------------------ #
    #  EEG marker placeholder
    # ------------------------------------------------------------------ #

    def send_eeg_marker(self, val: int) -> None:
        """Send an event marker to the EEG acquisition system.

        **This is a placeholder.**  Replace the body with the call
        appropriate for your hardware, for example::

            from psychopy import parallel
            parallel.setData(val)          # parallel-port trigger

            serial_port.write(bytes([val]))  # serial trigger

            lsl_outlet.push_sample([val])    # Lab Streaming Layer

        The function is registered via ``win.callOnFlip()`` so it
        executes at the exact moment of the screen refresh.
        """
        ts = self.global_clock.getTime()
        self.marker_log.append({"marker": val, "global_time": f"{ts:.6f}"})
        logging.data(f"EEG_MARKER  {val:>3d}  @  {ts:.6f} s")

    # ------------------------------------------------------------------ #
    #  Frame-counted wait helper
    # ------------------------------------------------------------------ #

    def _present_frames(
        self,
        duration_sec: float,
        draw_funcs: list | None = None,
        first_flip_marker: int | None = None,
    ) -> float:
        """Draw *draw_funcs* every frame for *duration_sec* seconds.

        Uses frame-counting instead of ``core.wait()`` for precise
        display durations.  Returns the timestamp of the **first** flip
        (stimulus onset).

        Parameters
        ----------
        duration_sec : float
            Target presentation duration.
        draw_funcs : list[callable] | None
            Each callable is invoked (no args) before every flip.
            Pass ``None`` for a blank screen.
        first_flip_marker : int | None
            EEG marker value sent via ``callOnFlip`` on the *first*
            frame only.
        """
        n_frames = max(1, int(round(duration_sec / self.frame_dur)))
        onset: float = 0.0

        for frame_n in range(n_frames):
            if draw_funcs:
                for fn in draw_funcs:
                    fn()

            if frame_n == 0:
                if first_flip_marker is not None:
                    self.win.callOnFlip(
                        self.send_eeg_marker, first_flip_marker
                    )
                # Capture the exact onset time returned by flip().
                onset = self.win.flip()
            else:
                self.win.flip()

        return onset

    # ------------------------------------------------------------------ #
    #  Central key-polling helper
    # ------------------------------------------------------------------ #

    def _wait_for_key(
        self,
        key_list: list[str],
        draw_funcs: list | None = None,
    ) -> str:
        """Poll for a keypress each frame, keeping the window alive.

        Redraws *draw_funcs* and flips every frame so macOS keeps
        delivering events to the window.  Returns the first matching
        key name as a plain string.  Automatically quits if the
        quit key is pressed.

        Parameters
        ----------
        key_list : list[str]
            Acceptable key names (should include the quit key if
            you want ESC to work).
        draw_funcs : list[callable] | None
            Callables invoked (no args) each frame before the flip.
        """
        quit_key = self.settings["keys"]["quit"]
        event.clearEvents()

        while True:
            if draw_funcs:
                for fn in draw_funcs:
                    fn()
            self.win.flip()
            keys = event.getKeys(keyList=key_list)
            if keys:
                if keys[0] == quit_key:
                    self.cleanup()
                return keys[0]

    # ------------------------------------------------------------------ #
    #  Information screens
    # ------------------------------------------------------------------ #

    def _show_text_wait(self, text: str, accept: list[str] | None = None):
        """Display *text* and block until an accepted key is pressed."""
        keys_cfg = self.settings["keys"]
        if accept is None:
            accept = [keys_cfg["continue"], keys_cfg["quit"]]

        self.stim_info.text = text
        return self._wait_for_key(accept, draw_funcs=[self.stim_info.draw])

    def show_consent(self):
        """Consent page — pressing SPACE constitutes agreement."""
        self._show_text_wait(self.settings["consent_text"])

    def show_instructions(self):
        self._show_text_wait(self.settings["instruction_text"])

    def show_end_practice(self):
        self._show_text_wait(self.settings["end_practice_text"])

    def show_break(self, done: int, total: int):
        txt = (
            f"You have completed {done} of {total} trials.\n\n"
            "Take a short rest if you need one.\n\n"
            "Press SPACE to continue."
        )
        self._show_text_wait(txt)

    def show_goodbye(self):
        txt = (
            "The experiment is now complete.\n\n"
            "Thank you for your participation!\n\n"
            "Press SPACE to exit."
        )
        self._show_text_wait(txt)

    # ------------------------------------------------------------------ #
    #  Menu
    # ------------------------------------------------------------------ #

    def show_menu(self) -> str:
        """Return ``'practice'`` or ``'experiment'``."""
        txt = (
            "Select a mode:\n\n"
            "  Press  P   for Practice  (5 trials)\n\n"
            "  Press  E   for Experiment (full run)\n\n"
            "  Press  ESC to quit"
        )
        key = self._show_text_wait(
            txt, accept=["p", "e", self.settings["keys"]["quit"]]
        )
        return "practice" if key == "p" else "experiment"

    # ------------------------------------------------------------------ #
    #  Trial phases
    # ------------------------------------------------------------------ #

    def _run_fixation(self) -> tuple[float, float]:
        """Jittered fixation cross.  Returns (onset, actual_duration)."""
        t = self.settings["timing"]
        dur = np.random.uniform(t["fixation_min_sec"], t["fixation_max_sec"])
        m = self.settings["eeg_markers"]["fixation_onset"]
        onset = self._present_frames(
            dur,
            draw_funcs=[self.stim_fixation.draw],
            first_flip_marker=m,
        )
        return onset, dur

    def _run_scenario(self, text: str) -> tuple[float, float]:
        """Self-paced scenario reading.  Returns (onset, reading_time)."""
        self.stim_scenario.text = text

        # Mark onset + reset RT clock on the very first flip
        self.stim_scenario.draw()
        self.stim_scenario_prompt.draw()
        self.win.callOnFlip(
            self.send_eeg_marker,
            self.settings["eeg_markers"]["scenario_onset"],
        )
        self.win.callOnFlip(self.rt_clock.reset)
        onset = self.win.flip()

        # Wait for SPACE (subsequent flips via polling loop)
        accept = [
            self.settings["keys"]["continue"],
            self.settings["keys"]["quit"],
        ]
        self._wait_for_key(
            accept,
            draw_funcs=[
                self.stim_scenario.draw,
                self.stim_scenario_prompt.draw,
            ],
        )

        reading_time = self.rt_clock.getTime()
        return onset, reading_time

    def _run_choice(
        self, action_left: str, action_right: str
    ) -> tuple[float, str, float]:
        """Binary choice screen.  Returns (onset, chosen_side, rt)."""
        self.stim_choice_left.text = action_left
        self.stim_choice_right.text = action_right

        # Draw everything + mark onset on the first flip
        choice_draws = [
            self.stim_choice_header.draw,
            self.stim_divider.draw,
            self.stim_choice_left.draw,
            self.stim_choice_right.draw,
            self.stim_label_left.draw,
            self.stim_label_right.draw,
            self.stim_or.draw,
        ]
        for fn in choice_draws:
            fn()

        self.win.callOnFlip(
            self.send_eeg_marker,
            self.settings["eeg_markers"]["choice_onset"],
        )
        self.win.callOnFlip(self.rt_clock.reset)
        onset = self.win.flip()

        # Collect response via polling loop
        kcfg = self.settings["keys"]
        key = self._wait_for_key(
            [kcfg["action_left"], kcfg["action_right"], kcfg["quit"]],
            draw_funcs=choice_draws,
        )

        rt = self.rt_clock.getTime()

        # Send response marker
        if key == kcfg["action_left"]:
            side = "left"
            self.send_eeg_marker(self.settings["eeg_markers"]["response_left"])
        else:
            side = "right"
            self.send_eeg_marker(
                self.settings["eeg_markers"]["response_right"]
            )

        return onset, side, rt

    def _run_anticipation(self, source: str) -> float:
        """Source cue (.. s).  Returns onset timestamp."""
        mrk = self.settings["eeg_markers"]
        if source == "AI":
            self.stim_anticipation.text = "AI is deciding…"
            marker = mrk["anticipation_ai"]
        else:
            self.stim_anticipation.text = "Panel of People are deciding…"
            marker = mrk["anticipation_human"]

        return self._present_frames(
            self.settings["timing"]["anticipation_sec"],
            draw_funcs=[self.stim_anticipation.draw],
            first_flip_marker=marker,
        )

    def _run_trigger(self, is_congruent: bool) -> float:
        """AGREE / DISAGREE — the ERP-locked event.  Returns onset."""
        t_cfg = self.settings["text"]
        mrk   = self.settings["eeg_markers"]

        if is_congruent:
            self.stim_trigger.text = "AGREE"
            self.stim_trigger.color = t_cfg["trigger_agree_color"]
            marker = mrk["trigger_agree"]
        else:
            self.stim_trigger.text = "DISAGREE"
            self.stim_trigger.color = t_cfg["trigger_disagree_color"]
            marker = mrk["trigger_disagree"]

        return self._present_frames(
            self.settings["timing"]["trigger_sec"],
            draw_funcs=[
                self.stim_anticipation.draw,
                self.stim_trigger.draw,
            ],
            first_flip_marker=marker,
        )

    def _run_blank(self) -> float:
        """Blank screen for baseline recovery.  Returns onset."""
        return self._present_frames(
            self.settings["timing"]["blank_sec"],
            draw_funcs=None,  # nothing drawn → background colour
            first_flip_marker=self.settings["eeg_markers"]["blank_onset"],
        )

    def _run_reconsider(
        self,
        chosen_action_text: str,
        unchosen_action_text: str,
    ) -> tuple[float, str, float]:
        """Ask whether to keep or change the previous choice.

        Behavioral phase only (no EEG marker emitted here).
        Returns (onset, decision, rt) where decision is keep|change.
        """
        self.stim_reconsider_initial.text = (
            f"Your initial choice: {chosen_action_text}"
        )
        self.stim_reconsider_change.text = (
            self._reconsider_change_template.format(unchosen_action_text)
        )
        draw_funcs = [
            self.stim_reconsider_prompt.draw,
            self.stim_reconsider_initial.draw,
            self.stim_reconsider_keep.draw,
            self.stim_reconsider_change.draw,
        ]
        for fn in draw_funcs:
            fn()

        self.win.callOnFlip(self.rt_clock.reset)
        onset = self.win.flip()

        kcfg = self.settings["keys"]
        key = self._wait_for_key(
            [kcfg["action_left"], kcfg["action_right"], kcfg["quit"]],
            draw_funcs=draw_funcs,
        )
        rt = self.rt_clock.getTime()
        decision = "keep" if key == kcfg["action_left"] else "change"
        return onset, decision, rt

    # ------------------------------------------------------------------ #
    #  Full single trial
    # ------------------------------------------------------------------ #

    def run_trial(
        self,
        trial: dict,
        trial_num: int,
        total: int,
    ) -> dict:
        """Execute one complete trial and return a result dict."""

        # Unpack stimulus fields
        scenario        = trial["scenario"]
        action_left     = trial["action_left"]
        action_right    = trial["action_right"]
        action_left_t   = trial["action_left_type"]
        action_right_t  = trial["action_right_type"]
        source          = trial["source"]
        source_label    = trial["source_label"]

        # Mark trial start
        self.send_eeg_marker(self.settings["eeg_markers"]["trial_start"])
        t_start = self.global_clock.getTime()

        # 1 ── fixation
        fix_onset, fix_dur = self._run_fixation()

        # 2 ── scenario (self-paced)
        scen_onset, reading_time = self._run_scenario(scenario)

        # 3 ── choice
        choice_onset, chosen_side, choice_rt = self._run_choice(
            action_left, action_right
        )

        if chosen_side == "left":
            chosen_type = action_left_t
            chosen_text = action_left
            unchosen_text = action_right
        else:
            chosen_type = action_right_t
            chosen_text = action_right
            unchosen_text = action_left


        # 4 ── anticipation (source cue)
        antic_onset = self._run_anticipation(source)

        # 5 ── trigger (ERP event)
        is_congruent = chosen_type == source_label
        trigger_onset = self._run_trigger(is_congruent)

        # 6 ── blank
        blank_onset = self._run_blank()

        # 7 ── post-feedback reconsideration (behavior only)
        reconsider_onset, reconsider_decision, reconsider_rt = (
            self._run_reconsider(chosen_text, unchosen_text)
        )

        did_change_choice = reconsider_decision == "change"
        if did_change_choice:
            final_side = "right" if chosen_side == "left" else "left"
        else:
            final_side = chosen_side

        if final_side == "left":
            final_type = action_left_t
            final_text = action_left
        else:
            final_type = action_right_t
            final_text = action_right

        # Mark trial end
        self.send_eeg_marker(self.settings["eeg_markers"]["trial_end"])
        t_end = self.global_clock.getTime()

        # Compile result row
        return {
            "participant_id":       self.participant_id,
            "session":              self.session_ts,
            "mode":                 self.mode,
            "trial_num":            trial_num,
            "dilemma_idx":          trial.get("dilemma_idx", ""),
            "topic_group":          trial.get("topic_group", ""),
            "scenario":             scenario,
            "action_left":          action_left,
            "action_right":         action_right,
            "action_left_type":     action_left_t,
            "action_right_type":    action_right_t,
            "chosen_side":          chosen_side,
            "chosen_action_type":   chosen_type,
            "chosen_action_text":   chosen_text,
            "reconsider_decision":  reconsider_decision,
            "did_change_choice":    int(did_change_choice),
            "final_chosen_side":    final_side,
            "final_chosen_action_type": final_type,
            "final_chosen_action_text": final_text,
            "choice_rt_sec":        f"{choice_rt:.6f}",
            "reading_time_sec":     f"{reading_time:.6f}",
            "reconsider_rt_sec":    f"{reconsider_rt:.6f}",
            "source":               source,
            "source_label":         source_label,
            "is_congruent":         int(is_congruent),
            "congruency_label":     "agree" if is_congruent else "disagree",
            "fixation_onset":       f"{fix_onset:.6f}",
            "fixation_duration_sec":f"{fix_dur:.6f}",
            "scenario_onset":       f"{scen_onset:.6f}",
            "choice_onset":         f"{choice_onset:.6f}",
            "anticipation_onset":   f"{antic_onset:.6f}",
            "trigger_onset":        f"{trigger_onset:.6f}",
            "blank_onset":          f"{blank_onset:.6f}",
            "reconsider_onset":     f"{reconsider_onset:.6f}",
            "trial_start_global":   f"{t_start:.6f}",
            "trial_end_global":     f"{t_end:.6f}",
        }

    # ------------------------------------------------------------------ #
    #  Data I/O
    # ------------------------------------------------------------------ #

    def _write_csv_header(self):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=self.CSV_FIELDS).writeheader()

    def _append_csv_row(self, row: dict):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=self.CSV_FIELDS).writerow(row)

    def _save_marker_log(self):
        with open(self.marker_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["marker", "global_time"])
            w.writeheader()
            w.writerows(self.marker_log)
        logging.info(f"Marker log → {self.marker_path}")

    # ------------------------------------------------------------------ #
    #  Main run loop
    # ------------------------------------------------------------------ #

    def run(self):
        """Entry point: consent → menu → instructions → trials → end."""
        try:
            # 1 ── consent
            self.show_consent()

            # 2 ── menu
            self.mode = self.show_menu()

            # 3 ── instructions
            self.show_instructions()

            # 4 ── load stimuli
            stim_path = Path(self.settings["trials"]["stimulus_file"])
            if not stim_path.exists():
                raise FileNotFoundError(
                    f"Stimulus file not found: {stim_path}\n"
                    "Run  python preprocess_data.py  first."
                )
            all_trials = data.importConditions(str(stim_path))

            # 5 ── select practice or full set
            if self.mode == "practice":
                n = min(
                    self.settings["trials"]["practice_count"],
                    len(all_trials),
                )
                trial_list = all_trials[:n]
            else:
                trial_list = all_trials

            n_trials = len(trial_list)

            trials = data.TrialHandler(
                trialList=trial_list,
                nReps=1,
                method="sequential" if self.mode == "practice" else "random",
                name="dilemma_trials",
            )

            # 6 ── prepare output file
            self._write_csv_header()
            break_every = self.settings["trials"]["break_every_n"]

            logging.info(
                f"Starting '{self.mode}' with {n_trials} trials."
            )

            # 7 ── trial loop
            for trial_num, trial_data in enumerate(trials, start=1):
                # optional rest break
                if (
                    self.mode == "experiment"
                    and trial_num > 1
                    and (trial_num - 1) % break_every == 0
                ):
                    self.show_break(trial_num - 1, n_trials)

                result = self.run_trial(trial_data, trial_num, n_trials)
                self.trial_results.append(result)
                self._append_csv_row(result)

            # 8 ── post-practice bridge
            if self.mode == "practice":
                self.show_end_practice()

            # 9 ── goodbye
            self.show_goodbye()

        except Exception:
            logging.error("Experiment error", exc_info=True)
            raise

        finally:
            self._save_marker_log()
            self.cleanup(save=False)

    # ------------------------------------------------------------------ #
    #  Clean-up
    # ------------------------------------------------------------------ #

    def cleanup(self, save: bool = True):
        """Gracefully close the window and quit PsychoPy."""
        if save:
            self._save_marker_log()
        logging.info("Experiment cleanup — shutting down.")
        try:
            self.win.close()
        except Exception:
            pass
        core.quit()


# ====================================================================== #
#  Entry point
# ====================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the EEG moral-dilemma PsychoPy experiment."
    )
    parser.add_argument(
        "--settings", default="settings.json",
        help="Path to the shared settings JSON file.",
    )
    args = parser.parse_args()

    exp = DilemmaExp(settings_path=args.settings)
    exp.run()
