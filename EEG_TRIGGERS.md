# EEG Trigger Reference

## Trigger path

1. Marker send entry point: [send_eeg_marker in main.py](main.py#L297).
2. Hardware send call: [setParallelData call in main.py](main.py#L304).
3. Transport implementation: [triggers.py](triggers.py).
4. Real parallel write path: [setParallelData = port.setData](triggers.py#L29).
5. Fallback fake path when parallel write is not implemented: [setParallelData function](triggers.py#L22).
6. Flip-locked send mechanism for frame-based phases: [_present_frames uses callOnFlip](main.py#L346).

## Marker catalog

| Marker key | Code | Where configured | Where it appears in task flow | How it is sent |
|---|---:|---|---|---|
| trial_start | 1 | [settings.json](settings.json#L39) | [run_trial start](main.py#L658) | Immediate direct call to send_eeg_marker |
| fixation_onset | 10 | [settings.json](settings.json#L40) | [set in _run_fixation](main.py#L466), passed as first_flip_marker in [main.py](main.py#L470) | Flip-locked via _present_frames callOnFlip at [main.py](main.py#L346) |
| scenario_onset | 20 | [settings.json](settings.json#L41) | Scheduled in [_run_scenario](main.py#L481) with marker key at [main.py](main.py#L483) | Flip-locked callOnFlip |
| choice_onset | 30 | [settings.json](settings.json#L42) | Scheduled in [_run_choice](main.py#L524) with marker key at [main.py](main.py#L526) | Flip-locked callOnFlip |
| response_left | 40 | [settings.json](settings.json#L43) | Sent after left response in [_run_choice](main.py#L543) | Immediate direct call to send_eeg_marker |
| response_right | 41 | [settings.json](settings.json#L44) | Sent after right response in [_run_choice](main.py#L546) | Immediate direct call to send_eeg_marker |
| anticipation_ai | 50 | [settings.json](settings.json#L45) | Selected in [_run_anticipation](main.py#L557), passed at [main.py](main.py#L565) | Flip-locked via _present_frames callOnFlip at [main.py](main.py#L346) |
| anticipation_human | 51 | [settings.json](settings.json#L46) | Selected in [_run_anticipation](main.py#L560), passed at [main.py](main.py#L565) | Flip-locked via _present_frames callOnFlip at [main.py](main.py#L346) |
| trigger_agree | 100 | [settings.json](settings.json#L47) | Selected in [_run_trigger](main.py#L576), passed at [main.py](main.py#L588) | Flip-locked via _present_frames callOnFlip at [main.py](main.py#L346) |
| trigger_disagree | 200 | [settings.json](settings.json#L48) | Selected in [_run_trigger](main.py#L580), passed at [main.py](main.py#L588) | Flip-locked via _present_frames callOnFlip at [main.py](main.py#L346) |
| blank_onset | 70 | [settings.json](settings.json#L49) | Passed in [_run_blank](main.py#L596) | Flip-locked via _present_frames callOnFlip at [main.py](main.py#L346) |
| trial_end | 9 | [settings.json](settings.json#L50) | [run_trial end](main.py#L711) | Immediate direct call to send_eeg_marker |

## Notes

- Response markers are intentionally sent immediately after key detection, not on the next flip.
- Reconsideration phase has no EEG marker emission in the current code.
- Every send also writes a timestamped marker log entry in [send_eeg_marker](main.py#L305).
