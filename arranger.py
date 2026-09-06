"""
Arrangement Engine — New Orleans Street Brass Band
Style DNA: Big 6, Rebirth, Hot 8, Young Pinstripes, To Be Continued, Lil Rascals

Rules baked in:
- Melody bounces between horns by section — not one instrument all song
- Trumpet cuts through on bright peaks, Tenor Sax on smoky/mid passages
- Trombones growl on backgrounds — stabs and hits, not sustained harmony
- Tuba thumps root/5th with a heavy bounce, not a walking bass line
- Pickups are aggressive, rhythmic, syncopated — not pretty
- Harmonies are raw — parallel 3rds/4ths, not textbook voice leading
- Unison lines sound harder than pretty chords — use them
- Space = power — leave holes, don't fill everything
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import anthropic
import json


# ── Instrument definitions ────────────────────────────────────────────────────

INSTRUMENTS = {
    "trumpet1":  {"transposition": -2, "clef": "treble", "range": (55, 84),  "timbre": "bright_piercing"},
    "trumpet2":  {"transposition": -2, "clef": "treble", "range": (52, 79),  "timbre": "bright_mid"},
    "trombone1": {"transposition":  0, "clef": "bass",   "range": (40, 67),  "timbre": "growl_lead"},
    "trombone2": {"transposition":  0, "clef": "bass",   "range": (38, 65),  "timbre": "growl_mid"},
    "trombone3": {"transposition":  0, "clef": "bass",   "range": (36, 62),  "timbre": "growl_low"},
    "tenorSax":  {"transposition": -2, "clef": "treble", "range": (49, 76),  "timbre": "smoky_mid"},
    "tuba":      {"transposition":  0, "clef": "bass",   "range": (28, 52),  "timbre": "thump_foundation"},
}

# Melody carrier preference by section and feel
MELODY_ASSIGN = {
    "intro":   ["trumpet1", "tenorSax"],
    "verse":   ["tenorSax", "trumpet1", "trombone1"],
    "chorus":  ["trumpet1", "trumpet2"],   # unison or tight harmony
    "bridge":  ["tenorSax", "trombone1"],
    "outro":   ["trumpet1", "tenorSax"],
    "section": ["trumpet1", "tenorSax"],
}

# Interval offsets for raw NOLA harmony (in semitones above melody)
# Prefer 3rds and 4ths — skip textbook voice leading
HARMONY_INTERVALS = {
    "major": [4, 7, 3],   # major 3rd, 5th, minor 3rd
    "minor": [3, 7, 4],
}


# ── Note utility functions ────────────────────────────────────────────────────

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

def pitch_to_name(midi_pitch):
    return NOTE_NAMES[midi_pitch % 12] + str((midi_pitch // 12) - 1)

def transpose_pitch(pitch, semitones):
    return pitch + semitones

def clamp_pitch(pitch, inst_id):
    lo, hi = INSTRUMENTS[inst_id]["range"]
    while pitch < lo:
        pitch += 12
    while pitch > hi:
        pitch -= 12
    return int(max(0, min(127, pitch)))

def concert_to_written(pitch, inst_id):
    """Convert concert pitch to written pitch for transposing instruments."""
    offset = INSTRUMENTS[inst_id]["transposition"]
    written = pitch - offset  # written = concert - transposition offset
    # Clamp to valid MIDI range
    return int(max(0, min(127, written)))


# ── NOLA Rhythmic Vocabulary ──────────────────────────────────────────────────

# All in units of 8th notes within a measure (4/4 = 8 eighth notes)
# 0 = beat 1, 2 = beat 2, 4 = beat 3, 6 = beat 4

PICKUP_PATTERNS = {
    "slow_drag": [
        [5, 7],        # "and" of 3, "and" of 4
        [6, 7],        # 4 and "and" of 4
        [3, 5, 7],
    ],
    "second_line": [
        [1, 3, 5, 7],  # all the ands
        [0, 3, 4, 7],
        [2, 5, 6],
    ],
    "mid_tempo": [
        [3, 5, 7],
        [1, 5, 6],
        [0, 3, 6],
    ],
    "up_tempo_funk": [
        [0, 1, 4, 5],
        [1, 3, 5, 7],
        [0, 2, 4, 6],
    ],
}

TUBA_PATTERNS = {
    "slow_drag":      [(0, "root"), (4, "5th"), (6, "root")],
    "mid_tempo":      [(0, "root"), (3, "5th"), (4, "root"), (7, "5th")],
    "second_line":    [(0, "root"), (2, "5th"), (4, "root"), (6, "5th")],
    "up_tempo_funk":  [(0, "root"), (1, "root"), (4, "5th"), (6, "root")],
}


# ── Note to MIDI for chord tones ─────────────────────────────────────────────

def chord_root_midi(chord_name, octave=2):
    """Get MIDI pitch of chord root in given octave."""
    # Strip quality suffix
    for note in reversed(NOTE_NAMES):
        if chord_name.startswith(note):
            idx = NOTE_NAMES.index(note)
            return 12 + (octave * 12) + idx
    return 36  # fallback C2

def chord_fifth_midi(chord_name, octave=2):
    return chord_root_midi(chord_name, octave) + 7


# ── Section-level note assignment ────────────────────────────────────────────

def assign_melody_to_section(section_label, note_sequence, section_start_beat, section_end_beat):
    """Pull notes from the full sequence that fall in this section."""
    return [
        n for n in note_sequence
        if section_start_beat <= n["start_beat"] < section_end_beat
    ]


def build_harmony_note(melody_pitch, chord_name, key_mode, inst_id, harmonic_role):
    """Build a harmony pitch above or below the melody note."""
    intervals = HARMONY_INTERVALS.get(key_mode, HARMONY_INTERVALS["major"])
    if harmonic_role == "third":
        offset = intervals[0]
    elif harmonic_role == "fifth":
        offset = intervals[1]
    else:
        offset = intervals[2]

    pitch = melody_pitch + offset
    pitch = clamp_pitch(pitch, inst_id)
    return pitch


def build_tuba_line(chord_sequence, feel, bpm):
    """
    Build tuba part: root-fifth bounce pattern based on feel.
    Returns list of {pitch, start_beat, duration_beats}
    """
    pattern = TUBA_PATTERNS.get(feel, TUBA_PATTERNS["second_line"])
    tuba_notes = []
    beat_cursor = 0

    for chord_entry in chord_sequence:
        chords = chord_entry.get("chords", ["C"])
        # Each chord gets ~4 beats
        for chord in chords:
            for (offset_8th, degree) in pattern:
                beat_pos = beat_cursor + offset_8th / 2.0
                if degree == "root":
                    pitch = chord_root_midi(chord, octave=2)
                else:
                    pitch = chord_fifth_midi(chord, octave=2)
                pitch = clamp_pitch(pitch, "tuba")
                tuba_notes.append({
                    "pitch": pitch,
                    "pitch_name": pitch_to_name(pitch),
                    "written_pitch": concert_to_written(pitch, "tuba"),
                    "start_beat": round(beat_pos, 3),
                    "duration_beats": 0.4,
                    "velocity": 110,
                })
            beat_cursor += 4.0

    return tuba_notes


def build_pickup_riff(chord_name, feel, start_beat, inst_id, key_mode):
    """
    Build a pickup/background riff for a non-melody instrument.
    Heavy, rhythmic, syncopated — not sustained.
    """
    import random
    patterns = PICKUP_PATTERNS.get(feel, PICKUP_PATTERNS["second_line"])
    pattern = random.choice(patterns)

    root = chord_root_midi(chord_name, octave=3)
    fifth = root + 7
    third = root + (4 if key_mode == "major" else 3)

    pitches = [root, fifth, third, root + 12]
    riff = []

    for i, offset_8th in enumerate(pattern):
        beat_pos = start_beat + offset_8th / 2.0
        pitch = pitches[i % len(pitches)]
        pitch = clamp_pitch(pitch, inst_id)
        riff.append({
            "pitch": pitch,
            "pitch_name": pitch_to_name(pitch),
            "written_pitch": concert_to_written(pitch, inst_id),
            "start_beat": round(beat_pos, 3),
            "duration_beats": 0.25,
            "velocity": 95,
        })

    return riff


# ── Main arrangement engine ───────────────────────────────────────────────────

def generate_arrangement(analysis):
    """
    Build full 7-part arrangement from audio analysis.
    Returns complete arrangement dict with notes per instrument.
    """
    feel = analysis["feel"]
    key_mode = analysis["key_mode"]
    bpm = analysis["bpm"]
    sections = analysis["sections"]
    chord_data = analysis["chord_progression"]
    note_sequence = analysis["note_sequence"]

    parts = {inst: [] for inst in INSTRUMENTS}

    chord_lookup = {c["section"]: c["chords"] for c in chord_data}

    for i, section in enumerate(sections):
        label = section["label"]
        s_beat = section["start_beat"]
        e_beat = section["end_beat"]
        chords = chord_lookup.get(label, ["C"])
        first_chord = chords[0] if chords else "C"

        # Who carries melody this section?
        melody_candidates = MELODY_ASSIGN.get(label, ["trumpet1", "tenorSax"])
        melody_inst = melody_candidates[i % len(melody_candidates)]

        # Get melody notes for this section
        section_notes = assign_melody_to_section(note_sequence, note_sequence, s_beat, e_beat)

        # If we have real extracted notes, use them; otherwise build from chords
        if section_notes:
            melody_notes = section_notes
        else:
            # Build a simple melodic line from chord tones when extraction is thin
            melody_notes = _chord_tone_melody(chords, s_beat, e_beat, feel, key_mode)

        # Assign melody instrument
        for note in melody_notes:
            p = clamp_pitch(note["pitch"], melody_inst)
            parts[melody_inst].append({
                "pitch": p,
                "pitch_name": pitch_to_name(p),
                "written_pitch": concert_to_written(p, melody_inst),
                "start_beat": note["start_beat"],
                "duration_beats": note["duration_beats"],
                "velocity": 112,
                "role": "melody",
            })

        # Tuba — always playing
        tuba_notes = build_tuba_line([{"chords": chords}], feel, bpm)
        # Offset to section start
        for tn in tuba_notes:
            tn["start_beat"] += s_beat
        parts["tuba"].extend(tuba_notes)

        # Background instruments — everyone else gets pickup riffs
        bg_instruments = [k for k in INSTRUMENTS if k != melody_inst and k != "tuba"]

        # Chorus: trumpet1+2 go unison or parallel 3rds if melody is on sax/trombone
        if label == "chorus" and melody_inst not in ["trumpet1", "trumpet2"]:
            # Trumpets take melody in unison during chorus
            for note in melody_notes:
                for tinst in ["trumpet1", "trumpet2"]:
                    p = clamp_pitch(note["pitch"], tinst)
                    parts[tinst].append({
                        "pitch": p,
                        "pitch_name": pitch_to_name(p),
                        "written_pitch": concert_to_written(p, tinst),
                        "start_beat": note["start_beat"],
                        "duration_beats": note["duration_beats"],
                        "velocity": 115,
                        "role": "unison_melody",
                    })
            bg_instruments = [k for k in bg_instruments if k not in ["trumpet1","trumpet2"]]

        # Trombones — block stabs on strong beats, pickup riffs on off beats
        for tbone in ["trombone1", "trombone2", "trombone3"]:
            if tbone in bg_instruments:
                riff = build_pickup_riff(first_chord, feel, s_beat, tbone, key_mode)
                parts[tbone].extend(riff)

        # Tenor sax — counter melody or response riff when not on lead
        if "tenorSax" in bg_instruments:
            # Response riff: slightly different pattern, offset by 2 beats
            riff = build_pickup_riff(first_chord, feel, s_beat + 2, "tenorSax", key_mode)
            parts["tenorSax"].extend(riff)

        # Trumpet 2 harmony when trump1 has melody
        if melody_inst == "trumpet1" and "trumpet2" in bg_instruments:
            for note in melody_notes:
                p = build_harmony_note(note["pitch"], first_chord, key_mode, "trumpet2", "third")
                parts["trumpet2"].append({
                    "pitch": p,
                    "pitch_name": pitch_to_name(p),
                    "written_pitch": concert_to_written(p, "trumpet2"),
                    "start_beat": note["start_beat"],
                    "duration_beats": note["duration_beats"],
                    "velocity": 100,
                    "role": "harmony",
                })

    # Sort all parts by start beat
    for inst in parts:
        parts[inst].sort(key=lambda n: n["start_beat"])

    # Use Claude API to add style intelligence — section descriptions, riff guidance
    style_notes = get_style_guidance(analysis, parts)

    return {
        "metadata": {
            "key": analysis["key"],
            "bpm": analysis["bpm"],
            "feel": feel,
            "style": "New Orleans Street Brass — Big 6 / Rebirth / Hot 8 DNA",
        },
        "sections": sections,
        "chord_progression": chord_data,
        "parts": parts,
        "style_notes": style_notes,
    }


def _chord_tone_melody(chords, start_beat, end_beat, feel, key_mode):
    """Fallback melody from chord tones when pitch extraction is thin."""
    notes = []
    beat = start_beat
    beat_step = 1.0 if feel in ["second_line", "up_tempo_funk"] else 2.0

    for chord in chords:
        root = chord_root_midi(chord, octave=4)
        third = root + (4 if key_mode == "major" else 3)
        fifth = root + 7

        for p in [root, third, fifth, root]:
            if beat >= end_beat:
                break
            notes.append({
                "pitch": p,
                "pitch_name": pitch_to_name(p),
                "start_beat": round(beat, 3),
                "duration_beats": beat_step * 0.9,
                "velocity": 100,
            })
            beat += beat_step

    return notes


def get_style_guidance(analysis, parts):
    """
    Call Claude API to generate human-readable style notes per instrument
    and overall performance direction — pure NOLA street feel.
    """
    try:
        client = anthropic.Anthropic()

        chord_summary = []
        for c in analysis["chord_progression"]:
            chord_summary.append(f"{c['section']}: {' - '.join(c['chords'][:4])}")

        section_summary = [f"{s['label']} ({s['start_sec']:.0f}s-{s['end_sec']:.0f}s)" 
                          for s in analysis["sections"]]

        prompt = f"""You are a New Orleans street brass band director. Style: Big 6, Rebirth Brass Band, Hot 8, Young Pinstripes, To Be Continued, Lil Rascals. Urban, raw, heavy pocket. NOT Storyville Stompers. NOT polished traditional. 

Song analysis:
- Key: {analysis['key']}
- BPM: {analysis['bpm']}
- Feel: {analysis['feel']}
- Sections: {', '.join(section_summary)}
- Chord progression: {'; '.join(chord_summary)}

Write performance direction for each instrument. Be specific to New Orleans street style:
- How should they play their tone (growl, push, lay back, etc.)
- Where do they sit in the pocket (on top, slightly behind, etc.)
- What's their energy in each section
- Any specific articulation (ghosted notes, bends, smears, stabs)

Respond ONLY as JSON, no markdown:
{{
  "trumpet1": "...",
  "trumpet2": "...",
  "trombone1": "...",
  "trombone2": "...",
  "trombone3": "...",
  "tenorSax": "...",
  "tuba": "...",
  "bandDirection": "...",
  "feelNotes": "..."
}}"""

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        text = message.content[0].text.strip()
        text = text.replace("```json","").replace("```","").strip()
        return json.loads(text)

    except Exception as e:
        print(f"[arranger] Style guidance failed: {e}")
        return {
            "bandDirection": "Heavy pocket. Sit behind the beat. Let the tuba breathe.",
            "feelNotes": "Raw and aggressive. Space is power. Don't fill everything.",
        }
