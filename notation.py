"""
Notation Renderer
Converts arrangement data → MusicXML → LilyPond → PDF per instrument
Handles transposition automatically for Bb instruments
"""

import os
import subprocess
import tempfile
from pathlib import Path
from music21 import stream, note, chord as m21chord, tempo, key, meter, clef, metadata, instrument as m21instrument, layout, environment

# ── Configure music21 to find MuseScore 4 ──────────────────────────────────────

MUSESCORE_PATH = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"

env = environment.Environment()
env['musescoreDirectPNGPath'] = MUSESCORE_PATH
env['musicxmlPath'] = MUSESCORE_PATH


# ── Instrument display config ──────────────────────────────────────────────────

INST_CONFIG = {
    "trumpet1": {
        "display_name": "Trumpet 1",
        "music21_instrument": m21instrument.Trumpet,
        "clef": "treble",
        "transposition_semitones": -2,  # Bb instrument: written is 2 semitones above concert
        "key_signature_offset": 2,       # add 2 sharps (or remove 2 flats) for written key
    },
    "trumpet2": {
        "display_name": "Trumpet 2",
        "music21_instrument": m21instrument.Trumpet,
        "clef": "treble",
        "transposition_semitones": -2,
        "key_signature_offset": 2,
    },
    "trombone1": {
        "display_name": "Trombone 1",
        "music21_instrument": m21instrument.Trombone,
        "clef": "bass",
        "transposition_semitones": 0,
        "key_signature_offset": 0,
    },
    "trombone2": {
        "display_name": "Trombone 2",
        "music21_instrument": m21instrument.Trombone,
        "clef": "bass",
        "transposition_semitones": 0,
        "key_signature_offset": 0,
    },
    "trombone3": {
        "display_name": "Trombone 3",
        "music21_instrument": m21instrument.Trombone,
        "clef": "bass",
        "transposition_semitones": 0,
        "key_signature_offset": 0,
    },
    "tenorSax": {
        "display_name": "Tenor Saxophone",
        "music21_instrument": m21instrument.TenorSaxophone,
        "clef": "treble",
        "transposition_semitones": -2,
        "key_signature_offset": 2,
    },
    "tuba": {
        "display_name": "Sousaphone / Tuba",
        "music21_instrument": m21instrument.Tuba,
        "clef": "bass",
        "transposition_semitones": 0,
        "key_signature_offset": 0,
    },
}


STANDARD_DURATIONS = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

def snap_duration(duration_beats):
    """Snap a beat duration to the nearest standard music21-compatible value."""
    return min(STANDARD_DURATIONS, key=lambda x: abs(x - duration_beats))

def snap_beat(beat):
    """Round a beat position to the nearest 16th note to avoid sub-32nd-note artifacts."""
    return round(beat * 4) / 4.0

def beats_to_duration(duration_beats):
    """Convert beat duration (quarter note = 1.0) to music21 duration type."""
    snapped = snap_duration(duration_beats)
    dur_map = [
        (0.125, "32nd"),
        (0.25,  "16th"),
        (0.5,   "eighth"),
        (0.75,  "eighth"),   # dotted 16th approx
        (1.0,   "quarter"),
        (1.5,   "quarter"),  # dotted quarter approx
        (2.0,   "half"),
        (3.0,   "half"),
        (4.0,   "whole"),
    ]
    closest = min(dur_map, key=lambda x: abs(x[0] - snapped))
    return closest[1]


def build_part(inst_id, notes_data, arrangement, bpm_val, key_root, key_mode):
    """
    Build a music21 Part for one instrument.
    Handles written transposition for Bb instruments.
    """
    cfg = INST_CONFIG[inst_id]
    p = stream.Part()
    p.id = inst_id

    # Instrument
    instr_class = cfg["music21_instrument"]
    p.insert(0, instr_class())

    # Clef
    if cfg["clef"] == "bass":
        p.insert(0, clef.BassClef())
    else:
        p.insert(0, clef.TrebleClef())

    # Time signature
    p.insert(0, meter.TimeSignature("4/4"))

    # Tempo
    mm = tempo.MetronomeMark(number=bpm_val)
    p.insert(0, mm)

    # Key signature (transposed for Bb instruments)
    from music21 import key as m21key
    # Determine concert key sharps/flats
    key_sig = m21key.Key(key_root, key_mode)
    if cfg["key_signature_offset"] != 0:
        # Shift key signature for transposing instruments
        sharps = key_sig.sharps + cfg["key_signature_offset"]
        key_sig = m21key.KeySignature(sharps)
    p.insert(0, key_sig)

    if not notes_data:
        # Empty measure so the part renders
        r = note.Rest(quarterLength=4)
        p.append(r)
        return p

    # Round all beat positions to 16th notes to avoid sub-32nd-note artifacts
    for n in notes_data:
        n["start_beat"] = snap_beat(n["start_beat"])
        n["duration_beats"] = snap_duration(n["duration_beats"])

    # Sort notes by beat
    sorted_notes = sorted(notes_data, key=lambda n: n["start_beat"])

    # Group notes into measures (4 beats per measure)
    BEATS_PER_MEASURE = 4.0
    current_beat = 0.0
    note_idx = 0
    measure_num = 1

    while note_idx < len(sorted_notes):
        m = stream.Measure(number=measure_num)
        measure_start = (measure_num - 1) * BEATS_PER_MEASURE
        measure_end = measure_num * BEATS_PER_MEASURE
        measure_beat_cursor = 0.0

        while note_idx < len(sorted_notes):
            n = sorted_notes[note_idx]
            note_beat = n["start_beat"]

            if note_beat >= measure_end:
                break

            # Fill gap before note with rest
            note_measure_pos = note_beat - measure_start
            if note_measure_pos > measure_beat_cursor + 0.05:
                gap = note_measure_pos - measure_beat_cursor
                gap = min(gap, BEATS_PER_MEASURE - measure_beat_cursor)
                gap = snap_duration(max(gap, 0.125))
                if gap > 0.1:
                    r = note.Rest(quarterLength=gap)
                    m.append(r)
                    measure_beat_cursor += gap

            # How much space left in measure
            space_left = BEATS_PER_MEASURE - measure_beat_cursor
            if space_left < 0.1:
                break

            # Written pitch (already transposed in arranger)
            written_pitch = n.get("written_pitch", n["pitch"])
            # Clamp to valid MIDI range and ensure int
            written_pitch = int(max(0, min(127, written_pitch)))
            dur_beats = min(n["duration_beats"], space_left)
            dur_beats = snap_duration(max(dur_beats, 0.125))

            try:
                nn = note.Note(written_pitch)
                nn.quarterLength = dur_beats
                nn.volume.velocity = int(n.get("velocity", 100))
                m.append(nn)
            except Exception:
                r = note.Rest(quarterLength=dur_beats)
                m.append(r)

            measure_beat_cursor += dur_beats
            note_idx += 1

        # Fill remainder of measure
        remainder = BEATS_PER_MEASURE - measure_beat_cursor
        if remainder > 0.1:
            remainder = snap_duration(max(remainder, 0.125))
            r = note.Rest(quarterLength=remainder)
            m.append(r)

        p.append(m)
        measure_num += 1

    return p


def render_sheet_music(arrangement, output_dir):
    """
    Render all 7 instrument parts as individual PDFs.
    Returns dict: {inst_id: pdf_path}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    meta = arrangement["metadata"]
    bpm_val = meta["bpm"]

    # Parse key
    key_parts = meta["key"].split(" ")
    key_root = key_parts[0]
    key_mode = key_parts[1] if len(key_parts) > 1 else "major"

    parts_data = arrangement["parts"]
    style_notes = arrangement.get("style_notes", {})

    pdf_paths = {}

    for inst_id, notes_data in parts_data.items():
        cfg = INST_CONFIG.get(inst_id)
        if not cfg:
            continue

        print(f"[notation] Rendering {cfg['display_name']}...")

        try:
            # Build score for this instrument
            s = stream.Score()
            s.metadata = metadata.Metadata()
            s.metadata.title = f"New Orleans Brass Band Arrangement"
            s.metadata.composer = cfg["display_name"]

            part = build_part(inst_id, notes_data, arrangement, bpm_val, key_root, key_mode)
            s.append(part)

            # Write MusicXML — this is the guaranteed output regardless of PDF rendering
            xml_path = output_dir / f"{inst_id}.xml"
            try:
                s.write("musicxml", fp=str(xml_path))
            except Exception as xml_err:
                print(f"[notation] MusicXML write failed for {inst_id}: {xml_err}")
                continue

            if not xml_path.exists() or xml_path.stat().st_size == 0:
                print(f"[notation] MusicXML file empty or missing for {inst_id}")
                continue

            # Try MuseScore for .mscz
            mscz_path = output_dir / f"{inst_id}.mscz"
            success = _render_with_musescore(str(xml_path), str(mscz_path))

            if success and mscz_path.exists():
                pdf_paths[inst_id] = str(mscz_path)
                print(f"[notation] {cfg['display_name']} -> MSCZ {mscz_path}")
            else:
                # Fall back to MusicXML so user can open in Finale/Sibelius/MuseScore
                pdf_paths[inst_id] = str(xml_path)
                print(f"[notation] {cfg['display_name']} -> MusicXML {xml_path}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[notation] Failed for {inst_id}: {e}")
            # Even on catastrophic failure, try to save whatever XML was generated
            xml_rescue = output_dir / f"{inst_id}.xml"
            if xml_rescue.exists() and xml_rescue.stat().st_size > 0:
                pdf_paths[inst_id] = str(xml_rescue)
                print(f"[notation] {cfg['display_name']} -> XML rescue {xml_rescue}")

    # Build full score (all 7 parts combined)
    print(f"[notation] Building full score...")
    try:
        full_score = build_full_score(parts_data, arrangement, bpm_val, key_root, key_mode)
        full_score_path = render_full_score(full_score, output_dir)
        if full_score_path:
            pdf_paths["fullScore"] = full_score_path
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[notation] Full score failed: {e}")

    return pdf_paths


# ── Full score (all 7 parts in score order) ──────────────────────────────────

SCORE_ORDER = ["trumpet1", "trumpet2", "tenorSax", "trombone1", "trombone2", "trombone3", "tuba"]


def build_full_score(parts_data, arrangement, bpm_val, key_root, key_mode):
    """
    Build a combined music21 Score with all 7 instruments in score order.
    Returns the Score object.
    """
    s = stream.Score()
    s.metadata = metadata.Metadata()
    s.metadata.title = "New Orleans Brass Band Arrangement"
    s.metadata.composer = "Brass Arranger — NOLA Street Style"

    for inst_id in SCORE_ORDER:
        notes_data = parts_data.get(inst_id, [])
        part = build_part(inst_id, notes_data, arrangement, bpm_val, key_root, key_mode)
        s.append(part)

    return s


def render_full_score(score, output_dir):
    """
    Render the full score as .mscz (MuseScore) or .xml (MusicXML fallback).
    Returns the file path string.
    """
    output_dir = Path(output_dir)
    mscz_path = output_dir / "full_score.mscz"
    xml_path = output_dir / "full_score.xml"

    # Write MusicXML first — guaranteed fallback
    try:
        score.write("musicxml", fp=str(xml_path))
    except Exception as e:
        print(f"[notation] Full score MusicXML write failed: {e}")
        return None

    if not xml_path.exists() or xml_path.stat().st_size == 0:
        print(f"[notation] Full score MusicXML empty or missing")
        return None

    # Try MuseScore to produce .mscz
    try:
        result = subprocess.run(
            [MUSESCORE_PATH, "-o", str(mscz_path), str(xml_path)],
            capture_output=True, timeout=60
        )
        if result.returncode == 0 and mscz_path.exists() and mscz_path.stat().st_size > 0:
            print(f"[notation] Full score -> MSCZ {mscz_path}")
            return str(mscz_path)
        stderr_text = result.stderr.decode("utf-8", errors="replace").strip()
        if stderr_text:
            print(f"[notation] MuseScore full score stderr: {stderr_text[:300]}")
    except FileNotFoundError:
        print(f"[notation] MuseScore not found at {MUSESCORE_PATH}")
    except subprocess.TimeoutExpired:
        print(f"[notation] MuseScore timed out on full score")
    except Exception as e:
        print(f"[notation] MuseScore full score error: {e}")

    # Fallback: return MusicXML
    print(f"[notation] Full score -> MusicXML fallback {xml_path}")
    return str(xml_path)


def _render_with_musescore(xml_path, pdf_path):
    """Try to render PDF using MuseScore CLI. Tries exact MS4 path first."""
    commands_to_try = [MUSESCORE_PATH, "mscore", "mscore3", "musescore", "musescore3", "MuseScore4"]
    for cmd in commands_to_try:
        try:
            result = subprocess.run(
                [cmd, "-o", pdf_path, xml_path],
                capture_output=True, timeout=60
            )
            if result.returncode == 0 and Path(pdf_path).exists():
                return True
            # Log MuseScore failures for debugging
            stderr_text = result.stderr.decode("utf-8", errors="replace").strip()
            if stderr_text:
                print(f"[notation] MuseScore ({cmd}) stderr: {stderr_text[:300]}")
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            print(f"[notation] MuseScore ({cmd}) timed out on {Path(xml_path).name}")
            continue
        except Exception as e:
            print(f"[notation] MuseScore ({cmd}) error: {e}")
            continue
    return False


def _render_with_lilypond(score, pdf_path):
    """Try to render PDF using LilyPond via music21."""
    try:
        # music21 can write LilyPond directly
        ly_path = pdf_path.replace(".pdf", ".ly")
        score.write("lilypond", fp=ly_path)

        result = subprocess.run(
            ["lilypond", "--pdf", "-o", pdf_path.replace(".pdf", ""), ly_path],
            capture_output=True, timeout=60
        )
        # LilyPond outputs filename.pdf
        actual_pdf = pdf_path.replace(".pdf","") + ".pdf"
        if Path(actual_pdf).exists():
            import shutil
            shutil.move(actual_pdf, pdf_path)
            return True
    except Exception as e:
        print(f"[notation] LilyPond failed: {e}")
    return False
