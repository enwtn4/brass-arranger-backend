"""
Audio Analyzer
- BPM detection via librosa
- Key detection via librosa chroma
- Chord progression via chroma + template matching
- Melody extraction via basic_pitch
- Section detection (intro, verse, chorus, bridge, outro)
"""

import numpy as np
import sys
import librosa
import librosa.display
from pathlib import Path
import subprocess
import json
import tempfile
import os


# ── Chord templates (major/minor/dom7/min7) ──────────────────────────────────

CHORD_TEMPLATES = {}

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

def _build_templates():
    major  = np.array([1,0,0,0,1,0,0,1,0,0,0,0], dtype=float)
    minor  = np.array([1,0,0,1,0,0,0,1,0,0,0,0], dtype=float)
    dom7   = np.array([1,0,0,0,1,0,0,1,0,0,1,0], dtype=float)
    min7   = np.array([1,0,0,1,0,0,0,1,0,0,1,0], dtype=float)
    maj7   = np.array([1,0,0,0,1,0,0,1,0,0,0,1], dtype=float)

    for i, name in enumerate(NOTE_NAMES):
        for template, suffix in [(major,""), (minor,"m"), (dom7,"7"), (min7,"m7"), (maj7,"maj7")]:
            rolled = np.roll(template, i)
            CHORD_TEMPLATES[f"{name}{suffix}"] = rolled / rolled.sum()

_build_templates()


def detect_chord(chroma_frame):
    """Match a chroma vector to the closest chord template."""
    chroma_norm = chroma_frame / (chroma_frame.sum() + 1e-8)
    best_chord = "N"
    best_score = -1
    for chord_name, template in CHORD_TEMPLATES.items():
        score = np.dot(chroma_norm, template)
        if score > best_score:
            best_score = score
            best_chord = chord_name
    return best_chord


def detect_key(y, sr):
    """Detect musical key using chroma energy + Krumhansl-Schmuckler profiles."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    # Krumhansl-Schmuckler key profiles
    major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

    best_key = "C"
    best_mode = "major"
    best_corr = -999

    for i, note in enumerate(NOTE_NAMES):
        major_corr = np.corrcoef(np.roll(major_profile, i), chroma_mean)[0,1]
        minor_corr = np.corrcoef(np.roll(minor_profile, i), chroma_mean)[0,1]
        if major_corr > best_corr:
            best_corr = major_corr
            best_key = note
            best_mode = "major"
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = note
            best_mode = "minor"

    return best_key, best_mode


def detect_sections(y, sr, bpm):
    """
    Segment audio into sections using spectral novelty + beat structure.
    Returns list of dicts: {label, start_beat, end_beat, start_sec, end_sec}
    """
    # Novelty-based segmentation
    hop = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop, bpm=bpm)

    # Segment using recurrence matrix + novelty peak detection
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    R = librosa.segment.recurrence_matrix(mfcc, width=3, mode='affinity', sym=True)

    # Compute novelty curve: frame-wise self-similarity energy
    row_energy = np.array(R.sum(axis=1)).flatten()

    # Smooth to reduce frame-level noise
    n_frames = len(row_energy)
    win = max(5, n_frames // 20)
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win) / win
    smoothed = np.convolve(row_energy, kernel, mode='same')

    # Boundaries are at local minima of self-similarity (invert to find peaks)
    inverted = smoothed.max() - smoothed
    n_boundaries = min(7, max(1, n_frames // 10))

    if n_boundaries > 0 and n_frames > n_boundaries:
        peak_indices = np.argpartition(inverted, -n_boundaries)[-n_boundaries:]
        peak_indices = np.sort(peak_indices)
    else:
        peak_indices = np.array([], dtype=int)

    boundaries_sec = librosa.frames_to_time(peak_indices, sr=sr, hop_length=hop)
    duration = librosa.get_duration(y=y, sr=sr)
    boundaries_sec = np.concatenate([[0], boundaries_sec, [duration]])

    section_labels = ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"]

    sections = []
    for i in range(len(boundaries_sec) - 1):
        label = section_labels[i] if i < len(section_labels) else "section"
        start_sec = float(boundaries_sec[i])
        end_sec = float(boundaries_sec[i+1])
        # Convert to beats
        start_beat = int(round(float(start_sec * bpm / 60)))
        end_beat = int(round(float(end_sec * bpm / 60)))
        if end_beat > start_beat:
            sections.append({
                "label": label,
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
                "start_beat": start_beat,
                "end_beat": end_beat,
            })

    return sections


def extract_chord_progression(y, sr, sections):
    """
    Extract chord per beat, then simplify to per-section chord list.
    """
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)

    section_chords = []
    for section in sections:
        mask = (times >= section["start_sec"]) & (times < section["end_sec"])
        if not mask.any():
            section_chords.append({"section": section["label"], "chords": ["N"]})
            continue

        frames = chroma[:, mask]
        # Detect chord every ~2 beats worth of frames
        beat_frames_count = max(1, int(sr * 2 / (hop * 1)))
        chords_seen = []
        for start in range(0, frames.shape[1], beat_frames_count):
            chunk = frames[:, start:start+beat_frames_count].mean(axis=1)
            chord = detect_chord(chunk)
            if not chords_seen or chord != chords_seen[-1]:
                chords_seen.append(chord)

        section_chords.append({
            "section": section["label"],
            "chords": chords_seen[:8]  # cap per section
        })

    return section_chords


def separate_with_demucs(audio_path, output_dir):
    """
    Run Demucs to split audio into stems (vocals, drums, bass, other).
    Returns path to the vocals stem, or None on failure.
    The vocals stem gives basic_pitch the cleanest signal for melody extraction.
    """
    print(f"[analyzer] Running Demucs source separation...")
    try:
        from demucs.api import Separator

        separator = Separator(model="htdemucs", segment=None)
        wav_path = Path(audio_path)

        # Demucs expects WAV — convert if needed
        input_path = Path(output_dir) / "demucs_input.wav"
        y, sr = librosa.load(audio_path, sr=44100, mono=False)
        import soundfile as sf
        if y.ndim == 1:
            y = y.reshape(1, -1)
        else:
            y = y.reshape(y.shape[0], -1)
        sf.write(str(input_path), y.T, sr)

        # Run separation
        origin, separated = separator.separate_audio_file(str(input_path))

        # separated is a dict: {"vocals": tensor, "drums": tensor, "bass": tensor, "other": tensor}
        vocals_path = Path(output_dir) / "vocals.wav"
        vocals_tensor = separated["vocals"]

        # Save vocals stem
        sf.write(str(vocals_path), vocals_tensor.T.cpu().numpy(), 44100)
        print(f"[analyzer] Demucs vocals stem saved: {vocals_path}")

        # Clean up temp input
        input_path.unlink(missing_ok=True)

        return str(vocals_path)

    except Exception as e:
        print(f"[analyzer] Demucs failed: {e}")
        return None


def extract_melody_midi(audio_path, output_dir):
    """
    Use basic_pitch to extract melody as MIDI.
    Returns path to MIDI file or None on failure.
    """
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH

        out = Path(output_dir)

        # predict() returns (model_output, midi_data, note_events)
        # No file I/O = no emoji encoding crashes on Windows
        model_output, midi_data, note_events = predict(
            audio_path,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
        )

        # Save the MIDI data ourselves
        stem = Path(audio_path).stem
        midi_path = out / f"{stem}_basic_pitch.mid"
        midi_data.write(str(midi_path))

        if midi_path.exists():
            return str(midi_path)
    except Exception as e:
        print(f"[analyzer] basic_pitch failed: {e}")
    return None


def midi_to_note_sequence(midi_path):
    """
    Parse MIDI and return a simplified note sequence:
    list of {pitch, start_beat, duration_beats, velocity}
    """
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(midi_path)
        if not pm.instruments:
            return []

        # Use the instrument with most notes (likely melody)
        instr = max(pm.instruments, key=lambda i: len(i.notes))
        tempo = pm.estimate_tempo()

        notes = []
        for n in instr.notes:
            start_beat = n.start * tempo / 60.0
            dur_beats = (n.end - n.start) * tempo / 60.0
            notes.append({
                "pitch": int(n.pitch),
                "pitch_name": pretty_midi.note_number_to_name(n.pitch),
                "start_beat": round(start_beat, 3),
                "duration_beats": round(dur_beats, 3),
                "velocity": int(n.velocity),
            })

        return sorted(notes, key=lambda x: x["start_beat"])
    except Exception as e:
        print(f"[analyzer] MIDI parse failed: {e}")
        return []


def analyze_audio(audio_path):
    """
    Full audio analysis pipeline.
    Returns dict with all data needed for arrangement generation.
    """
    print(f"[analyzer] Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"[analyzer] Duration: {duration:.1f}s, sr={sr}")

    # BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = round(float(tempo), 1)
    print(f"[analyzer] BPM: {bpm}")

    # Key
    key_root, key_mode = detect_key(y, sr)
    key_str = f"{key_root} {key_mode}"
    print(f"[analyzer] Key: {key_str}")

    # Sections
    sections = detect_sections(y, sr, bpm)
    print(f"[analyzer] Sections: {[s['label'] for s in sections]}")

    # Chord progression per section
    chord_data = extract_chord_progression(y, sr, sections)

    # Spectral features for timbre/feel classification
    spectral_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    rms = float(librosa.feature.rms(y=y).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())

    # Classify feel
    if bpm < 70:
        feel = "slow_drag"
    elif bpm < 90:
        feel = "mid_tempo"
    elif bpm < 115:
        feel = "second_line"
    else:
        feel = "up_tempo_funk"

    # Extract melody MIDI — Demucs first for cleaner separation
    with tempfile.TemporaryDirectory() as tmpdir:
        # Try Demucs source separation to isolate vocals
        vocals_path = separate_with_demucs(audio_path, tmpdir)

        if vocals_path:
            # Run basic_pitch on the isolated vocals stem
            print(f"[analyzer] Running basic_pitch on vocals stem...")
            midi_path = extract_melody_midi(vocals_path, tmpdir)
        else:
            # Fall back to full mix
            print(f"[analyzer] Demucs unavailable, running basic_pitch on full mix...")
            midi_path = extract_melody_midi(audio_path, tmpdir)

        note_sequence = midi_to_note_sequence(midi_path) if midi_path else []

    print(f"[analyzer] Melody notes extracted: {len(note_sequence)}")

    return {
        "bpm": bpm,
        "key": key_str,
        "key_root": key_root,
        "key_mode": key_mode,
        "feel": feel,
        "duration": round(float(duration), 1),
        "sections": sections,
        "chord_progression": chord_data,
        "note_sequence": note_sequence,
        "spectral_centroid": round(spectral_centroid, 1),
        "rms": round(rms, 4),
    }
