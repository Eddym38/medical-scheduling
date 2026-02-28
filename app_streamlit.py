from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap

BASE_DIR = Path(__file__).resolve().parent
AGENT_DIR = BASE_DIR / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from sma import AgentOrdonnanceur, CoordinateurSMA, StatutPersonnel, Urgence


st.set_page_config(
    page_title="UrgenceFlow",
    page_icon="HF",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=DM+Serif+Display:ital@0;1&display=swap');
        :root {
            color-scheme: light;
            --bg-1: #f3f7fb;
            --bg-2: #e9f4f5;
            --surface: #ffffff;
            --surface-soft: #f9fcff;
            --line: #dce7ee;
            --ink: #18323b;
            --ink-soft: #415f69;
            --blue: #0e7490;
            --blue-dark: #12566d;
            --mint: #14b8a6;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(850px 420px at -10% -10%, #fff8ea 0%, rgba(255, 248, 234, 0) 65%),
                radial-gradient(900px 450px at 110% -15%, #e9fbf6 0%, rgba(233, 251, 246, 0) 65%),
                linear-gradient(135deg, var(--bg-1), var(--bg-2));
            color: var(--ink);
            font-family: "Manrope", sans-serif;
        }
        [data-testid="stSidebar"] {
            background: rgba(251, 254, 255, 0.92);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] * {
            color: var(--ink);
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            font-family: "DM Serif Display", serif;
            color: #11313c;
            letter-spacing: 0.2px;
        }
        .hero {
            border-radius: 18px;
            padding: 1rem 1.2rem;
            border: 1px solid rgba(255,255,255,0.65);
            background: linear-gradient(120deg, rgba(15,116,144,0.9), rgba(20,184,166,0.85));
            box-shadow: 0 12px 30px rgba(17, 76, 102, 0.22);
            color: white;
            margin-bottom: 0.8rem;
        }
        .hero h1 {
            color: white;
            margin: 0;
            font-size: 1.8rem;
            letter-spacing: 0.2px;
        }
        .hero p {
            margin-top: 0.24rem;
            margin-bottom: 0;
            color: rgba(255, 255, 255, 0.95);
            font-weight: 500;
        }
        .card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            box-shadow: 0 8px 22px rgba(26, 60, 80, 0.08);
        }
        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.5rem 0.8rem;
        }
        [data-testid="stMetricLabel"] {
            color: var(--ink-soft);
            font-weight: 600;
        }
        [data-testid="stMetricValue"] {
            color: #11313c;
            font-family: "DM Serif Display", serif;
        }
        div.stButton > button {
            border: none;
            border-radius: 12px;
            font-weight: 700;
            padding: 0.48rem 1rem;
            background: linear-gradient(120deg, var(--blue), var(--mint));
            color: white;
        }
        div.stButton > button:hover {
            background: linear-gradient(120deg, var(--blue-dark), var(--blue));
            color: white;
        }
        [data-baseweb="input"] input,
        textarea,
        [data-baseweb="select"] > div,
        [data-baseweb="popover"] * {
            background: var(--surface) !important;
            color: var(--ink) !important;
        }
        [data-baseweb="select"] > div {
            border: 1px solid var(--line) !important;
        }
        [data-testid="stNumberInput"] button {
            background: #ffffff !important;
            color: var(--ink) !important;
            border: 1px solid var(--line) !important;
        }
        [data-testid="stNumberInput"] button:hover {
            background: #f0f7fb !important;
            color: #11313c !important;
        }
        [data-testid="stNumberInput"] button svg {
            fill: var(--ink) !important;
            stroke: var(--ink) !important;
        }
        [data-testid="stDataFrame"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 12px;
        }
        [data-testid="stDataFrame"] * {
            color: var(--ink) !important;
        }
        [data-testid="stDataFrame"] div[role="grid"] {
            background: #ffffff !important;
        }
        [data-testid="stDataFrame"] {
            --gdg-bg-cell: #ffffff;
            --gdg-bg-cell-medium: #f7fbff;
            --gdg-bg-header: #eef6fb;
            --gdg-bg-header-hovered: #e3f0f8;
            --gdg-bg-header-has-selected: #d8ebf6;
            --gdg-border-color: #d5e4ec;
            --gdg-text-dark: #17323b;
            --gdg-text-medium: #17323b;
            --gdg-text-light: #4f6870;
            --gdg-text-header: #11313c;
            --gdg-accent-color: #0e7490;
            --gdg-accent-light: #d6eff6;
        }
        code {
            color: #0f4554 !important;
        }
        .small-note {
            color: var(--ink-soft);
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_simulation(
    nb_competences: int,
    nb_personnel: int,
    steps_par_phase: int,
    seed: Optional[int],
    preserve_preview: bool = False,
) -> None:
    st.session_state.sma = CoordinateurSMA(
        nb_competences=nb_competences,
        nb_personnel=nb_personnel,
        steps_par_phase=steps_par_phase,
        seed=seed,
    )
    st.session_state.last_result = None
    st.session_state.absence_events = []
    st.session_state.scenario_meta = {}
    st.session_state.live_events = []
    st.session_state.live_index = 0
    st.session_state.live_play = False
    st.session_state.live_processed = []
    st.session_state.live_arrivals_since_schedule = 0
    if not preserve_preview:
        st.session_state.scenario_preview = {}
    st.session_state.live_policy = "Chaque evenement"
    st.session_state.live_every_n = 3
    st.session_state.live_auto_reord = True
    st.session_state.live_trigger_critical = True
    st.session_state.live_ord_mode = "pipeline"
    st.session_state.live_ord_steps = 120
    st.session_state.live_respect_timing = True
    st.session_state.live_speed = 4.0
    st.session_state.live_projection_enabled = True
    st.session_state.live_next_tick_at = 0.0
    st.session_state.live_patients_state = []
    st.session_state.live_committed_plan = None
    st.session_state.live_executed_timeline = [[] for _ in range(nb_competences)]


def get_sma() -> CoordinateurSMA:
    return st.session_state.sma


def ensure_runtime_state() -> None:
    defaults = {
        "last_result": None,
        "absence_events": [],
        "scenario_meta": {},
        "live_events": [],
        "live_index": 0,
        "live_play": False,
        "live_processed": [],
        "live_arrivals_since_schedule": 0,
        "scenario_preview": {},
        "live_policy": "Chaque evenement",
        "live_every_n": 3,
        "live_auto_reord": True,
        "live_trigger_critical": True,
        "live_ord_mode": "pipeline",
        "live_ord_steps": 120,
        "live_respect_timing": True,
        "live_speed": 4.0,
        "live_projection_enabled": True,
        "live_next_tick_at": 0.0,
        "live_patients_state": [],
        "live_committed_plan": None,
        "live_executed_timeline": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def patients_to_df(patients: List[Any]) -> pd.DataFrame:
    rows = []
    for p in patients:
        rows.append(
            {
                "id": p.id,
                "nom": p.nom,
                "urgence": p.urgence.name,
                "arrivee": round(p.heure_arrivee, 2),
                "nb_ops": len(p.operations),
                "competences": ", ".join(f"C{i + 1}" for i in sorted(p.competences_requises)),
                "statut": p.statut,
            }
        )
    return pd.DataFrame(rows)


def personnel_to_df(personnel: List[Any]) -> pd.DataFrame:
    rows = []
    for p in personnel:
        rows.append(
            {
                "id": p.id,
                "nom": p.nom,
                "statut": p.statut.name,
                "competences": ", ".join(f"C{i + 1}" for i in p.competences),
                "charge": round(p.charge_travail, 2),
            }
        )
    return pd.DataFrame(rows)


def messages_to_df(messages: List[Any]) -> pd.DataFrame:
    rows = []
    for m in messages:
        rows.append(
            {
                "t": round(m.timestamp, 2),
                "type": m.type.value,
                "emetteur": m.emetteur,
                "destinataire": m.destinataire,
                "priorite": m.priorite,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("t", ascending=True).reset_index(drop=True)
    return df


def timeline_to_df(events: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for e in events:
        detail = ""
        if e.get("type") == "patient_arrival":
            payload = e.get("payload", {})
            detail = payload.get("nom", "patient")
        elif e.get("type") == "absence":
            payload = e.get("payload", {})
            detail = f"personnel_id={payload.get('personnel_id')}"
        rows.append(
            {
                "t": e.get("timestamp"),
                "event": e.get("type"),
                "detail": detail,
            }
        )
    return pd.DataFrame(rows)


def parse_operations_json(raw: str, nb_competences: int) -> List[List[int]]:
    data = json.loads(raw)
    if not isinstance(data, list) or not data:
        raise ValueError("Le JSON doit etre une liste d'operations non vide.")
    parsed: List[List[int]] = []
    for op in data:
        if not isinstance(op, list):
            raise ValueError("Chaque operation doit etre une liste.")
        if len(op) != nb_competences:
            raise ValueError(f"Chaque operation doit avoir {nb_competences} competences.")
        parsed.append([int(v) for v in op])
    return parsed


def random_operations(
    nb_competences: int,
    min_ops: int,
    max_ops: int,
    max_skills_per_op: int,
    max_duration: int,
) -> List[List[int]]:
    if min_ops > max_ops:
        min_ops, max_ops = max_ops, min_ops
    nb_ops = random.randint(min_ops, max_ops)
    ops: List[List[int]] = []
    for _ in range(nb_ops):
        vec = [0] * nb_competences
        k = random.randint(1, max(1, min(max_skills_per_op, nb_competences)))
        comps = random.sample(range(nb_competences), k=k)
        for comp_idx in comps:
            vec[comp_idx] = random.randint(1, max_duration)
        ops.append(vec)
    return ops


def planning_figure(solution: List[List[Any]], title: str):
    if not solution or not solution[0]:
        return None

    nb_skills = len(solution)
    horizon = max(len(skill) for skill in solution)
    patient_set = set()
    for skill_row in solution:
        for slot in skill_row:
            if slot is not None:
                patient_set.add(slot[0])

    patient_list = sorted(patient_set)
    patient_to_color = {p: i + 1 for i, p in enumerate(patient_list)}

    color_data: List[List[int]] = []
    op_data: List[List[Optional[int]]] = []
    for s in range(nb_skills):
        c_row = []
        o_row = []
        for t in range(horizon):
            slot = solution[s][t] if t < len(solution[s]) else None
            if slot is None:
                c_row.append(0)
                o_row.append(None)
            else:
                c_row.append(patient_to_color[slot[0]])
                o_row.append(slot[1] + 1)
        color_data.append(c_row)
        op_data.append(o_row)

    base = plt.get_cmap("Set3")
    colors = [(0.97, 0.99, 0.99, 1.0)] + [base(i % base.N) for i in range(len(patient_list))]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(max(7, horizon * 0.42), max(3.8, nb_skills * 0.62)))
    ax.imshow(color_data, cmap=cmap, aspect="auto", origin="upper")

    for s in range(nb_skills):
        for t in range(horizon):
            op_number = op_data[s][t]
            if op_number is not None:
                ax.text(t, s, str(op_number), ha="center", va="center", fontsize=8, color="#1b2932")

    ax.set_xticks(range(horizon))
    ax.set_yticks(range(nb_skills))
    ax.set_xlabel("Temps")
    ax.set_ylabel("Competences")
    ax.set_yticklabels([f"C{i + 1}" for i in range(nb_skills)])
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(which="major", axis="x", linewidth=0.3, color="#c8dce5")
    ax.tick_params(axis="both", which="both", length=0, colors="#24404a")

    patches = [mpatches.Patch(color=colors[patient_to_color[p]], label=f"P{p + 1}") for p in patient_list]
    if patches:
        ax.legend(handles=patches, title="Patients", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    return fig


def generate_arrival_times(
    nb_patients: int,
    mode: str,
    fixed_interval: float,
    min_interval: float,
    max_interval: float,
    start_time: float,
) -> List[float]:
    current_t = start_time
    arrivals: List[float] = []
    for _ in range(nb_patients):
        if mode == "Fixe":
            current_t += fixed_interval
        else:
            current_t += random.uniform(min_interval, max_interval)
        arrivals.append(current_t)
    return arrivals


def inject_absence_events(
    sma: CoordinateurSMA,
    nb_absences: int,
    start_t: float,
    end_t: float,
) -> List[Dict[str, Any]]:
    if nb_absences <= 0 or not sma.identificateur.registre_personnel:
        return []

    events: List[Dict[str, Any]] = []
    eligible_ids = [p.id for p in sma.identificateur.registre_personnel]
    nb_effective = min(nb_absences, len(eligible_ids))

    sampled_ids = random.sample(eligible_ids, k=nb_effective)
    for personnel_id in sampled_ids:
        t_evt = random.uniform(start_t, end_t)
        msg = sma.identificateur.signaler_absence(personnel_id=personnel_id, timestamp=t_evt)
        if msg is not None:
            sma._envoyer_message(msg)
            events.append(
                {
                    "timestamp": round(t_evt, 2),
                    "personnel_id": personnel_id,
                    "nom": msg.contenu.get("personnel_absent", f"ID {personnel_id}"),
                    "impact_critique": msg.contenu.get("impact_critique", False),
                }
            )

    events.sort(key=lambda x: x["timestamp"])
    if events:
        sma.timestamp = max(sma.timestamp, max(event["timestamp"] for event in events))
    return events


def build_generated_scenario(
    *,
    nb_competences: int,
    nb_personnel: int,
    steps_par_phase: int,
    seed: Optional[int],
    nb_patients: int,
    arrival_mode: str,
    fixed_interval: float,
    min_interval: float,
    max_interval: float,
    start_time: float,
    min_ops: int,
    max_ops: int,
    max_skills_per_op: int,
    max_duration: int,
    urgency_mode: str,
    fixed_urgency: str,
    simulate_absences: bool,
    nb_absences: int,
    absence_start: float,
    absence_end: float,
) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)

    init_simulation(
        nb_competences=nb_competences,
        nb_personnel=nb_personnel,
        steps_par_phase=steps_par_phase,
        seed=seed,
    )
    sma = get_sma()

    arrivals = generate_arrival_times(
        nb_patients=nb_patients,
        mode=arrival_mode,
        fixed_interval=fixed_interval,
        min_interval=min_interval,
        max_interval=max_interval,
        start_time=start_time,
    )

    ops_stats: List[int] = []
    for idx, t_arrivee in enumerate(arrivals, start=1):
        sma.timestamp = t_arrivee
        operations = random_operations(
            nb_competences=nb_competences,
            min_ops=min_ops,
            max_ops=max_ops,
            max_skills_per_op=max_skills_per_op,
            max_duration=max_duration,
        )
        ops_stats.append(len(operations))
        urgence = None
        if urgency_mode == "Fixe":
            urgence = Urgence[fixed_urgency]
        patient_obj = sma.simuler_arrivee_patient(
            operations=operations,
            urgence=urgence,
            nom=f"Patient_{idx:03d}",
        )
        _ensure_live_patient_state_from_event(
            nom=patient_obj.nom,
            urgence_name=patient_obj.urgence.name,
            operations=operations,
            arrivee=float(t_arrivee),
        )

    events = []
    if simulate_absences:
        start_evt = max(start_time, absence_start)
        end_evt = max(start_evt, absence_end)
        events = inject_absence_events(
            sma=sma,
            nb_absences=nb_absences,
            start_t=start_evt,
            end_t=end_evt,
        )

    st.session_state.absence_events = events
    summary = {
        "nb_patients": nb_patients,
        "nb_personnel": nb_personnel,
        "nb_competences": nb_competences,
        "arrivee_min": round(min(arrivals), 2) if arrivals else 0,
        "arrivee_max": round(max(arrivals), 2) if arrivals else 0,
        "ops_moyennes": round(sum(ops_stats) / len(ops_stats), 2) if ops_stats else 0,
        "nb_absences": len(events),
        "nb_evenements": nb_patients + len(events),
        "mode": "instant",
    }
    st.session_state.scenario_meta = summary
    return summary


def build_live_timeline(
    *,
    nb_competences: int,
    nb_personnel: int,
    steps_par_phase: int,
    seed: Optional[int],
    nb_patients: int,
    arrival_mode: str,
    fixed_interval: float,
    min_interval: float,
    max_interval: float,
    start_time: float,
    min_ops: int,
    max_ops: int,
    max_skills_per_op: int,
    max_duration: int,
    urgency_mode: str,
    fixed_urgency: str,
    simulate_absences: bool,
    nb_absences: int,
    absence_start: float,
    absence_end: float,
    preserve_preview: bool = False,
) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)

    init_simulation(
        nb_competences=nb_competences,
        nb_personnel=nb_personnel,
        steps_par_phase=steps_par_phase,
        seed=seed,
        preserve_preview=preserve_preview,
    )
    sma = get_sma()

    arrivals = generate_arrival_times(
        nb_patients=nb_patients,
        mode=arrival_mode,
        fixed_interval=fixed_interval,
        min_interval=min_interval,
        max_interval=max_interval,
        start_time=start_time,
    )

    live_events: List[Dict[str, Any]] = []
    ops_stats: List[int] = []
    for idx, t_arrivee in enumerate(arrivals, start=1):
        operations = random_operations(
            nb_competences=nb_competences,
            min_ops=min_ops,
            max_ops=max_ops,
            max_skills_per_op=max_skills_per_op,
            max_duration=max_duration,
        )
        ops_stats.append(len(operations))
        urgence_name = fixed_urgency if urgency_mode == "Fixe" else None
        live_events.append(
            {
                "timestamp": round(t_arrivee, 2),
                "type": "patient_arrival",
                "payload": {
                    "nom": f"Patient_{idx:03d}",
                    "operations": operations,
                    "urgence": urgence_name,
                },
            }
        )

    absence_events: List[Dict[str, Any]] = []
    if simulate_absences and sma.identificateur.registre_personnel and nb_absences > 0:
        eligible_ids = [p.id for p in sma.identificateur.registre_personnel]
        sample_count = min(nb_absences, len(eligible_ids))
        sampled_ids = random.sample(eligible_ids, k=sample_count)
        start_evt = max(start_time, absence_start)
        end_evt = max(start_evt, absence_end)
        for personnel_id in sampled_ids:
            t_evt = round(random.uniform(start_evt, end_evt), 2)
            payload = {"personnel_id": personnel_id}
            live_events.append(
                {
                    "timestamp": t_evt,
                    "type": "absence",
                    "payload": payload,
                }
            )
            absence_events.append(
                {"timestamp": t_evt, "personnel_id": personnel_id, "impact_critique": None}
            )

    live_events.sort(key=lambda e: (e["timestamp"], 0 if e["type"] == "patient_arrival" else 1))

    st.session_state.live_events = live_events
    st.session_state.live_index = 0
    st.session_state.live_play = False
    st.session_state.live_processed = []
    st.session_state.live_arrivals_since_schedule = 0
    st.session_state.absence_events = absence_events

    summary = {
        "nb_patients": nb_patients,
        "nb_personnel": nb_personnel,
        "nb_competences": nb_competences,
        "arrivee_min": round(min(arrivals), 2) if arrivals else 0,
        "arrivee_max": round(max(arrivals), 2) if arrivals else 0,
        "ops_moyennes": round(sum(ops_stats) / len(ops_stats), 2) if ops_stats else 0,
        "nb_absences": len(absence_events),
        "nb_evenements": len(live_events),
        "mode": "timeline",
    }
    st.session_state.scenario_meta = summary
    return summary


def _schedule_now_if_needed(
    sma: CoordinateurSMA,
    mode: str,
    n_steps: int,
) -> Optional[Dict[str, Any]]:
    if not sma.accueil.patients_en_attente:
        return None
    result = sma.lancer_ordonnancement(mode=mode, n_steps=n_steps, verbose=False)
    st.session_state.last_result = result
    return result


def compute_live_projection(
    sma: CoordinateurSMA,
    mode: str,
    n_steps: int,
) -> Optional[Dict[str, Any]]:
    """
    Calcule un planning "what-if" a partir des patients en attente,
    sans modifier l'etat accueil/en_cours.
    """
    _advance_live_progress_to(sma.timestamp)
    matrix, active = build_live_remaining_matrix()
    if not matrix:
        return None

    projection = AgentOrdonnanceur(nom="ProjectionLive", steps_par_phase=sma.ordonnanceur.steps_par_phase)
    result = projection.ordonnancer(
        competence_matrix=matrix,
        mode=mode,
        n_steps=n_steps,
        verbose=False,
    )

    planning_local = result.get("planning", [])
    planning_global: List[List[Any]] = []
    for row in planning_local:
        new_row = []
        for slot in row:
            if slot is None:
                new_row.append(None)
                continue
            local_idx, local_op = slot
            patient_state = active[int(local_idx)]
            pid = int(patient_state["pid"])
            global_op = int(patient_state["completed_ops"]) + int(local_op)
            new_row.append((pid, global_op))
        planning_global.append(new_row)

    return {
        "makespan": result.get("makespan"),
        "planning": planning_global,
        "solution": result.get("solution"),
        "duree_calcul": result.get("duree_calcul"),
        "historique": result.get("historique"),
    }


def compute_global_preview_from_timeline(
    *,
    timeline_events: List[Dict[str, Any]],
    steps_par_phase: int,
    ord_mode: str,
    ord_steps: int,
) -> Optional[Dict[str, Any]]:
    """Calcule un planning global sur tous les patients du scenario."""
    matrix: List[List[List[int]]] = []
    for event in timeline_events:
        if event.get("type") != "patient_arrival":
            continue
        payload = event.get("payload", {})
        operations = payload.get("operations")
        if isinstance(operations, list):
            matrix.append(operations)

    if not matrix:
        return None

    ord_agent = AgentOrdonnanceur(nom="PreviewGlobal", steps_par_phase=steps_par_phase)
    return ord_agent.ordonnancer(
        competence_matrix=matrix,
        mode=ord_mode,
        n_steps=ord_steps,
        verbose=False,
    )


def count_patients_in_planning(solution: List[List[Any]]) -> int:
    patient_set = set()
    for row in solution or []:
        for slot in row:
            if slot is not None:
                patient_set.add(slot[0])
    return len(patient_set)


def _ensure_live_patient_state_from_event(
    *,
    nom: str,
    urgence_name: str,
    operations: List[List[int]],
    arrivee: float,
) -> None:
    states: List[Dict[str, Any]] = st.session_state.get("live_patients_state", [])
    if any(s.get("nom") == nom for s in states):
        return

    urgence = Urgence[urgence_name]
    states.append(
        {
            "pid": len(states),
            "nom": nom,
            "urgence_name": urgence.name,
            "urgence_level": urgence.value,
            "arrivee": float(arrivee),
            "operations": operations,
            "completed_ops": 0,
        }
    )
    st.session_state.live_patients_state = states


def _extract_op_end_times_global(planning: List[List[Any]]) -> Dict[tuple[int, int], int]:
    op_end: Dict[tuple[int, int], int] = {}
    for row in planning or []:
        for t, slot in enumerate(row):
            if slot is None:
                continue
            pid, op_idx = slot
            key = (int(pid), int(op_idx))
            op_end[key] = max(op_end.get(key, 0), t + 1)
    return op_end


def _ensure_executed_timeline_shape(nb_rows: int) -> List[List[Any]]:
    timeline = st.session_state.get("live_executed_timeline")
    if not isinstance(timeline, list):
        timeline = []
    while len(timeline) < nb_rows:
        timeline.append([])
    for idx, row in enumerate(timeline):
        if not isinstance(row, list):
            timeline[idx] = []
    st.session_state.live_executed_timeline = timeline
    return timeline


def _materialize_plan_progress(timestamp: float) -> None:
    """Ajoute a la timeline executee la partie du plan deja consommee par le temps."""
    plan = st.session_state.get("live_committed_plan")
    if not isinstance(plan, dict):
        return

    planning = plan.get("planning", [])
    if not isinstance(planning, list) or not planning:
        return

    start_t = float(plan.get("start_t", 0.0))
    elapsed = max(0, int(float(timestamp) - start_t))
    horizon = max((len(r) for r in planning), default=0)
    if horizon <= 0:
        return

    target = min(elapsed, horizon)
    applied = int(plan.get("applied_ticks", 0))
    if target <= applied:
        return

    nb_rows = max(len(planning), int(get_sma().nb_competences))
    timeline = _ensure_executed_timeline_shape(nb_rows)

    for row_idx in range(nb_rows):
        src = planning[row_idx] if row_idx < len(planning) else []
        dst = timeline[row_idx]
        for t in range(applied, target):
            dst.append(src[t] if t < len(src) else None)

    plan["applied_ticks"] = target
    st.session_state.live_executed_timeline = timeline
    st.session_state.live_committed_plan = plan


def _advance_live_progress_to(timestamp: float) -> None:
    plan = st.session_state.get("live_committed_plan")
    if not isinstance(plan, dict):
        return

    _materialize_plan_progress(timestamp)

    start_t = float(plan.get("start_t", 0.0))
    elapsed = max(0, int(float(timestamp) - start_t))
    op_end: Dict[tuple[int, int], int] = plan.get("op_end_times", {})
    states: List[Dict[str, Any]] = st.session_state.get("live_patients_state", [])

    for state in states:
        pid = int(state.get("pid", -1))
        completed = int(state.get("completed_ops", 0))
        total_ops = len(state.get("operations", []))
        while completed < total_ops:
            end_tick = op_end.get((pid, completed))
            if end_tick is None or end_tick > elapsed:
                break
            completed += 1
        state["completed_ops"] = completed

    st.session_state.live_patients_state = states


def get_live_remaining_patients() -> List[Dict[str, Any]]:
    sma = get_sma()
    now_t = float(sma.timestamp)
    states: List[Dict[str, Any]] = st.session_state.get("live_patients_state", [])
    active = []
    for s in states:
        if float(s.get("arrivee", 0.0)) <= now_t and int(s.get("completed_ops", 0)) < len(
            s.get("operations", [])
        ):
            active.append(s)
    return active


def build_live_remaining_matrix() -> tuple[List[List[List[int]]], List[Dict[str, Any]]]:
    active = get_live_remaining_patients()
    active.sort(key=lambda s: (s["urgence_level"], s["arrivee"], s["pid"]))

    matrix: List[List[List[int]]] = []
    for s in active:
        completed = int(s.get("completed_ops", 0))
        matrix.append(s.get("operations", [])[completed:])
    return matrix, active


def schedule_live_progress(
    *,
    timestamp: float,
    mode: str,
    n_steps: int,
    steps_par_phase: int,
) -> Optional[Dict[str, Any]]:
    _advance_live_progress_to(timestamp)
    matrix, active = build_live_remaining_matrix()
    if not matrix:
        return None

    ord_agent = AgentOrdonnanceur(nom="LiveProgress", steps_par_phase=steps_par_phase)
    result = ord_agent.ordonnancer(
        competence_matrix=matrix,
        mode=mode,
        n_steps=n_steps,
        verbose=False,
    )

    planning_local = result.get("planning", [])
    planning_global: List[List[Any]] = []
    for row in planning_local:
        new_row = []
        for slot in row:
            if slot is None:
                new_row.append(None)
                continue
            local_idx, local_op = slot
            patient_state = active[int(local_idx)]
            pid = int(patient_state["pid"])
            global_op = int(patient_state["completed_ops"]) + int(local_op)
            new_row.append((pid, global_op))
        planning_global.append(new_row)

    op_end_times = _extract_op_end_times_global(planning_global)
    st.session_state.live_committed_plan = {
        "start_t": float(timestamp),
        "planning": planning_global,
        "op_end_times": op_end_times,
        "makespan": result.get("makespan"),
        "applied_ticks": 0,
    }

    result_global = {
        "makespan": result.get("makespan"),
        "planning": planning_global,
        "solution": result.get("solution"),
        "duree_calcul": result.get("duree_calcul"),
        "historique": result.get("historique"),
    }
    st.session_state.last_result = result_global
    return result_global


def get_live_queue_df() -> pd.DataFrame:
    active = get_live_remaining_patients()
    rows = []
    for s in active:
        total_ops = len(s.get("operations", []))
        done = int(s.get("completed_ops", 0))
        rows.append(
            {
                "pid": s.get("pid"),
                "nom": s.get("nom"),
                "urgence": s.get("urgence_name"),
                "arrivee": round(float(s.get("arrivee", 0.0)), 2),
                "ops_terminees": done,
                "ops_restantes": max(0, total_ops - done),
            }
        )
    return pd.DataFrame(rows)


def _copy_planning(planning: List[List[Any]]) -> List[List[Any]]:
    return [list(row) for row in planning or []]


def _concat_plannings(left: List[List[Any]], right: List[List[Any]]) -> List[List[Any]]:
    nb_rows = max(len(left), len(right))
    merged: List[List[Any]] = []
    for row_idx in range(nb_rows):
        l_row = list(left[row_idx]) if row_idx < len(left) else []
        r_row = list(right[row_idx]) if row_idx < len(right) else []
        merged.append(l_row + r_row)
    return merged


def build_live_graph_result(
    *,
    sma: CoordinateurSMA,
    mode: str,
    n_steps: int,
    projection_enabled: bool,
) -> Optional[Dict[str, Any]]:
    """
    Construit un graphe evolutif:
    - prefixe: partie deja executee
    - suffixe: projection (ou reste du plan valide en cours)
    """
    _advance_live_progress_to(float(sma.timestamp))

    executed = _copy_planning(st.session_state.get("live_executed_timeline", []))
    future: List[List[Any]] = []
    title = "Live evolutif"

    if projection_enabled:
        projection = compute_live_projection(
            sma=sma,
            mode=mode,
            n_steps=n_steps,
        )
        if isinstance(projection, dict):
            future = _copy_planning(projection.get("planning", []))
            title = f"Live evolutif (fait + projection) - CMax {projection.get('makespan', '?')}"

    if not future:
        plan = st.session_state.get("live_committed_plan")
        if isinstance(plan, dict):
            planning = plan.get("planning", [])
            if isinstance(planning, list) and planning:
                horizon = max((len(r) for r in planning), default=0)
                start_idx = min(int(plan.get("applied_ticks", 0)), horizon)
                future = []
                for row in planning:
                    if start_idx < len(row):
                        future.append(list(row[start_idx:]))
                    else:
                        future.append([])
                title = f"Live evolutif (fait + plan valide)"

    merged = _concat_plannings(executed, future)
    horizon = max((len(r) for r in merged), default=0)
    if horizon <= 0:
        return None

    return {
        "planning": merged,
        "makespan": horizon,
        "title": title,
    }


def process_next_live_event(
    *,
    scheduling_policy: str,
    schedule_every_n: int,
    auto_reord_on_absence: bool,
    critical_trigger: bool,
    ord_mode: str,
    ord_steps: int,
) -> Optional[Dict[str, Any]]:
    sma = get_sma()
    events = st.session_state.get("live_events", [])
    index = st.session_state.get("live_index", 0)
    if index >= len(events):
        st.session_state.live_play = False
        return None

    event = events[index]
    st.session_state.live_index = index + 1
    sma.timestamp = float(event["timestamp"])
    _advance_live_progress_to(sma.timestamp)

    event_log: Dict[str, Any] = {
        "timestamp": event["timestamp"],
        "type": event["type"],
        "details": "",
        "scheduled": False,
        "cmax": None,
        "next_t": None,
    }

    if event["type"] == "patient_arrival":
        payload = event["payload"]
        urgence_name = payload.get("urgence")
        urgence = Urgence[urgence_name] if urgence_name else None
        patient_obj = sma.simuler_arrivee_patient(
            operations=payload["operations"],
            urgence=urgence,
            nom=payload.get("nom"),
        )
        _ensure_live_patient_state_from_event(
            nom=payload.get("nom", f"Patient_{index+1:03d}"),
            urgence_name=patient_obj.urgence.name,
            operations=payload["operations"],
            arrivee=float(event["timestamp"]),
        )
        st.session_state.live_arrivals_since_schedule += 1
        event_log["details"] = f"{payload.get('nom', 'patient')} ({patient_obj.urgence.name})"

        should_schedule = False
        if scheduling_policy == "Chaque evenement":
            should_schedule = True
        elif scheduling_policy == "Toutes les N arrivees":
            if st.session_state.live_arrivals_since_schedule >= max(1, schedule_every_n):
                should_schedule = True
        if critical_trigger and patient_obj.urgence == Urgence.CRITIQUE:
            should_schedule = True
            event_log["details"] += " [TRIGGER CRITIQUE]"

        if should_schedule:
            result = schedule_live_progress(
                timestamp=sma.timestamp,
                mode=ord_mode,
                n_steps=ord_steps,
                steps_par_phase=sma.ordonnanceur.steps_par_phase,
            )
            st.session_state.live_arrivals_since_schedule = 0
            if result is not None:
                event_log["scheduled"] = True
                event_log["cmax"] = result["makespan"]

    elif event["type"] == "absence":
        payload = event["payload"]
        personnel_id = int(payload["personnel_id"])
        msg = sma.identificateur.signaler_absence(personnel_id=personnel_id, timestamp=sma.timestamp)
        if msg is not None:
            sma._envoyer_message(msg)
            event_log["details"] = msg.contenu.get("personnel_absent", f"ID {personnel_id}")
            for item in st.session_state.get("absence_events", []):
                if item.get("timestamp") == event["timestamp"] and item.get("personnel_id") == personnel_id:
                    item["impact_critique"] = bool(msg.contenu.get("impact_critique"))
                    break
            if auto_reord_on_absence:
                result = schedule_live_progress(
                    timestamp=sma.timestamp,
                    mode=ord_mode,
                    n_steps=ord_steps,
                    steps_par_phase=sma.ordonnanceur.steps_par_phase,
                )
                if result is not None:
                    event_log["scheduled"] = True
                    event_log["cmax"] = result["makespan"]
        else:
            event_log["details"] = f"ID {personnel_id}"

    st.session_state.live_processed.append(event_log)

    new_index = st.session_state.get("live_index", 0)
    if new_index < len(events):
        event_log["next_t"] = events[new_index]["timestamp"]

    if st.session_state.live_index >= len(events):
        st.session_state.live_play = False

    return event_log


def run_live_until_end(
    *,
    scheduling_policy: str,
    schedule_every_n: int,
    auto_reord_on_absence: bool,
    critical_trigger: bool,
    ord_mode: str,
    ord_steps: int,
) -> Optional[Dict[str, Any]]:
    safety = 0
    while (
        st.session_state.get("live_index", 0) < len(st.session_state.get("live_events", []))
        and safety < 100000
    ):
        process_next_live_event(
            scheduling_policy=scheduling_policy,
            schedule_every_n=schedule_every_n,
            auto_reord_on_absence=auto_reord_on_absence,
            critical_trigger=critical_trigger,
            ord_mode=ord_mode,
            ord_steps=ord_steps,
        )
        safety += 1

    # Garantit un resultat final exploitable meme en politique "Manuel"
    sma = get_sma()
    result = schedule_live_progress(
        timestamp=sma.timestamp,
        mode=ord_mode,
        n_steps=ord_steps,
        steps_par_phase=sma.ordonnanceur.steps_par_phase,
    )
    if result is not None:
        st.session_state.live_processed.append(
            {
                "timestamp": round(get_sma().timestamp, 2),
                "type": "final_schedule",
                "details": "Ordonnancement final force",
                "scheduled": True,
                "cmax": result["makespan"],
            }
        )
    return st.session_state.get("last_result")


@st.fragment(run_every=0.12)
def live_autoplay_fragment() -> None:
    """Tick autoplay sans recharger la page navigateur."""
    if not st.session_state.get("live_play", False):
        return

    events = st.session_state.get("live_events", [])
    index = int(st.session_state.get("live_index", 0))
    if index >= len(events):
        st.session_state.live_play = False
        return

    now = time.time()
    next_due = float(st.session_state.get("live_next_tick_at", 0.0))
    if now < next_due:
        return

    current_t = float(events[index]["timestamp"])

    process_next_live_event(
        scheduling_policy=str(st.session_state.get("live_policy", "Chaque evenement")),
        schedule_every_n=int(st.session_state.get("live_every_n", 3)),
        auto_reord_on_absence=bool(st.session_state.get("live_auto_reord", True)),
        critical_trigger=bool(st.session_state.get("live_trigger_critical", True)),
        ord_mode=str(st.session_state.get("live_ord_mode", "pipeline")),
        ord_steps=int(st.session_state.get("live_ord_steps", 120)),
    )

    events = st.session_state.get("live_events", [])
    new_index = int(st.session_state.get("live_index", 0))
    if new_index >= len(events):
        st.session_state.live_play = False
        return

    respect_timing = bool(st.session_state.get("live_respect_timing", True))
    speed = max(0.5, float(st.session_state.get("live_speed", 4.0)))
    if respect_timing:
        next_t = float(events[new_index]["timestamp"])
        dt_sim = max(0.0, next_t - current_t)
        delay = max(0.08, min(2.0, dt_sim / speed))
    else:
        delay = max(0.08, min(1.0, 0.35 / speed))

    st.session_state.live_next_tick_at = now + delay
    st.rerun(scope="app")


def main() -> None:
    inject_css()
    st.markdown(
        """
        <div class="hero">
          <h1>UrgenceFlow Control Room</h1>
          <p>Simulation d'une journee d'urgences: arrivees patients, disponibilites medecins et planning dynamique.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Generateur de scenario")
        nb_competences = st.number_input("Nb competences (services)", min_value=2, max_value=16, value=6, step=1)
        nb_personnel = st.number_input("Nb medecins/personnel", min_value=1, max_value=120, value=12, step=1)
        nb_patients = st.number_input("Nb patients a simuler", min_value=1, max_value=500, value=30, step=1)
        steps_par_phase = st.number_input("Steps par phase (pipeline)", min_value=1, max_value=1000, value=50, step=1)
        seed_txt = st.text_input("Seed (optionnel)", value="42")
        seed = int(seed_txt) if seed_txt.strip() else None

        st.subheader("Arrivees patients")
        start_time = st.number_input("Heure debut (t0)", min_value=0.0, max_value=1000.0, value=0.0, step=0.5)
        arrival_mode = st.radio("Intervalle entre 2 arrivees", ["Fixe", "Aleatoire"], horizontal=True)
        if arrival_mode == "Fixe":
            fixed_interval = st.slider("Intervalle fixe", min_value=0.1, max_value=60.0, value=3.0, step=0.1)
            min_interval = 0.5
            max_interval = 3.0
        else:
            min_interval, max_interval = st.slider(
                "Intervalle min/max",
                min_value=0.1,
                max_value=60.0,
                value=(0.8, 4.0),
                step=0.1,
            )
            fixed_interval = 2.0

        st.subheader("Profil patients")
        min_ops, max_ops = st.slider("Operations par patient (min/max)", 1, 12, (1, 4))
        max_skills_per_op = st.slider("Competences max par operation", 1, int(nb_competences), min(3, int(nb_competences)))
        max_duration = st.slider("Duree max par competence", 1, 12, 4)
        urgency_mode = st.radio("Urgence", ["Aleatoire", "Fixe"], horizontal=True)
        fixed_urgency = st.selectbox("Urgence fixe", [u.name for u in Urgence], index=2, disabled=urgency_mode != "Fixe")

        st.subheader("Absences pendant la journee")
        simulate_absences = st.checkbox("Simuler des absences", value=True)
        nb_absences = st.number_input(
            "Nb absences",
            min_value=0,
            max_value=int(nb_personnel),
            value=min(2, int(nb_personnel)),
        )
        absence_start, absence_end = st.slider(
            "Fenetre temporelle des absences",
            min_value=0.0,
            max_value=300.0,
            value=(10.0, 80.0),
            step=0.5,
            disabled=not simulate_absences,
        )

        st.subheader("Regles ordonnancement (preview + live)")
        gen_policy = st.selectbox(
            "Politique auto",
            ["Chaque evenement", "Toutes les N arrivees", "Manuel"],
            index=1,
        )
        gen_every_n = st.number_input("N arrivees", min_value=1, max_value=50, value=3)
        gen_auto_reord = st.checkbox("Auto re-ordonnancement sur absence", value=True)
        gen_trigger_critical = st.checkbox("Ordonnancer immediatement si CRITIQUE", value=True)
        gen_ord_mode = st.selectbox("Mode ordonnancement", ["pipeline", "parallele"], index=0)
        gen_ord_steps = st.number_input(
            "Iterations ordonnancement",
            min_value=1,
            max_value=2000,
            value=120,
        )

        if "sma" not in st.session_state:
            init_simulation(
                nb_competences=int(nb_competences),
                nb_personnel=int(nb_personnel),
                steps_par_phase=int(steps_par_phase),
                seed=seed,
            )
        ensure_runtime_state()

        scenario_args = dict(
            nb_competences=int(nb_competences),
            nb_personnel=int(nb_personnel),
            steps_par_phase=int(steps_par_phase),
            seed=seed,
            nb_patients=int(nb_patients),
            arrival_mode=arrival_mode,
            fixed_interval=float(fixed_interval),
            min_interval=float(min_interval),
            max_interval=float(max_interval),
            start_time=float(start_time),
            min_ops=int(min_ops),
            max_ops=int(max_ops),
            max_skills_per_op=int(max_skills_per_op),
            max_duration=int(max_duration),
            urgency_mode=urgency_mode,
            fixed_urgency=fixed_urgency,
            simulate_absences=simulate_absences,
            nb_absences=int(nb_absences),
            absence_start=float(absence_start),
            absence_end=float(absence_end),
        )

        if st.button("Generer scenario (resume + live replay)"):
            summary = build_live_timeline(**scenario_args, preserve_preview=False)
            st.session_state.live_policy = gen_policy
            st.session_state.live_every_n = int(gen_every_n)
            st.session_state.live_auto_reord = bool(gen_auto_reord)
            st.session_state.live_trigger_critical = bool(gen_trigger_critical)
            st.session_state.live_ord_mode = gen_ord_mode
            st.session_state.live_ord_steps = int(gen_ord_steps)

            global_result = compute_global_preview_from_timeline(
                timeline_events=st.session_state.get("live_events", []),
                steps_par_phase=int(steps_par_phase),
                ord_mode=gen_ord_mode,
                ord_steps=int(gen_ord_steps),
            )
            preview_result = run_live_until_end(
                scheduling_policy=gen_policy,
                schedule_every_n=int(gen_every_n),
                auto_reord_on_absence=gen_auto_reord,
                critical_trigger=gen_trigger_critical,
                ord_mode=gen_ord_mode,
                ord_steps=int(gen_ord_steps),
            )
            preview_processed = list(st.session_state.get("live_processed", []))
            st.session_state.scenario_preview = {
                "result_global": global_result,
                "result_last_live": preview_result,
                "processed": preview_processed,
                "policy": gen_policy,
                "ord_mode": gen_ord_mode,
                "ord_steps": int(gen_ord_steps),
            }

            build_live_timeline(**scenario_args, preserve_preview=True)
            st.session_state.scenario_meta.update(
                {
                    "preview_cmax": (
                        global_result.get("makespan")
                        if isinstance(global_result, dict) and global_result
                        else None
                    ),
                    "preview_policy": gen_policy,
                    "preview_scope": "global",
                }
            )
            st.success(
                f"Scenario genere: {summary['nb_evenements']} evenements. "
                f"Resume calcule, pret pour replay live."
            )

        if st.button("Reinitialiser simulation vide"):
            init_simulation(
                nb_competences=int(nb_competences),
                nb_personnel=int(nb_personnel),
                steps_par_phase=int(steps_par_phase),
                seed=seed,
            )
            st.success("Simulation reinitialisee.")

    sma = get_sma()
    live_queue_count = len(get_live_remaining_patients())
    completed_patients = sum(
        1
        for s in st.session_state.get("live_patients_state", [])
        if int(s.get("completed_ops", 0)) >= len(s.get("operations", []))
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Patients en attente", live_queue_count)
    m2.metric("Patients termines", completed_patients)
    m3.metric(
        "Personnel disponible",
        sum(1 for p in sma.identificateur.registre_personnel if p.statut == StatutPersonnel.DISPONIBLE),
    )
    m4.metric("Messages", len(sma.messages_echanges))
    m5.metric("Temps simule", round(sma.timestamp, 2))

    tab_global, tab_live, tab_patients, tab_staff, tab_schedule, tab_logs = st.tabs(
        ["Scenario", "Simulation live", "Patients", "Personnel", "Ordonnancement", "Messages"]
    )

    with tab_global:
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Resume scenario")
            meta = st.session_state.get("scenario_meta", {})
            preview = st.session_state.get("scenario_preview", {})
            if meta:
                c1, c2, c3 = st.columns(3)
                c1.metric("Patients", meta.get("nb_patients", 0))
                c2.metric("Pers./medecins", meta.get("nb_personnel", 0))
                c3.metric("Competences", meta.get("nb_competences", 0))
                st.caption(
                    f"Arrivees entre t={meta.get('arrivee_min', 0)} et t={meta.get('arrivee_max', 0)} | "
                    f"Ops moyennes/patient: {meta.get('ops_moyennes', 0)} | "
                    f"Absences: {meta.get('nb_absences', 0)}"
                )
                if meta.get("preview_cmax") is not None:
                    st.metric("CMax planning global", meta.get("preview_cmax"))
                    st.caption(f"Politique utilisee: {meta.get('preview_policy', '-')}")
            else:
                st.info("Genere un scenario dans la barre latérale.")
            st.markdown("</div>", unsafe_allow_html=True)

            if preview and isinstance(preview.get("result_global"), dict):
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Planning global scenario (tous les patients)")
                result = preview["result_global"]
                n_pat_preview = count_patients_in_planning(result.get("planning", []))
                st.caption(f"Patients visibles dans ce planning: {n_pat_preview}")
                fig_preview = planning_figure(
                    result.get("planning", []),
                    title=f"Global - CMax {result.get('makespan', '?')}",
                )
                if fig_preview is not None:
                    st.pyplot(fig_preview, use_container_width=True)
                else:
                    st.caption("Aucun planning disponible dans le resume.")
                st.markdown("</div>", unsafe_allow_html=True)

            current_result = st.session_state.get("last_result")
            if isinstance(current_result, dict):
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Planning courant (etat simulation)")
                fig_current = planning_figure(
                    current_result.get("planning", []),
                    title=f"Courant - CMax {current_result.get('makespan', '?')}",
                )
                if fig_current is not None:
                    st.pyplot(fig_current, use_container_width=True)
                else:
                    st.caption("Aucun planning courant disponible.")
                st.markdown("</div>", unsafe_allow_html=True)
            elif sma.accueil.patients_en_attente:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Projection courante (file d'attente)")
                proj_now = compute_live_projection(
                    sma=sma,
                    mode=str(st.session_state.get("live_ord_mode", "pipeline")),
                    n_steps=int(st.session_state.get("live_ord_steps", 120)),
                )
                if isinstance(proj_now, dict):
                    fig_proj_now = planning_figure(
                        proj_now.get("planning", []),
                        title=f"Projection - CMax {proj_now.get('makespan', '?')}",
                    )
                    if fig_proj_now is not None:
                        st.pyplot(fig_proj_now, use_container_width=True)
                else:
                    st.caption("Projection indisponible.")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("File d'attente")
            st.caption("Patients non encore integres dans un planning valide.")
            st.dataframe(get_live_queue_df(), use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Absences simulees")
            events = st.session_state.get("absence_events", [])
            if events:
                st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)
            else:
                st.caption("Aucune absence simulee pour ce scenario.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Etat personnel")
            st.dataframe(personnel_to_df(sma.identificateur.registre_personnel), use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab_live:
        st.subheader("Lecture temporelle des evenements")
        events = st.session_state.get("live_events", [])
        live_index = int(st.session_state.get("live_index", 0))
        remaining = max(0, len(events) - live_index)

        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Evenements total", len(events))
        p2.metric("Traites", live_index)
        p3.metric("Restants", remaining)
        next_t = events[live_index]["timestamp"] if live_index < len(events) else "-"
        p4.metric("Prochain t", next_t)
        p5.metric("File attente", len(get_live_remaining_patients()))

        with st.expander("Parametres live", expanded=False):
            scheduling_policy = st.selectbox(
                "Politique d'ordonnancement live",
                ["Manuel", "Chaque evenement", "Toutes les N arrivees"],
                key="live_policy",
            )
            schedule_every_n = st.number_input(
                "N arrivees avant ordonnancement (si politique N)",
                min_value=1,
                max_value=50,
                key="live_every_n",
            )
            auto_reord_on_absence = st.checkbox(
                "Re-ordonnancer automatiquement sur absence",
                key="live_auto_reord",
            )
            trigger_critical = st.checkbox(
                "Ordonnancer immediatement si CRITIQUE",
                key="live_trigger_critical",
            )
            ord_mode_live = st.selectbox(
                "Mode ordonnancement auto",
                ["pipeline", "parallele"],
                key="live_ord_mode",
            )
            ord_steps_live = st.number_input(
                "Iterations auto",
                min_value=1,
                max_value=2000,
                key="live_ord_steps",
            )
            projection_enabled = st.checkbox(
                "Projection planning live (sans valider)",
                key="live_projection_enabled",
            )
            respect_timing = st.checkbox(
                "Respecter les ecarts temporels des arrivees",
                key="live_respect_timing",
            )
            speed_factor = st.number_input(
                "Vitesse de lecture (x)",
                min_value=0.5,
                max_value=20.0,
                value=float(st.session_state.get("live_speed", 4.0)),
                step=0.5,
                key="live_speed",
            )

        events = st.session_state.get("live_events", [])
        live_index = int(st.session_state.get("live_index", 0))
        remaining = max(0, len(events) - live_index)

        bar1, bar2, bar3, bar4, bar5 = st.columns(5)
        with bar1:
            if st.button("Avancer +1", disabled=remaining == 0):
                process_next_live_event(
                    scheduling_policy=scheduling_policy,
                    schedule_every_n=int(schedule_every_n),
                    auto_reord_on_absence=auto_reord_on_absence,
                    critical_trigger=trigger_critical,
                    ord_mode=ord_mode_live,
                    ord_steps=int(ord_steps_live),
                )
                st.rerun()
        with bar2:
            if st.button("Auto-play", disabled=remaining == 0):
                st.session_state.live_play = True
                st.session_state.live_next_tick_at = 0.0
                st.rerun()
        with bar3:
            if st.button("Stop"):
                st.session_state.live_play = False
        with bar4:
            if st.button("Executer jusqu'au bout", disabled=remaining == 0):
                run_live_until_end(
                    scheduling_policy=scheduling_policy,
                    schedule_every_n=int(schedule_every_n),
                    auto_reord_on_absence=auto_reord_on_absence,
                    critical_trigger=trigger_critical,
                    ord_mode=ord_mode_live,
                    ord_steps=int(ord_steps_live),
                )
                st.session_state.live_play = False
                st.rerun()
        with bar5:
            if st.button("Ordonnancer maintenant"):
                result = schedule_live_progress(
                    timestamp=sma.timestamp,
                    mode=ord_mode_live,
                    n_steps=int(ord_steps_live),
                    steps_par_phase=sma.ordonnanceur.steps_par_phase,
                )
                if result is not None:
                    st.success(f"Ordonnancement lance. CMax={result['makespan']}")
                else:
                    st.info("Aucun patient en attente.")

        st.markdown("**Graphe live**")
        live_graph = build_live_graph_result(
            sma=sma,
            mode=ord_mode_live,
            n_steps=int(ord_steps_live),
            projection_enabled=bool(projection_enabled),
        )
        if isinstance(live_graph, dict):
            fig_live = planning_figure(
                live_graph.get("planning", []),
                title=str(live_graph.get("title", "Live evolutif")),
            )
            if fig_live is not None:
                st.pyplot(fig_live, use_container_width=True)
            else:
                st.caption("Aucun graphe disponible.")
        else:
            st.caption(
                "Aucun graphe live pour l'instant (attendre une arrivee, ou activer la projection)."
            )

        tables_cols = st.columns([1, 1])
        with tables_cols[0]:
            st.markdown("**File d'attente (evolution live)**")
            queue_df = get_live_queue_df()
            st.dataframe(queue_df, use_container_width=True, hide_index=True)

        with tables_cols[1]:
            st.markdown("**Evenements traites**")
            processed = st.session_state.get("live_processed", [])
            if processed:
                st.dataframe(pd.DataFrame(processed[-30:]), use_container_width=True, hide_index=True)
            else:
                st.caption("Aucun evenement traite pour le moment.")

        # Tick autoplay en fin d'ecran pour laisser le rendu se faire avant la prochaine etape.
        live_autoplay_fragment()

    with tab_patients:
        st.subheader("Ajouter un patient manuellement")
        c_left, c_right = st.columns([1, 1.4])
        with c_left:
            nom_patient = st.text_input("Nom patient (optionnel)")
            urgence_pick = st.selectbox("Urgence", ["AUTO"] + [u.name for u in Urgence])
            mode_input = st.radio("Saisie operations", ["JSON manuel", "Generation aleatoire"], horizontal=True)
        with c_right:
            if mode_input == "JSON manuel":
                ex1 = [0] * sma.nb_competences
                ex2 = [0] * sma.nb_competences
                ex1[0] = 2
                ex1[min(1, sma.nb_competences - 1)] = 1
                ex2[min(1, sma.nb_competences - 1)] = 1
                ex2[min(2, sma.nb_competences - 1)] = 2
                example_json = json.dumps([ex1, ex2])
                st.caption(f"Exemple: {example_json}")
                raw_ops = st.text_area("Operations", value=example_json, height=120)
            else:
                min_ops = st.slider("Ops min", 1, 8, 1, key="manual_min_ops")
                max_ops = st.slider("Ops max", 1, 12, 3, key="manual_max_ops")
                max_comp = st.slider("Competences max par op", 1, max(1, sma.nb_competences), min(3, max(1, sma.nb_competences)))
                max_dur = st.slider("Duree max", 1, 12, 4)

        if st.button("Ajouter patient en attente"):
            try:
                if mode_input == "JSON manuel":
                    operations = parse_operations_json(raw_ops, sma.nb_competences)
                else:
                    operations = random_operations(
                        nb_competences=sma.nb_competences,
                        min_ops=min_ops,
                        max_ops=max_ops,
                        max_skills_per_op=max_comp,
                        max_duration=max_dur,
                )
                urgence_obj = None if urgence_pick == "AUTO" else Urgence[urgence_pick]
                sma.timestamp += random.uniform(0.2, 2.0)
                manual_name = nom_patient or f"Patient_{len(st.session_state.get('live_patients_state', [])) + 1:03d}"
                patient_obj = sma.simuler_arrivee_patient(
                    operations=operations,
                    urgence=urgence_obj,
                    nom=manual_name,
                )
                _ensure_live_patient_state_from_event(
                    nom=manual_name,
                    urgence_name=patient_obj.urgence.name,
                    operations=operations,
                    arrivee=float(sma.timestamp),
                )
                st.success("Patient ajoute.")
            except Exception as exc:
                st.error(f"Erreur ajout patient: {exc}")

        st.subheader("Patients en attente")
        st.dataframe(patients_to_df(sma.accueil.patients_en_attente), use_container_width=True, hide_index=True)
        st.subheader("Patients en cours")
        st.dataframe(patients_to_df(sma.accueil.patients_en_cours), use_container_width=True, hide_index=True)

    with tab_staff:
        st.subheader("Edition personnel")
        st.dataframe(personnel_to_df(sma.identificateur.registre_personnel), use_container_width=True, hide_index=True)

        if sma.identificateur.registre_personnel:
            selected_id = st.selectbox(
                "Selection personnel",
                options=[p.id for p in sma.identificateur.registre_personnel],
                format_func=lambda pid: (
                    f"{pid} - {next(p.nom for p in sma.identificateur.registre_personnel if p.id == pid)}"
                ),
            )
            person = next(p for p in sma.identificateur.registre_personnel if p.id == selected_id)

            s1, s2, s3 = st.columns(3)
            with s1:
                statuses = [s.name for s in StatutPersonnel]
                status_idx = statuses.index(person.statut.name)
                new_status = st.selectbox("Statut", statuses, index=status_idx)
            with s2:
                new_comp = st.multiselect(
                    "Competences",
                    options=list(range(sma.nb_competences)),
                    default=person.competences,
                    format_func=lambda c: f"C{c + 1}",
                )
            with s3:
                st.write("")
                st.write("")
                if st.button("Appliquer modifications"):
                    person.statut = StatutPersonnel[new_status]
                    person.competences = sorted(new_comp)
                    st.success("Personnel mis a jour.")

    with tab_schedule:
        st.subheader("Lancer ordonnancement")
        c_mode, c_steps, c_phase = st.columns(3)
        with c_mode:
            mode = st.selectbox("Mode", ["pipeline", "parallele"])
        with c_steps:
            n_steps = st.number_input("Iterations (parallele)", min_value=1, max_value=2000, value=150)
        with c_phase:
            run_steps_phase = st.number_input(
                "Steps par phase (pipeline)",
                min_value=1,
                max_value=1000,
                value=int(sma.ordonnanceur.steps_par_phase),
            )

        if st.button("Calculer planning"):
            sma.ordonnanceur.steps_par_phase = int(run_steps_phase)
            resultats = sma.lancer_ordonnancement(mode=mode, n_steps=int(n_steps), verbose=False)
            st.session_state.last_result = resultats
            st.success(f"Planning calcule. CMax = {resultats['makespan']}")

        st.divider()
        st.subheader("Absence instantanee")
        if sma.identificateur.registre_personnel:
            absence_id = st.selectbox(
                "ID personnel absent",
                options=[p.id for p in sma.identificateur.registre_personnel],
                key="absence_id_manual",
            )
            if st.button("Declencher absence maintenant"):
                resultats_re = sma.simuler_absence(int(absence_id), verbose=False)
                if resultats_re is not None:
                    st.session_state.last_result = resultats_re
                    st.success(f"Re-ordonnancement effectue. CMax = {resultats_re['makespan']}")
                else:
                    st.info("Absence enregistree, pas de re-ordonnancement immediat.")

        result = st.session_state.get("last_result")
        if result:
            st.divider()
            st.metric("CMax final", result["makespan"])
            fig = planning_figure(result["planning"], title=f"Planning final - CMax {result['makespan']}")
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
            else:
                st.warning("Planning vide.")

    with tab_logs:
        st.subheader("Messages inter-agents")
        st.dataframe(messages_to_df(sma.messages_echanges), use_container_width=True, hide_index=True)
        with st.expander("Logs Accueil"):
            st.code("\n".join(sma.accueil.log[-200:]) or "(aucun log)")
        with st.expander("Logs Identificateur"):
            st.code("\n".join(sma.identificateur.log[-200:]) or "(aucun log)")
        with st.expander("Logs Ordonnanceur"):
            st.code("\n".join(sma.ordonnanceur.log[-200:]) or "(aucun log)")


if __name__ == "__main__":
    main()
