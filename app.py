"""Streamlit decision-support system for reverse-logistics collection planning."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from src.data_processing import (
    DEMAND,
    DEPOT_ID,
    LATITUDE,
    LONGITUDE,
    PICKUP_REQUEST,
    STORAGE_CAPACITY,
    STORE_ID,
    STORE_NAME,
    build_geodesic_matrix,
    load_default_inputs,
    validate_distance_matrix,
    validate_stores,
)
from src.evaluation import compare_solutions
from src.export_results import (
    executive_report,
    route_plan_frame,
    slot_utilization_frame,
    store_collection_frame,
    vehicle_performance_frame,
)
from src.local_search import LocalSearchResult, improve_routes
from src.priority import calculate_store_priorities
from src.routing import generate_routes
from src.scenario_analysis import run_scenario, summarize_scenario
from src.scheduling import SchedulingInfeasibleError, schedule_routes
from src.visualization import (
    ROUTE_COLORS,
    build_space_time_figure,
    build_vehicle_gantt,
    route_color,
)


ROOT = Path(__file__).resolve().parent
PAGE_NAMES = [
    "Executive Dashboard",
    "Store & Inventory",
    "Optimization Centre",
    "Route Explorer",
    "Space-Time Analysis",
    "Performance Benchmarking",
    "Scenario Analysis",
    "Exports & Reports",
]
NAV_ICONS = ["📊", "🏪", "⚙️", "🗺️", "⏱️", "📈", "🔬", "📥"]
NAV_LABELS = [
    f"{icon}  {name}" for icon, name in zip(NAV_ICONS, PAGE_NAMES, strict=True)
]

st.set_page_config(
    page_title="Reverse Logistics Control Tower",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def _cached_default_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    return load_default_inputs(ROOT)


def _initialize_state() -> None:
    if "stores" not in st.session_state:
        original, matrix, parameters = _cached_default_inputs()
        validated, report = validate_stores(original)
        st.session_state.stores_original = original.copy(deep=True)
        st.session_state.stores = validated
        st.session_state.distance_matrix = matrix
        st.session_state.store_validation_report = report
        st.session_state.parameters = parameters
        st.session_state.warehouse = parameters["warehouse"]
        st.session_state.optimization_config = {
            "vehicle_capacity": parameters["vehicle_capacity_crates"]["value"],
            "vehicle_count": 1,
            "number_of_slots": parameters["number_of_available_time_slots"]["value"],
            "slot_duration_minutes": parameters[
                "maximum_route_duration_minutes_per_slot"
            ]["value"],
            "route_duration_limit_minutes": parameters[
                "maximum_route_duration_minutes_per_slot"
            ]["value"],
            "max_routes_per_slot": parameters["maximum_routes_per_time_slot"]["value"],
            "vehicle_speed_kmh": parameters["average_vehicle_speed_km_per_hour"]["value"],
            "fixed_collection_time_minutes": parameters[
                "fixed_collection_time_minutes_per_store"
            ]["value"],
            "variable_collection_time_minutes_per_crate": parameters[
                "variable_collection_time_minutes_per_crate"
            ]["value"],
            "max_iterations": 100,
            "min_improvement_minutes": 0.01,
        }
        st.session_state.greedy_solution = None
        st.session_state.optimized_solution = None
        st.session_state.scenarios = {}
        st.session_state.last_optimization_error = None


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --navy:#12304a; --green:#198754; --amber:#d97706; --red:#b42318; }
        .stApp { background:#f6f8fb; color:#152536; }
        [data-testid="stSidebar"] { background:#102a43; }
        [data-testid="stSidebar"] * { color:#f8fafc; }
        h1,h2,h3 { color:#12304a; }
        [data-testid="stMetric"] {
            background:#ffffff; border:1px solid #dce3eb; border-radius:10px;
            padding:14px 16px; box-shadow:0 1px 3px rgba(16,42,67,.05);
        }
        .control-tower-note { color:#526579; font-size:.92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f'<p class="control-tower-note">{subtitle}</p>', unsafe_allow_html=True)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _stores() -> pd.DataFrame:
    return st.session_state.stores


def _matrix() -> pd.DataFrame:
    return st.session_state.distance_matrix


def _config() -> dict[str, Any]:
    return st.session_state.optimization_config


def _current_store_report() -> pd.DataFrame:
    report = st.session_state.store_validation_report.copy()
    matrix_report = validate_distance_matrix(_matrix(), _stores())
    return pd.concat([report, pd.DataFrame(matrix_report)], ignore_index=True)


def _optimization_ready() -> tuple[bool, str]:
    report = _current_store_report()
    failed = report[report["Status"] == "Failed"]
    if not failed.empty:
        return False, "Resolve failed data or distance-matrix validation checks first."
    missing = {DEPOT_ID, *_stores()[STORE_ID].astype(str)}.difference(_matrix().index)
    if missing:
        return False, "Distance matrix does not cover every store and the warehouse."
    required_coordinates = {LATITUDE, LONGITUDE}
    if not required_coordinates.issubset(_stores().columns):
        return False, "Coordinates are required by the route model."
    return True, ""


def _current_parameters() -> dict[str, Any]:
    config = _config()
    return {
        "vehicle_capacity": int(config["vehicle_capacity"]),
        "vehicle_speed_kmh": float(config["vehicle_speed_kmh"]),
        "fixed_collection_time_minutes": float(
            config["fixed_collection_time_minutes"]
        ),
        "variable_collection_time_minutes_per_crate": float(
            config["variable_collection_time_minutes_per_crate"]
        ),
        "route_duration_limit_minutes": float(
            config["route_duration_limit_minutes"]
        ),
        "slot_duration_minutes": float(config["slot_duration_minutes"]),
        "number_of_slots": int(config["number_of_slots"]),
        "max_routes_per_slot": int(config["max_routes_per_slot"]),
    }


def _build_greedy_solution() -> dict[str, Any]:
    ready, explanation = _optimization_ready()
    if not ready:
        raise ValueError(explanation)
    parameters = _current_parameters()
    priority_stores = calculate_store_priorities(_stores())
    route_result = generate_routes(
        priority_stores,
        _matrix(),
        vehicle_capacity=parameters["vehicle_capacity"],
        vehicle_speed_kmh=parameters["vehicle_speed_kmh"],
        fixed_collection_time_minutes=parameters[
            "fixed_collection_time_minutes"
        ],
        variable_collection_time_minutes_per_crate=parameters[
            "variable_collection_time_minutes_per_crate"
        ],
        route_duration_limit_minutes=parameters[
            "route_duration_limit_minutes"
        ],
    )
    schedule = schedule_routes(
        route_result.routes,
        slot_duration_minutes=parameters["slot_duration_minutes"],
        max_routes_per_slot=parameters["max_routes_per_slot"],
        number_of_slots=parameters["number_of_slots"],
    )
    unserved = route_result.unserved_stores.copy()
    visited = {
        store_id
        for route in route_result.routes
        for store_id in route["sequence"][1:-1]
    }
    positive = set(
        _stores().loc[_stores()[DEMAND] > 0, STORE_ID].astype(str)
    )
    missing_positive = positive - visited
    for store_id in sorted(missing_positive):
        if store_id not in set(unserved.get(STORE_ID, pd.Series(dtype=str))):
            unserved.loc[len(unserved)] = [
                store_id,
                "infeasible",
                "positive_demand_store_not_routed",
            ]
    feasible = unserved.empty or not (unserved["status"] == "infeasible").any()
    return {
        "routes": route_result.routes,
        "schedule": schedule,
        "unserved_stores": unserved,
        "feasible": feasible,
        "parameters": parameters,
        "reason": "All positive-demand stores were routed and scheduled."
        if feasible
        else "One or more stores were reported as infeasible.",
    }


def _run_greedy() -> bool:
    try:
        candidate = _build_greedy_solution()
    except SchedulingInfeasibleError as exc:
        st.session_state.last_optimization_error = str(exc)
        st.session_state.last_infeasible_attempt = {
            "partial_schedule": exc.partial_schedule,
            "unscheduled_route_ids": exc.unscheduled_route_ids,
            "reason": str(exc),
        }
        st.error(f"Scheduling is infeasible: {exc}")
        return False
    except (ValueError, TypeError, KeyError) as exc:
        st.session_state.last_optimization_error = str(exc)
        st.error(f"Optimization could not be completed: {exc}")
        return False
    if not candidate["feasible"]:
        st.session_state.last_infeasible_attempt = candidate
        st.warning(
            "The attempted route plan is incomplete. The last successful greedy solution, "
            "if any, has been preserved."
        )
        return False
    st.session_state.greedy_solution = candidate
    st.session_state.optimized_solution = None
    st.session_state.last_optimization_error = None
    st.success("Greedy routes were generated and scheduled successfully.")
    return True


def _run_local_search() -> None:
    initial = st.session_state.greedy_solution
    if initial is None or not initial["feasible"]:
        st.error("Generate a feasible greedy solution before running local search.")
        return
    parameters = initial["parameters"]
    config = _config()
    try:
        result = improve_routes(
            initial["routes"],
            initial["schedule"],
            _stores(),
            _matrix(),
            vehicle_capacity=parameters["vehicle_capacity"],
            vehicle_speed_kmh=parameters["vehicle_speed_kmh"],
            fixed_collection_time_minutes=parameters[
                "fixed_collection_time_minutes"
            ],
            variable_collection_time_minutes_per_crate=parameters[
                "variable_collection_time_minutes_per_crate"
            ],
            route_duration_limit_minutes=parameters[
                "route_duration_limit_minutes"
            ],
            slot_duration_minutes=parameters["slot_duration_minutes"],
            max_routes_per_slot=parameters["max_routes_per_slot"],
            max_iterations=int(config["max_iterations"]),
            min_improvement_minutes=float(config["min_improvement_minutes"]),
        )
    except (ValueError, TypeError, KeyError) as exc:
        st.session_state.last_optimization_error = str(exc)
        st.error(f"Local search could not be completed: {exc}")
        return
    if not result.feasible:
        st.error("Local search returned an infeasible solution; prior results are preserved.")
        return
    st.session_state.optimized_solution = {
        "routes": result.optimized_routes,
        "schedule": result.optimized_schedule,
        "feasible": result.feasible,
        "reason": result.termination_reason,
        "parameters": parameters,
        "search_result": result,
    }
    st.session_state.last_optimization_error = None
    st.success("Local search completed; only feasible strict improvements were accepted.")


def _solution(name: str) -> dict[str, Any] | None:
    if name == "Optimized local search":
        return st.session_state.optimized_solution
    return st.session_state.greedy_solution


def _route_records_table(solution: dict[str, Any]) -> pd.DataFrame:
    slot_by_route = dict(
        zip(solution["schedule"]["Route ID"], solution["schedule"]["Time slot"])
    )
    parameters = solution["parameters"]
    rows = []
    for index, route in enumerate(solution["routes"]):
        rows.append(
            {
                "Route ID": route["route_id"],
                "Vehicle ID": "Vehicle 1",
                "Time slot": slot_by_route.get(route["route_id"]),
                "Ordered stores": "DEPOT → " + " → ".join(route["sequence"][1:-1]) + " → DEPOT",
                "Collection quantity (crates)": route["total_crates_collected"],
                "Remaining capacity (crates)": parameters["vehicle_capacity"]
                - route["total_crates_collected"],
                "Distance (km)": route["total_distance_km"],
                "Travel time (min)": route["total_travel_time_minutes"],
                "Collection time (min)": route["total_collection_time_minutes"],
                "Duration (min)": route["total_duration_minutes"],
                "Capacity utilization (%)": route["vehicle_utilization_percent"],
                "Feasible": route["total_duration_minutes"]
                <= parameters["route_duration_limit_minutes"] + 1e-9,
                "Color": route_color(index),
            }
        )
    return pd.DataFrame(rows)


def _render_alerts(solution: dict[str, Any] | None) -> None:
    stores = _stores()
    high_fill = stores[stores["Filling rate"] >= 0.90]
    left, right = st.columns(2)
    with left:
        st.subheader("Inventory alerts")
        if high_fill.empty:
            st.success("No stores are at or above 90% of storage capacity.")
        else:
            st.warning(f"{len(high_fill)} stores are at or above 90% filling rate.")
            st.dataframe(
                high_fill[[STORE_ID, STORE_NAME, DEMAND, STORAGE_CAPACITY, "Filling rate"]],
                width="stretch",
                hide_index=True,
            )
    with right:
        st.subheader("Operational alerts")
        if solution is None:
            st.info("Run the greedy heuristic to generate route-level alerts.")
            return
        served = {
            stop
            for route in solution["routes"]
            for stop in route["sequence"][1:-1]
        }
        high_priority_unserved = stores[
            (~stores[STORE_ID].isin(served)) & (stores["Collection priority score"] >= 0.5)
        ]
        if not solution["unserved_stores"].empty:
            st.error(
                f"{len(solution['unserved_stores'])} stores are explicitly listed as unserved/infeasible."
            )
        elif not high_priority_unserved.empty:
            st.warning(f"{len(high_priority_unserved)} high-priority stores are unserved.")
        else:
            st.success("All positive-demand stores are assigned.")
        over_duration = [
            route["route_id"]
            for route in solution["routes"]
            if route["total_duration_minutes"]
            > solution["parameters"]["route_duration_limit_minutes"]
        ]
        if over_duration:
            st.error(f"Routes over the duration limit: {', '.join(over_duration)}")
        utilization = [
            route["vehicle_utilization_percent"] for route in solution["routes"]
        ]
        if utilization and float(np.mean(utilization)) < 40:
            st.warning(f"Average route vehicle-capacity utilization is low ({np.mean(utilization):.1f}%).")


def page_dashboard() -> None:
    _page_header(
        "Executive Dashboard",
        "Live view of store inventory, collection requests, and the latest feasible route plan.",
    )
    stores = _stores()
    solution = st.session_state.greedy_solution
    active_stores = int((stores[DEMAND] > 0).sum())
    crates = int(stores[DEMAND].sum())
    bottles = crates * int(st.session_state.parameters["crate"]["bottles_per_crate"])
    requests = int(stores[PICKUP_REQUEST].sum())
    routes = len(solution["routes"]) if solution else 0
    slots = solution["schedule"]["Time slot"].nunique() if solution and not solution["schedule"].empty else 0
    utilization = (
        float(
            sum(route["total_crates_collected"] for route in solution["routes"])
            / (len(solution["routes"]) * solution["parameters"]["vehicle_capacity"])
            * 100
        )
        if solution and solution["routes"]
        else 0.0
    )
    coverage = (
        100
        * len(
            {
                store
                for route in solution["routes"]
                for store in route["sequence"][1:-1]
            }
            & set(stores.loc[stores[DEMAND] > 0, STORE_ID])
        )
        / active_stores
        if solution and active_stores
        else 0.0
    )
    values = [
        ("Active stores", active_stores),
        ("Available crates", crates),
        ("Available bottles", bottles),
        ("Pickup requests", requests),
        ("Generated routes", routes),
        ("Time slots used", slots),
        ("Vehicle utilization", f"{utilization:.1f}%"),
        ("Store coverage", f"{coverage:.1f}%"),
    ]
    for start in range(0, len(values), 4):
        cols = st.columns(4)
        for col, (label, value) in zip(cols, values[start : start + 4]):
            col.metric(label, value)

    st.subheader("Inventory analytics")
    left, right = st.columns(2)
    with left:
        inventory = stores.sort_values(DEMAND, ascending=True)
        st.plotly_chart(
            px.bar(
                inventory,
                x=DEMAND,
                y=STORE_ID,
                orientation="h",
                color="Filling rate",
                color_continuous_scale=["#d9e2ec", "#d97706", "#b42318"],
                title="Current crate inventory by store",
            ),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            px.histogram(
                stores,
                x="Filling rate",
                nbins=10,
                title="Store filling-rate distribution",
                color_discrete_sequence=["#1769aa"],
            ),
            width="stretch",
        )
    left, right = st.columns(2)
    with left:
        priority_counts = stores.assign(
            priority_band=pd.cut(
                stores["Collection priority score"],
                bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
                labels=["Low", "Moderate", "High", "Very high"],
            )
        )["priority_band"].value_counts().rename_axis("Priority band").reset_index(name="Stores")
        st.plotly_chart(
            px.pie(priority_counts, names="Priority band", values="Stores", hole=0.55,
                   title="Collection priority distribution",
                   color_discrete_sequence=["#cbd5e1", "#1769aa", "#d97706", "#198754"]),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            px.scatter(
                stores,
                x=STORAGE_CAPACITY,
                y=DEMAND,
                color=PICKUP_REQUEST,
                hover_name=STORE_ID,
                title="Demand versus storage capacity",
                labels={PICKUP_REQUEST: "Pickup request"},
                color_discrete_map={0: "#1769aa", 1: "#d97706"},
            ),
            width="stretch",
        )
    _render_alerts(solution)


def page_store_inventory() -> None:
    _page_header(
        "Store & Inventory Management",
        "Upload and validate store data without silently dropping or repairing records.",
    )
    with st.expander("Upload store CSV and optional distance matrix", expanded=True):
        with st.form("store_upload_form"):
            store_file = st.file_uploader("Store dataset (CSV)", type=["csv"], key="store_csv_upload")
            matrix_file = st.file_uploader(
                "Distance matrix (CSV, optional)", type=["csv"], key="matrix_csv_upload"
            )
            submit = st.form_submit_button("Validate and load uploaded data")
        if submit and store_file is not None:
            try:
                candidate = pd.read_csv(io.BytesIO(store_file.getvalue()))
                validated, report = validate_stores(candidate)
                st.session_state.upload_validation_report = report
                failures = report[report["Status"] == "Failed"]
                if not failures.empty:
                    st.error("The uploaded dataset has validation failures; the last valid dataset is retained.")
                else:
                    st.session_state.stores_original = candidate.copy(deep=True)
                    st.session_state.stores = validated
                    st.session_state.store_validation_report = report
                    st.session_state.greedy_solution = None
                    st.session_state.optimized_solution = None
                    st.success(f"Loaded {len(validated)} validated store records.")
                    if matrix_file is not None:
                        uploaded_matrix = pd.read_csv(io.BytesIO(matrix_file.getvalue())).set_index("NodeID")
                        matrix_report = validate_distance_matrix(uploaded_matrix, validated)
                        if any(row["Status"] == "Failed" for row in matrix_report):
                            st.error("Uploaded matrix is invalid; the current matrix was retained.")
                        else:
                            st.session_state.distance_matrix = uploaded_matrix
                    elif set(validated[STORE_ID].astype(str)) != set(_matrix().index) - {DEPOT_ID}:
                        st.warning("The current distance matrix does not match these store IDs.")
            except (ValueError, KeyError, pd.errors.ParserError) as exc:
                st.error(f"Could not load upload: {exc}")
        elif submit:
            st.warning("Choose a store CSV before submitting.")
    if "upload_validation_report" in st.session_state:
        with st.expander("Latest upload validation report"):
            st.dataframe(
                st.session_state.upload_validation_report,
                width="stretch",
                hide_index=True,
            )

    if {LATITUDE, LONGITUDE}.issubset(_stores().columns):
        if st.button("Build straight-line distance matrix from supplied coordinates"):
            try:
                warehouse = st.session_state.warehouse
                matrix = build_geodesic_matrix(
                    _stores(),
                    warehouse["latitude"],
                    warehouse["longitude"],
                )
                st.session_state.distance_matrix = matrix
                st.success("Built geodesic distances from supplied coordinates (not road distances).")
            except (ValueError, TypeError) as exc:
                st.error(f"Could not build distance matrix: {exc}")
    else:
        st.warning("Coordinates are missing. Geographic visualization and coordinate-derived distances are unavailable.")

    st.subheader("Validation report")
    report = _current_store_report()
    st.dataframe(report, width="stretch", hide_index=True)
    st.download_button(
        "Download validated dataset",
        _csv_bytes(_stores()),
        file_name="validated_stores.csv",
        mime="text/csv",
    )

    stores = _stores()
    st.subheader("Filter store records")
    c1, c2, c3 = st.columns(3)
    ids = c1.multiselect("Store IDs", sorted(stores[STORE_ID].astype(str)), default=[])
    priority_min, priority_max = c2.slider(
        "Collection priority range", 0.0, 1.0, (0.0, 1.0), 0.01
    )
    fill_min, fill_max = c3.slider("Filling-rate range", 0.0, 1.0, (0.0, 1.0), 0.01)
    c1, c2 = st.columns(2)
    pickup_filter = c1.selectbox("Pickup request", ["All", "Requested", "Not requested"])
    max_inventory = int(stores[DEMAND].max()) if not stores.empty else 0
    inventory_range = c2.slider(
        "Current crate quantity",
        0,
        max(1, max_inventory),
        (0, max(1, max_inventory)),
    )
    filtered = stores[
        stores["Collection priority score"].between(priority_min, priority_max)
        & stores["Filling rate"].between(fill_min, fill_max)
        & stores[DEMAND].between(inventory_range[0], inventory_range[1])
    ]
    if ids:
        filtered = filtered[filtered[STORE_ID].astype(str).isin(ids)]
    if pickup_filter == "Requested":
        filtered = filtered[stores[PICKUP_REQUEST] == 1]
    elif pickup_filter == "Not requested":
        filtered = filtered[stores[PICKUP_REQUEST] == 0]
    st.dataframe(filtered, width="stretch", hide_index=True)
    if not filtered.empty:
        st.plotly_chart(
            px.bar(
                filtered.sort_values("Collection priority score"),
                x="Collection priority score",
                y=STORE_ID,
                orientation="h",
                color="Filling rate",
                title="Store priority ranking",
                color_continuous_scale=["#1769aa", "#d97706", "#b42318"],
            ),
            width="stretch",
        )


def page_optimization() -> None:
    _page_header(
        "Optimization Control Centre",
        "Configure and run the existing priority-greedy heuristic, scheduler, and best-improvement local search.",
    )
    ready, explanation = _optimization_ready()
    if not ready:
        st.error(explanation)
        st.dataframe(_current_store_report(), width="stretch", hide_index=True)
        return

    config = _config()
    cols = st.columns(4)
    config["vehicle_capacity"] = cols[0].number_input(
        "Vehicle capacity (crates)", min_value=1, value=int(config["vehicle_capacity"]), step=1
    )
    cols[1].number_input("Available vehicles", min_value=1, value=1, disabled=True)
    config["number_of_slots"] = cols[2].number_input(
        "Available time slots", min_value=1, value=int(config["number_of_slots"]), step=1
    )
    config["slot_duration_minutes"] = cols[3].number_input(
        "Slot duration (minutes)", min_value=1.0,
        value=float(config["slot_duration_minutes"]), step=15.0,
    )
    cols = st.columns(4)
    config["route_duration_limit_minutes"] = cols[0].number_input(
        "Max route duration (minutes)", min_value=1.0,
        value=float(config["route_duration_limit_minutes"]), step=15.0,
    )
    config["max_routes_per_slot"] = cols[1].number_input(
        "Maximum routes per slot", min_value=1, value=int(config["max_routes_per_slot"]), step=1
    )
    config["vehicle_speed_kmh"] = cols[2].number_input(
        "Average speed (km/h)", min_value=0.1, value=float(config["vehicle_speed_kmh"]), step=1.0
    )
    config["max_iterations"] = cols[3].number_input(
        "Local-search iterations", min_value=0, max_value=10000,
        value=int(config["max_iterations"]), step=10,
    )
    cols = st.columns(3)
    config["fixed_collection_time_minutes"] = cols[0].number_input(
        "Fixed service time (min/store)", min_value=0.0,
        value=float(config["fixed_collection_time_minutes"]), step=0.5,
    )
    config["variable_collection_time_minutes_per_crate"] = cols[1].number_input(
        "Variable service time (min/crate)", min_value=0.0,
        value=float(config["variable_collection_time_minutes_per_crate"]), step=0.05,
    )
    config["min_improvement_minutes"] = cols[2].number_input(
        "Minimum improvement (min)", min_value=0.001,
        value=float(config["min_improvement_minutes"]), step=0.01, format="%.3f",
    )
    st.session_state.optimization_config = config
    st.info(
        "Current route generation supports one vehicle making sequential depot-return trips. "
        "The fleet-size control is fixed at one; multiple simultaneous vehicles are not modeled."
    )
    actions = st.columns(3)
    if actions[0].button("Run Greedy Heuristic", type="primary", width="stretch"):
        _run_greedy()
    if actions[1].button("Run Local Search", width="stretch"):
        _run_local_search()
    if actions[2].button("Run Complete Optimization", width="stretch"):
        if _run_greedy():
            _run_local_search()

    if st.session_state.greedy_solution is not None:
        solution = st.session_state.greedy_solution
        st.subheader("Initial greedy route plan")
        st.caption(solution["reason"])
        st.dataframe(_route_records_table(solution), width="stretch", hide_index=True)
        st.dataframe(solution["schedule"], width="stretch", hide_index=True)
        if st.session_state.optimized_solution is not None:
            optimized = st.session_state.optimized_solution
            result: LocalSearchResult = optimized["search_result"]
            st.subheader("Optimized solution")
            columns = st.columns(4)
            columns[0].metric("Time saved", f"{result.time_saved_minutes:.2f} min")
            columns[1].metric("Distance change", f"{result.distance_saved_km:.2f} km")
            columns[2].metric("Accepted moves", result.relocation_moves_accepted + result.swap_moves_accepted + result.two_opt_moves_accepted)
            columns[3].metric("Iterations", result.iterations)
            st.dataframe(_route_records_table(optimized), width="stretch", hide_index=True)
            with st.expander("Local-search convergence and move counts"):
                st.write(
                    f"Relocate: {result.relocation_moves_accepted} · "
                    f"Swap: {result.swap_moves_accepted} · "
                    f"2-opt: {result.two_opt_moves_accepted} · "
                    f"Termination: {result.termination_reason}"
                )
                st.line_chart(
                    result.convergence_history.set_index("Iteration")["Total route time (min)"]
                )
        if not solution["unserved_stores"].empty:
            st.error("Unserved stores are explicitly listed below.")
            st.dataframe(solution["unserved_stores"], width="stretch", hide_index=True)


def _make_route_map(
    solution: dict[str, Any],
    selected_route: str,
    selected_slot: str,
    show_markers: bool,
    show_lines: bool,
) -> folium.Map:
    stores = _stores()
    warehouse = st.session_state.warehouse
    center = [
        float(warehouse["latitude"]),
        float(warehouse["longitude"]),
    ]
    map_view = folium.Map(location=center, zoom_start=13, control_scale=True, tiles="OpenStreetMap")
    store_lookup = stores.set_index(STORE_ID).to_dict("index")
    slot_by_route = dict(zip(solution["schedule"]["Route ID"], solution["schedule"]["Time slot"]))
    routes = solution["routes"]
    visible_routes = routes if selected_route == "All routes" else [
        route for route in routes if route["route_id"] == selected_route
    ]
    if selected_slot != "All slots":
        visible_routes = [
            route
            for route in visible_routes
            if str(slot_by_route[route["route_id"]]) == selected_slot
        ]
    if show_markers:
        folium.Marker(
            center,
            popup=f"{warehouse['name']} · Central warehouse",
            tooltip="Central warehouse / depot",
            icon=folium.Icon(color="darkblue", icon="home", prefix="fa"),
        ).add_to(map_view)
        for store_id, record in store_lookup.items():
            if pd.isna(record.get(LATITUDE)) or pd.isna(record.get(LONGITUDE)):
                continue
            folium.CircleMarker(
                [float(record[LATITUDE]), float(record[LONGITUDE])],
                radius=5,
                color="#64748b",
                fill=True,
                fill_opacity=0.85,
                tooltip=f"{store_id} · {record.get(STORE_NAME, '')}",
                popup=(
                    f"<b>{store_id}</b><br>{record.get(STORE_NAME, '')}"
                    f"<br>Inventory: {record[DEMAND]} crates"
                    f"<br>Priority: {record.get('Collection priority score', 0):.3f}"
                ),
            ).add_to(map_view)
    route_order = {route["route_id"]: index for index, route in enumerate(routes)}
    for route in visible_routes:
        route_id = route["route_id"]
        color = route_color(route_order[route_id])
        latlon = [center]
        for stop_number, store_id in enumerate(route["sequence"][1:-1], start=1):
            record = store_lookup[store_id]
            point = [float(record[LATITUDE]), float(record[LONGITUDE])]
            latlon.append(point)
            if show_markers:
                folium.Marker(
                    point,
                    tooltip=f"{route_id} · Stop {stop_number} · {store_id}",
                    popup=(
                        f"<b>{store_id}</b><br>Inventory: {record[DEMAND]} crates"
                        f"<br>Collection quantity: {record[DEMAND]} crates"
                        f"<br>Priority: {record.get('Collection priority score', 0):.3f}"
                        f"<br>Vehicle 1<br>Route {route_id}<br>Time slot {slot_by_route[route_id]}"
                    ),
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="background:{color};color:white;border-radius:50%;'
                            f'width:24px;height:24px;text-align:center;line-height:24px;'
                            f'font-weight:bold;border:2px solid white">{stop_number}</div>'
                        )
                    ),
                ).add_to(map_view)
        latlon.append(center)
        if show_lines:
            folium.PolyLine(
                latlon,
                color=color,
                weight=4,
                opacity=0.8,
                tooltip=f"{route_id} · Slot {slot_by_route[route_id]} · straight-line connections",
            ).add_to(map_view)
    return map_view


def page_route_explorer() -> None:
    _page_header(
        "Route Explorer",
        "Explore stop order and straight-line route connections. These are not road-network directions.",
    )
    solution_kind = st.radio(
        "Solution",
        ["Initial greedy", "Optimized local search"],
        horizontal=True,
        key="map_solution",
    )
    solution = _solution(solution_kind)
    if solution is None:
        st.info("Generate a feasible route plan on the Optimization Centre page.")
        return
    if not {LATITUDE, LONGITUDE}.issubset(_stores().columns):
        st.warning("Coordinates are unavailable; geographic mapping cannot be displayed.")
        return
    if _stores()[[LATITUDE, LONGITUDE]].isna().any().any():
        st.warning("Some coordinates are missing; fix them before displaying the map.")
        return
    route_names = ["All routes"] + [route["route_id"] for route in solution["routes"]]
    col1, col2, col3, col4 = st.columns(4)
    selected_route = col1.selectbox("Route filter", route_names)
    slot_names = ["All slots"] + sorted(
        str(int(slot)) for slot in solution["schedule"]["Time slot"].dropna().unique()
    )
    selected_slot = col2.selectbox("Time-slot filter", slot_names)
    col3.selectbox("Vehicle", ["Vehicle 1"], disabled=True)
    show_markers = col4.toggle("Show store markers", value=True)
    show_lines = st.toggle("Show route lines", value=True)
    st.caption("Straight-line segments use the project's geodesic distance matrix; actual road paths may differ.")
    map_view = _make_route_map(
        solution, selected_route, selected_slot, show_markers, show_lines
    )
    st_folium(map_view, use_container_width=True, height=580, returned_objects=[])
    table = _route_records_table(solution)
    st.subheader("Route details")
    if selected_route != "All routes":
        table = table[table["Route ID"] == selected_route]
    if selected_slot != "All slots":
        table = table[table["Time slot"].astype(str) == selected_slot]
    st.dataframe(table, width="stretch", hide_index=True)
    chosen = st.selectbox(
        "View complete stop sequence",
        [route["route_id"] for route in solution["routes"]],
    )
    if chosen:
        selected = next(route for route in solution["routes"] if route["route_id"] == chosen)
        st.code(" → ".join(selected["sequence"]), language=None)


def page_space_time() -> None:
    _page_header(
        "Space-Time Analysis & Scheduling",
        "Elapsed route time is calculated from distance, speed, and collection service; no clock times are assumed.",
    )
    kind = st.radio("Route set", ["Initial greedy", "Optimized local search"], horizontal=True)
    solution = _solution(kind)
    if solution is None:
        st.info("Generate a route plan before viewing vehicle activity.")
        return
    if not solution["routes"]:
        st.warning("There are no generated routes to visualize.")
        return
    parameters = solution["parameters"]
    st.plotly_chart(
        build_space_time_figure(
            solution["routes"],
            solution["schedule"],
            _stores(),
            _matrix(),
            parameters["vehicle_speed_kmh"],
            parameters["fixed_collection_time_minutes"],
            parameters["variable_collection_time_minutes_per_crate"],
        ),
        width="stretch",
    )
    st.plotly_chart(
        build_vehicle_gantt(
            solution["routes"],
            solution["schedule"],
            _stores(),
            _matrix(),
            parameters["vehicle_speed_kmh"],
            parameters["fixed_collection_time_minutes"],
            parameters["variable_collection_time_minutes_per_crate"],
            parameters["slot_duration_minutes"],
        ),
        width="stretch",
    )
    occupancy = slot_utilization_frame(
        solution["schedule"], parameters["slot_duration_minutes"]
    )
    st.subheader("Time-slot occupancy")
    st.dataframe(occupancy, width="stretch", hide_index=True)
    if not occupancy.empty:
        st.plotly_chart(
            px.bar(
                occupancy,
                x="Time slot",
                y="Utilization (%)",
                text="Utilization (%)",
                title="Route operating-time utilization by slot",
                color_discrete_sequence=["#1769aa"],
            ),
            width="stretch",
        )
        longest = max(solution["routes"], key=lambda item: item["total_duration_minutes"])
        longest_service = max(solution["routes"], key=lambda item: item["total_collection_time_minutes"])
        st.info(
            f"Longest route: {longest['route_id']} ({longest['total_duration_minutes']:.1f} min). "
            f"Longest collection service: {longest_service['route_id']} "
            f"({longest_service['total_collection_time_minutes']:.1f} min). "
            "Unused slot capacity is shown as idle; no vehicle overlap is inferred."
        )


def _comparison_table() -> pd.DataFrame | None:
    initial = st.session_state.greedy_solution
    optimized = st.session_state.optimized_solution
    if initial is None or optimized is None:
        return None
    parameters = initial["parameters"]
    return compare_solutions(
        _stores(),
        initial["routes"],
        initial["schedule"],
        optimized["routes"],
        optimized["schedule"],
        _matrix(),
        vehicle_capacity=parameters["vehicle_capacity"],
        vehicle_speed_kmh=parameters["vehicle_speed_kmh"],
        fixed_collection_time_minutes=parameters["fixed_collection_time_minutes"],
        variable_collection_time_minutes_per_crate=parameters[
            "variable_collection_time_minutes_per_crate"
        ],
        slot_duration_minutes=parameters["slot_duration_minutes"],
        bottles_per_crate=st.session_state.parameters["crate"]["bottles_per_crate"],
        route_duration_limit_minutes=parameters["route_duration_limit_minutes"],
        number_of_available_time_slots=parameters["number_of_slots"],
    )


def page_benchmarking() -> None:
    _page_header(
        "Performance Benchmarking",
        "KPIs are independently recalculated from original demand, distance matrix, route sequences, and final schedules.",
    )
    summary = _comparison_table()
    if summary is None:
        st.info("Run Complete Optimization to compare the initial and improved solutions.")
        return
    st.dataframe(summary, width="stretch", hide_index=True)
    indexed = summary.set_index("KPI")
    metrics = [
        "Total distance travelled",
        "Total operational time",
        "Vehicle capacity utilization",
        "Total time-slot utilization",
        "Percentage of stores served",
    ]
    for kpi in metrics:
        row = indexed.loc[kpi]
        left, right = st.columns(2)
        left.metric(
            kpi,
            f"{row['Optimized Heuristic']:.2f} {row['Unit']}",
            delta=f"{row['Change (optimized - initial)']:+.2f} {row['Change unit']}",
        )
        right.metric("Initial greedy", f"{row['Initial Greedy']:.2f} {row['Unit']}")

    st.subheader("Visual comparison")
    compare_cols = st.columns(2)
    for col, kpi, title in [
        (compare_cols[0], "Total distance travelled", "Distance comparison"),
        (compare_cols[1], "Total operational time", "Operational time comparison"),
    ]:
        data = pd.DataFrame(
            {
                "Solution": ["Initial Greedy", "Optimized Heuristic"],
                "Value": [indexed.loc[kpi, "Initial Greedy"], indexed.loc[kpi, "Optimized Heuristic"]],
            }
        )
        col.plotly_chart(
            px.bar(
                data,
                x="Solution",
                y="Value",
                title=title,
                color="Solution",
                color_discrete_map={"Initial Greedy": "#1769aa", "Optimized Heuristic": "#198754"},
            ),
            width="stretch",
        )
    cols = st.columns(2)
    for col, kpi, title in [
        (cols[0], "Vehicle capacity utilization", "Vehicle capacity utilization"),
        (cols[1], "Total time-slot utilization", "Time-slot utilization"),
    ]:
        col.plotly_chart(
            go.Figure(
                data=[
                    go.Bar(
                        x=["Initial Greedy", "Optimized Heuristic"],
                        y=[indexed.loc[kpi, "Initial Greedy"], indexed.loc[kpi, "Optimized Heuristic"]],
                        marker_color=["#1769aa", "#198754"],
                    )
                ]
            ).update_layout(title=title, yaxis_title="Percent", template="plotly_white"),
            width="stretch",
        )
    served = pd.DataFrame(
        {
            "Solution": ["Initial Greedy", "Optimized Heuristic"],
            "Stores served (%)": [
                indexed.loc["Percentage of stores served", "Initial Greedy"],
                indexed.loc["Percentage of stores served", "Optimized Heuristic"],
            ],
        }
    )
    st.plotly_chart(
        px.bar(served, x="Solution", y="Stores served (%)", color="Solution",
               color_discrete_map={"Initial Greedy": "#1769aa", "Optimized Heuristic": "#198754"},
               title="Store collection coverage"),
        width="stretch",
    )
    route_rows = []
    for label, solution in [
        ("Initial Greedy", st.session_state.greedy_solution),
        ("Optimized Heuristic", st.session_state.optimized_solution),
    ]:
        route_rows.extend(
            {"Route ID": route["route_id"], "Solution": label, "Duration (min)": route["total_duration_minutes"]}
            for route in solution["routes"]
        )
    st.plotly_chart(
        px.bar(pd.DataFrame(route_rows), x="Route ID", y="Duration (min)", color="Solution",
               barmode="group", color_discrete_map={"Initial Greedy": "#1769aa", "Optimized Heuristic": "#198754"},
               title="Route-level duration comparison"),
        width="stretch",
    )
    count_rows = [
        {
            "Solution": label,
            "Count": indexed.loc[kpi, label],
            "Measure": measure,
        }
        for label in ["Initial Greedy", "Optimized Heuristic"]
        for kpi, measure in [
            ("Number of routes", "Routes"),
            ("Number of time slots used", "Time slots"),
        ]
    ]
    st.plotly_chart(
        px.bar(
            pd.DataFrame(count_rows),
            x="Measure",
            y="Count",
            color="Solution",
            barmode="group",
            color_discrete_map={
                "Initial Greedy": "#1769aa",
                "Optimized Heuristic": "#198754",
            },
            title="Route and time-slot counts",
        ),
        width="stretch",
    )
    breakdown = []
    for label, solution in [
        ("Initial Greedy", st.session_state.greedy_solution),
        ("Optimized Heuristic", st.session_state.optimized_solution),
    ]:
        breakdown.extend(
            [
                {"Solution": label, "Activity": "Travel", "Minutes": sum(r["total_travel_time_minutes"] for r in solution["routes"])},
                {"Solution": label, "Activity": "Collection", "Minutes": sum(r["total_collection_time_minutes"] for r in solution["routes"])},
            ]
        )
    st.plotly_chart(
        px.bar(pd.DataFrame(breakdown), x="Solution", y="Minutes", color="Activity",
               barmode="stack", title="Travel versus collection-time breakdown",
               color_discrete_map={"Travel": "#1769aa", "Collection": "#198754"}),
        width="stretch",
    )
    result: LocalSearchResult = st.session_state.optimized_solution["search_result"]
    st.subheader("Local-search convergence")
    history = result.convergence_history
    st.plotly_chart(
        px.line(
            history,
            x="Iteration",
            y="Total route time (min)",
            markers=True,
            title="Objective after each accepted local-search move",
        ),
        width="stretch",
    )
    st.caption(
        f"Termination: {result.termination_reason}. "
        f"Accepted relocation={result.relocation_moves_accepted}, "
        f"swap={result.swap_moves_accepted}, 2-opt={result.two_opt_moves_accepted}."
    )
    with st.expander("KPI formula definitions"):
        st.markdown(
            """
            - **Vehicle utilization:** collected crates ÷ (routes × vehicle capacity).
            - **Store coverage:** unique positive-demand stores visited ÷ positive-demand stores.
            - **High-volume not collected:** demand ÷ vehicle capacity ≥ 90%, and store absent from that solution.
            - **Time-slot utilization:** recalculated route operating time ÷ (used slots × slot duration).
            - Percentage KPI changes are percentage points (optimized minus initial).
            """
        )
        st.caption("A higher capacity utilization is not automatically better; feasibility, service and time are considered together.")


def page_scenarios() -> None:
    _page_header(
        "Scenario & What-if Analysis",
        "Run alternative configurations separately; the baseline route and local-search results are not overwritten.",
    )
    base = _current_parameters()
    st.info("The underlying heuristic supports one vehicle. Fleet-size what-if scenarios are not modeled.")
    cols = st.columns(4)
    scenario_name = cols[0].text_input("Scenario name", value="Alternative scenario")
    capacity = cols[1].number_input("Vehicle capacity", min_value=1, value=base["vehicle_capacity"], step=1)
    slots = cols[2].number_input("Available slots", min_value=1, value=base["number_of_slots"], step=1)
    slot_duration = cols[3].number_input(
        "Slot duration (min)", min_value=1.0, value=base["slot_duration_minutes"], step=15.0
    )
    cols = st.columns(3)
    max_duration = cols[0].number_input(
        "Maximum route duration (min)", min_value=1.0,
        value=base["route_duration_limit_minutes"], step=15.0,
    )
    max_routes = cols[1].number_input(
        "Maximum routes per slot", min_value=1, value=base["max_routes_per_slot"], step=1
    )
    demand_multiplier = cols[2].number_input(
        "Demand multiplier", min_value=0.1, value=1.0, step=0.1, format="%.1f"
    )
    if st.button("Run scenario", type="primary"):
        try:
            scenario_stores = _stores().copy(deep=True)
            if demand_multiplier != 1.0:
                scenario_stores[DEMAND] = np.ceil(
                    scenario_stores[DEMAND] * demand_multiplier
                ).astype(int)
            if (scenario_stores[DEMAND] > scenario_stores[STORAGE_CAPACITY]).any():
                scenario_result = {
                    "scenario_name": scenario_name,
                    "feasible": False,
                    "routes": [],
                    "schedule": pd.DataFrame(columns=["Time slot", "Route ID"]),
                    "unserved_stores": pd.DataFrame(
                        [
                            {
                                STORE_ID: row[STORE_ID],
                                "status": "infeasible",
                                "reason": "scenario_demand_exceeds_store_capacity",
                            }
                            for _, row in scenario_stores[
                                scenario_stores[DEMAND] > scenario_stores[STORAGE_CAPACITY]
                            ].iterrows()
                        ]
                    ),
                    "unscheduled_route_ids": [],
                    "reason": "Scenario demand exceeds one or more store storage capacities.",
                    "parameters": {},
                }
            else:
                scenario_stores = calculate_store_priorities(scenario_stores)
                scenario_parameters = {
                    **base,
                    "vehicle_capacity": int(capacity),
                    "number_of_slots": int(slots),
                    "slot_duration_minutes": float(slot_duration),
                    "route_duration_limit_minutes": float(max_duration),
                    "max_routes_per_slot": int(max_routes),
                }
                scenario_result = run_scenario(
                    scenario_stores,
                    _matrix(),
                    scenario_parameters,
                    scenario_name,
                )
            st.session_state.scenarios[scenario_name] = {
                "result": scenario_result,
                "stores": scenario_stores,
                "parameters": scenario_result.get("parameters", {}),
            }
            if scenario_result["feasible"]:
                st.success(f"Scenario “{scenario_name}” completed.")
            else:
                st.warning(f"Scenario “{scenario_name}” is infeasible: {scenario_result['reason']}")
        except (ValueError, TypeError, KeyError, SchedulingInfeasibleError) as exc:
            st.error(f"Scenario could not be evaluated: {exc}")

    scenario_rows = []
    for record in st.session_state.scenarios.values():
        result = record["result"]
        params = record["parameters"]
        if params:
            scenario_rows.append(
                summarize_scenario(
                    result,
                    params["vehicle_capacity"],
                    params["slot_duration_minutes"],
                    record["stores"],
                )
            )
        else:
            scenario_rows.append(
                {
                    "Scenario": result["scenario_name"],
                    "Feasibility": "Infeasible",
                    "Stores served": 0,
                    "Crates collected": 0,
                    "Total distance (km)": 0.0,
                    "Total route duration (min)": 0.0,
                    "Number of routes": 0,
                    "Used time slots": 0,
                    "Vehicle utilization (%)": 0.0,
                    "Time-slot utilization (%)": 0.0,
                    "High-volume stores not collected": 0,
                    "Unserved stores": len(result["unserved_stores"]),
                    "Reason": result["reason"],
                }
            )
    if scenario_rows:
        st.subheader("Retained scenario results")
        st.dataframe(pd.DataFrame(scenario_rows), width="stretch", hide_index=True)
        selected = st.selectbox("Scenario details", list(st.session_state.scenarios))
        st.dataframe(
            st.session_state.scenarios[selected]["result"]["schedule"],
            width="stretch",
            hide_index=True,
        )


def page_exports() -> None:
    _page_header(
        "Exports & Reports",
        "Download CSV plans and KPI summaries generated from the same route schedules shown in the application.",
    )
    initial = st.session_state.greedy_solution
    optimized = st.session_state.optimized_solution
    if initial is None:
        st.info("Run the greedy heuristic before exporting route plans.")
        return
    parameters = initial["parameters"]
    solution_kind = st.radio(
        "Route plan to export",
        ["Initial greedy", "Optimized local search"],
        horizontal=True,
    )
    selected = optimized if solution_kind == "Optimized local search" and optimized else initial
    route_plan = route_plan_frame(
        selected["routes"],
        selected["schedule"],
        _stores(),
        parameters["vehicle_capacity"],
        _matrix(),
        parameters["vehicle_speed_kmh"],
        parameters["fixed_collection_time_minutes"],
        parameters["variable_collection_time_minutes_per_crate"],
    )
    st.download_button(
        "Download route plan CSV",
        _csv_bytes(route_plan),
        file_name=f"{solution_kind.lower().replace(' ', '_')}_route_plan.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download store collection summary",
        _csv_bytes(store_collection_frame(_stores(), selected["routes"], selected["schedule"])),
        file_name="store_collection_summary.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download vehicle-wise performance",
        _csv_bytes(vehicle_performance_frame(selected["routes"], parameters["vehicle_capacity"])),
        file_name="vehicle_performance.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download time-slot utilization",
        _csv_bytes(slot_utilization_frame(selected["schedule"], parameters["slot_duration_minutes"])),
        file_name="time_slot_utilization.csv",
        mime="text/csv",
    )
    if optimized:
        comparison = _comparison_table()
        if comparison is not None:
            st.download_button(
                "Download KPI comparison",
                _csv_bytes(comparison),
                file_name="kpi_comparison.csv",
                mime="text/csv",
            )
            report_text = executive_report(
                _stores(), comparison, comparison, selected["feasible"]
            )
            st.download_button(
                "Download executive report",
                report_text.encode("utf-8"),
                file_name="executive_report.txt",
                mime="text/plain",
            )
    st.download_button(
        "Download unserved stores",
        _csv_bytes(selected["unserved_stores"]),
        file_name="unserved_stores.csv",
        mime="text/csv",
    )
    if st.session_state.scenarios:
        rows = []
        for record in st.session_state.scenarios.values():
            result = record["result"]
            params = record["parameters"]
            if params:
                rows.append(
                    summarize_scenario(
                        result,
                        params["vehicle_capacity"],
                        params["slot_duration_minutes"],
                        record["stores"],
                    )
                )
        if rows:
            st.download_button(
                "Download scenario comparison",
                _csv_bytes(pd.DataFrame(rows)),
                file_name="scenario_comparison.csv",
                mime="text/csv",
            )
    st.subheader("Route plan preview")
    st.dataframe(route_plan, width="stretch", hide_index=True)
    st.caption(
        "If a table column requires clock-time or vehicle assignment data not modeled by the current "
        "single-vehicle heuristic, the export identifies the vehicle as Vehicle 1 and reports elapsed "
        "time from the route departure, not fabricated clock timestamps."
    )


def main() -> None:
    _initialize_state()
    _inject_styles()
    with st.sidebar:
        st.markdown("## ♻️ Reverse Logistics")
        st.caption("Collection planning control tower")
        selected_page = st.radio(
            "Navigation",
            NAV_LABELS,
        )
        page = PAGE_NAMES[NAV_LABELS.index(selected_page)]
        st.divider()
        st.caption(f"Stores loaded: {len(_stores())}")
        report = _current_store_report()
        failures = int((report["Status"] == "Failed").sum())
        if failures:
            st.error(f"Validation failures: {failures}")
        else:
            st.success("Input validation passed")
        st.caption("One vehicle is modeled; all route lines are straight-line visualizations.")
    render_page = {
        PAGE_NAMES[0]: page_dashboard,
        PAGE_NAMES[1]: page_store_inventory,
        PAGE_NAMES[2]: page_optimization,
        PAGE_NAMES[3]: page_route_explorer,
        PAGE_NAMES[4]: page_space_time,
        PAGE_NAMES[5]: page_benchmarking,
        PAGE_NAMES[6]: page_scenarios,
        PAGE_NAMES[7]: page_exports,
    }
    render_page[page]()


if __name__ == "__main__":
    main()
