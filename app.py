import os
import time
import math
import datetime
import traceback
import requests
import numpy as np
import joblib
import holidays
from zoneinfo import ZoneInfo
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- Timezone Configuration (IST) ---
IST = ZoneInfo("Asia/Kolkata")

# --- Configuration ---
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
MODEL_BUNDLE_PATH = os.path.join(
    os.path.dirname(__file__), "trivandrum_traffic_prediction_bundle.pkl"
)

# --- Load ML Artifacts ---
bundle = joblib.load(MODEL_BUNDLE_PATH)
scaler = bundle["scaler"]
encoder = bundle.get("encoder")
model = bundle["model"]

# Kerala State Holidays Setup
kerala_holidays = holidays.India(subdiv="KL")

# Pre-Festival Shopping Surge Dates (Trivandrum commercial corridors)
PRE_FESTIVAL_DATES = {
    "2026-03-02",  # Attukal Pongala Eve
    "2026-03-20",  # Eid ul-Fitr Eve
    "2026-04-13",  # Vishu Eve
    "2026-05-26",  # Bakrid Eve
    "2026-08-14",  # Independence Day Eve
    "2026-08-23",  # Pre-Onam Sunday
    "2026-08-25",  # Uthradom Shopping Rush
    "2026-10-19",  # Deepavali Eve
    "2026-12-23",  # Christmas Eve-Eve
    "2026-12-24",  # Christmas Eve
    "2026-12-31",  # New Year's Eve
}

# Major Local Festival Days (Massive city congestion / temple blockades)
LOCAL_FESTIVAL_DAYS = {
    "2026-03-03",  # Attukal Pongala Day
    "2026-08-26",  # Thiruvonam
}

# In-memory store for crowdsourced incident reports (TTL: 45 minutes)
user_incident_reports = []


def get_live_rain() -> float:
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=8.5241&longitude=76.9366&current=precipitation"
        res = requests.get(url, timeout=2)
        return float(res.json().get("current", {}).get("precipitation", 0.0))
    except Exception:
        return 0.0


def get_severity_attributes(vc_ratio: float) -> tuple[str, str]:
    if vc_ratio < 0.50:
        return "#1E88E5", "Low Traffic (Free Flow)"
    elif vc_ratio < 0.80:
        return "#FB8C00", "Moderate Congestion"
    else:
        return "#D32F2F", "Severe Bottleneck"


def check_incidents(lat: float, lng: float) -> bool:
    current_time = time.time()
    valid_incidents = [
        r for r in user_incident_reports if current_time - r["timestamp"] < 2700
    ]
    for inc in valid_incidents:
        if abs(inc["lat"] - lat) < 0.005 and abs(inc["lng"] - lng) < 0.005:
            return True
    return False


def parse_duration_to_seconds(duration_str: str) -> int:
    if not duration_str:
        return 0
    cleaned = str(duration_str).rstrip("s").strip()
    try:
        return int(float(cleaned))
    except ValueError:
        return 0


def predict_route_traffic(
    distance_m: float,
    base_duration_s: float,
    sim_time: datetime.datetime,
    highway_type: str = "trunk",
    rain_mm: float = 0.0,
    has_incident: bool = False,
) -> dict:
    hour = sim_time.hour
    day_of_week = sim_time.weekday()
    month = sim_time.month
    is_weekend = 1 if day_of_week in [5, 6] else 0

    sim_date = sim_time.date()
    date_str = sim_time.strftime("%Y-%m-%d")

    # Calendar checks
    is_holiday = (
        1 if (sim_date in kerala_holidays or date_str in LOCAL_FESTIVAL_DAYS) else 0
    )
    is_pre_festival = 1 if date_str in PRE_FESTIVAL_DATES else 0

    # Return rush check
    yesterday_date = sim_date - datetime.timedelta(days=1)
    was_yesterday_holiday = 1 if yesterday_date in kerala_holidays else 0

    # Cyclical representations
    sin_hour = math.sin(2 * math.pi * hour / 24.0)
    cos_hour = math.cos(2 * math.pi * hour / 24.0)
    sin_day = math.sin(2 * math.pi * day_of_week / 7.0)
    cos_day = math.cos(2 * math.pi * day_of_week / 7.0)

    # Binary contextual indicators matching training schema
    is_peak_hour = 1 if (8 <= hour <= 10 or 17 <= hour <= 20) else 0
    is_school_vacation = 1 if month in [4, 5] else 0
    is_shopping_peak = (
        1 if (is_pre_festival or (is_weekend and 15 <= hour <= 21)) else 0
    )
    is_monsoon = 1 if month in [6, 7, 8, 9, 10, 11] else 0

    # Weather categorization
    if rain_mm == 0.0:
        weather_cat = "Clear"
    elif rain_mm <= 2.5:
        weather_cat = "Light Rain"
    elif rain_mm <= 7.5:
        weather_cat = "Moderate Rain"
    else:
        weather_cat = "Heavy Rain"

    capacity_map = {"motorway": 6000, "trunk": 4500, "primary": 3500, "secondary": 2200}
    ffs_map = {"motorway": 80, "trunk": 65, "primary": 45, "secondary": 35}

    cap_pcu = capacity_map.get(highway_type, 3500)
    ffs = ffs_map.get(highway_type, 45)
    length_km = max(0.05, distance_m / 1000.0)

    # Realistic 24-hour Diurnal Profile
    if hour < 5:
        diurnal_factor = 0.08  # Deep night (e.g. 2:00 AM) ~8% capacity
    elif 5 <= hour < 8:
        diurnal_factor = 0.25  # Early morning ramp-up
    elif 8 <= hour <= 11:
        diurnal_factor = 0.85  # Morning rush peak
    elif 11 < hour < 16:
        diurnal_factor = 0.55  # Midday steady flow
    elif 16 <= hour <= 20:
        diurnal_factor = 0.90  # Evening rush peak
    else:
        diurnal_factor = 0.20  # Late evening tapering

    approx_base_vol = cap_pcu * diurnal_factor
    if was_yesterday_holiday and (8 <= hour <= 11):
        approx_base_vol *= 1.20

    vol_lag_1 = approx_base_vol
    vol_rolling_mean_3 = approx_base_vol

    # Scale the 6 numerical features in exact fit order:
    # ['capacity_pcu', 'free_flow_speed_kmh', 'rain_mm', 'length_km', 'vol_lag_1', 'vol_rolling_mean_3']
    raw_nums = np.array(
        [[cap_pcu, ffs, rain_mm, length_km, vol_lag_1, vol_rolling_mean_3]],
        dtype=float,
    )
    scaled_cap, scaled_ffs, scaled_rain, scaled_len, scaled_lag1, scaled_roll3 = (
        scaler.transform(raw_nums)[0]
    )

    # Encode categorical features as raw array (matches fitting on .values)
    cat_features = np.zeros(6, dtype=float)
    if encoder:
        cat_encoded = encoder.transform([[highway_type, weather_cat]])
        if hasattr(cat_encoded, "toarray"):
            cat_encoded = cat_encoded.toarray()
        cat_features = cat_encoded[0]

    # Assemble all 23 features in exact training column order
    model_input = np.array(
        [
            [
                scaled_len,  # length_km
                scaled_cap,  # capacity_pcu
                scaled_ffs,  # free_flow_speed_kmh
                sin_hour,  # sin_hour
                cos_hour,  # cos_hour
                sin_day,  # sin_dow
                cos_day,  # cos_dow
                is_peak_hour,  # is_peak_hour
                is_weekend,  # is_weekend
                is_holiday,  # is_holiday
                is_pre_festival,  # is_pre_festival
                is_school_vacation,  # is_school_vacation
                is_shopping_peak,  # is_shopping_peak
                scaled_rain,  # rain_mm
                is_monsoon,  # is_monsoon
                scaled_lag1,  # vol_lag_1
                scaled_roll3,  # vol_rolling_mean_3
                *cat_features,  # highway (3 cols) + weather (3 cols)
            ]
        ],
        dtype=np.float32,
    )

    pred_volume = float(model.predict(model_input)[0])

    # Enforce nocturnal boundary constraints (prevents decision-tree minimum-leaf bias at night)
    if hour < 5:
        # Deep night (12:00 AM – 4:59 AM) volume cannot exceed 22% of link capacity
        pred_volume = min(pred_volume, cap_pcu * 0.22)
    elif 5 <= hour < 7:
        pred_volume = min(pred_volume, cap_pcu * 0.40)

    # Pre-festival shopping surge boost
    if is_pre_festival and (11 <= hour <= 21):
        pred_volume *= 1.25

    # Local Festival Day surge
    if date_str == "2026-03-03":
        pred_volume *= 1.45

    vc_ratio = max(0.05, pred_volume / cap_pcu)
    if has_incident:
        vc_ratio = max(vc_ratio, 0.95)

    color, label = get_severity_attributes(vc_ratio)
    delay_factor = 1.0 + 0.15 * (vc_ratio**4)
    predicted_eta_min = round((base_duration_s * delay_factor) / 60.0, 1)

    return {
        "predicted_pcu": round(pred_volume, 1),
        "v_c_ratio": round(vc_ratio, 2),
        "stroke_color": color,
        "severity_label": label,
        "predicted_eta_min": predicted_eta_min,
        "is_holiday": is_holiday,
        "is_pre_festival": is_pre_festival,
        "was_yesterday_holiday": was_yesterday_holiday,
    }


@app.route("/")
def index():
    return render_template("index.html", google_maps_key=GOOGLE_MAPS_API_KEY)


@app.route("/api/get-routes", methods=["POST"])
def get_routes():
    try:
        payload = request.get_json() or {}
        origin_lat = payload.get("origin_lat")
        origin_lng = payload.get("origin_lng")
        dest_lat = payload.get("dest_lat")
        dest_lng = payload.get("dest_lng")
        departure_time_str = payload.get("departure_time")

        if not all([origin_lat, origin_lng, dest_lat, dest_lng]):
            return jsonify({"error": "Missing origin or destination coordinates"}), 400

        now_ist = datetime.datetime.now(IST)

        # Parse departure time and enforce IST timezone
        if departure_time_str:
            try:
                cleaned_str = departure_time_str.split("+")[0].replace("Z", "")
                parsed_dt = datetime.datetime.fromisoformat(cleaned_str)
                sim_time = parsed_dt.replace(tzinfo=IST)
            except Exception:
                sim_time = now_ist
        else:
            sim_time = now_ist

        live_rain = get_live_rain()

        routes_url = "https://routes.googleapis.com/directions/v2:computeRoutes"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
            "X-Goog-FieldMask": (
                "routes.duration,routes.distanceMeters,routes.legs.steps,"
                "routes.description,routes.polyline.encodedPolyline"
            ),
        }

        request_body = {
            "origin": {
                "location": {
                    "latLng": {
                        "latitude": float(origin_lat),
                        "longitude": float(origin_lng),
                    }
                }
            },
            "destination": {
                "location": {
                    "latLng": {
                        "latitude": float(dest_lat),
                        "longitude": float(dest_lng),
                    }
                }
            },
            "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_AWARE_OPTIMAL",
            "computeAlternativeRoutes": True,
        }

        # Google Routes API requires departureTime to be in the future (at least 60 seconds)
        if sim_time > (now_ist + datetime.timedelta(minutes=1)):
            utc_departure = sim_time.astimezone(datetime.timezone.utc)
            request_body["departureTime"] = utc_departure.strftime("%Y-%m-%dT%H:%M:%SZ")
            request_body["trafficModel"] = "BEST_GUESS"

        resp = requests.post(routes_url, json=request_body, headers=headers)
        if resp.status_code != 200:
            return jsonify({"error": f"Google Routes API: {resp.text}"}), 502

        routes_data = resp.json().get("routes", [])
        if not routes_data:
            return (
                jsonify({"error": "No viable routes found between those locations"}),
                404,
            )

        road_hierarchy = ["trunk", "primary", "secondary"]
        processed = []

        for idx, r in enumerate(routes_data):
            route_base_sec = parse_duration_to_seconds(r.get("duration", "0s"))
            route_dist_m = r.get("distanceMeters", 0)
            description = r.get("description", f"Route Option {idx + 1}")

            segments = []
            total_predicted_sec = 0
            worst_vc = 0
            highway = road_hierarchy[idx % len(road_hierarchy)]

            legs = r.get("legs", [])
            steps = legs[0].get("steps", []) if legs else []

            for step in steps:
                step_dist = step.get("distanceMeters", 0)
                step_duration_str = step.get(
                    "staticDuration", step.get("duration", "0s")
                )
                step_sec = parse_duration_to_seconds(step_duration_str)

                poly_obj = step.get("polyline", {})
                step_poly = poly_obj.get("encodedPolyline", "")
                if not step_poly:
                    continue

                step_start = step.get("startLocation", {}).get("latLng", {})
                step_lat = step_start.get("latitude", float(origin_lat))
                step_lng = step_start.get("longitude", float(origin_lng))

                has_incident = check_incidents(step_lat, step_lng)

                prediction = predict_route_traffic(
                    step_dist, step_sec, sim_time, highway, live_rain, has_incident
                )

                total_predicted_sec += prediction["predicted_eta_min"] * 60
                worst_vc = max(worst_vc, prediction["v_c_ratio"])

                segments.append(
                    {
                        "polyline": step_poly,
                        "stroke_color": prediction["stroke_color"],
                        "v_c_ratio": prediction["v_c_ratio"],
                    }
                )

            overall_color, overall_label = get_severity_attributes(worst_vc)

            processed.append(
                {
                    "id": idx,
                    "name": description,
                    "distance_km": round(route_dist_m / 1000.0, 1),
                    "base_eta_min": round(route_base_sec / 60.0, 1),
                    "predicted_eta_min": (
                        round(total_predicted_sec / 60.0, 1)
                        if total_predicted_sec
                        else round(route_base_sec / 60.0, 1)
                    ),
                    "v_c_ratio": round(worst_vc, 2),
                    "severity_label": overall_label,
                    "stroke_color": overall_color,
                    "segments": segments,
                    "rain_mm": live_rain,
                    "is_holiday": 1 if sim_time.date() in kerala_holidays else 0,
                    "simulated_time_iso": sim_time.isoformat(),
                }
            )

        return jsonify({"routes": processed})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server Error: {str(e)}"}), 500


@app.route("/api/places-along-route", methods=["POST"])
def places_along_route():
    try:
        data = request.get_json() or {}
        waypoints = data.get("waypoints", [])
        category = data.get("category", "cafe")

        if not waypoints:
            return jsonify({"error": "Missing route coordinates"}), 400

        type_mapping = {
            "cafe": "cafe",
            "gas_station": "gas_station",
            "parking": "parking",
            "electric_vehicle_charging_station": "electric_vehicle_charging_station",
        }
        place_type = type_mapping.get(category, "cafe")

        url = "https://places.googleapis.com/v1/places:searchNearby"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
            "X-Goog-FieldMask": "places.displayName,places.location,places.rating,places.formattedAddress",
        }

        seen_coords = set()
        aggregated_places = []

        for wp in waypoints[:6]:
            lat = wp.get("lat")
            lng = wp.get("lng")
            if not lat or not lng:
                continue

            body = {
                "includedTypes": [place_type],
                "maxResultCount": 6,
                "locationRestriction": {
                    "circle": {
                        "center": {"latitude": float(lat), "longitude": float(lng)},
                        "radius": 1800.0,
                    }
                },
            }

            resp = requests.post(url, json=body, headers=headers)
            if resp.status_code == 200:
                results = resp.json().get("places", [])
                for r in results:
                    p_lat = round(r.get("location", {}).get("latitude", 0), 4)
                    p_lng = round(r.get("location", {}).get("longitude", 0), 4)
                    coord_key = (p_lat, p_lng)

                    if coord_key not in seen_coords:
                        seen_coords.add(coord_key)
                        aggregated_places.append(
                            {
                                "name": r.get("displayName", {}).get("text", "Unknown"),
                                "lat": r.get("location", {}).get("latitude"),
                                "lng": r.get("location", {}).get("longitude"),
                                "rating": r.get("rating", "N/A"),
                                "address": r.get("formattedAddress", ""),
                            }
                        )

        return jsonify({"places": aggregated_places})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server Error: {str(e)}"}), 500


@app.route("/api/report-incident", methods=["POST"])
def report_incident():
    data = request.get_json() or {}
    lat = data.get("lat")
    lng = data.get("lng")
    report_type = data.get("type", "heavy_traffic")

    if not lat or not lng:
        return jsonify({"error": "Missing coordinates"}), 400

    report = {
        "lat": float(lat),
        "lng": float(lng),
        "type": report_type,
        "timestamp": time.time(),
    }
    user_incident_reports.append(report)
    return jsonify(
        {"status": "Report logged", "total_active": len(user_incident_reports)}
    )


@app.route("/api/get-incidents", methods=["GET"])
def get_incidents():
    current_time = time.time()
    active_incidents = [
        r for r in user_incident_reports if current_time - r["timestamp"] < 2700
    ]
    return jsonify({"incidents": active_incidents})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
