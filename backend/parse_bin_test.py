# from __future__ import annotations

# from dataclasses import dataclass, field
# from datetime import datetime
# from pathlib import Path
# from typing import Dict, Any, Iterable

# import pandas as pd
# from pymavlink import mavutil  # pip install pymavlink

# # ---------- Public dataclasses ------------------------------------------------
# @dataclass(slots=True)
# class FlightMeta:
#     vehicle: str | None = None
#     fw_version: str | None = None
#     start_time_utc: datetime | None = None
#     duration_s: float | None = None
#     logfile: str | None = None

# @dataclass(slots=True)
# class TelemetryBundle:
#     meta: FlightMeta
#     messages: Dict[str, pd.DataFrame] = field(default_factory=dict)

# # ---------- Parsing -----------------------------------------------------------
# def parse_bin(path: str | Path, *, min_rows: int = 3) -> TelemetryBundle:
#     """
#     Parse an ArduPilot *.bin* or *.tlog* into a TelemetryBundle.
#     * Uses pymavlink's iterator → constant memory
#     * Builds one DataFrame per message-type with automatic dtype inference
#     * Drops message-types that have < min_rows rows (usually headers/spam)
#     """
#     p = Path(path).expanduser()
#     if not p.exists():
#         raise FileNotFoundError(p)

#     mlog = mavutil.mavlink_connection(str(p), dialect="ardupilotmega")
#     col_buff: dict[str, list[dict[str, Any]]] = {}

#     first_ts, last_ts = None, None
#     while True:
#         msg = mlog.recv_match(blocking=False)
#         if msg is None:
#             break

#         # Track time for duration
#         ts = getattr(msg, "TimeUS", None) or getattr(msg, "time_usec", None)
#         if ts:
#             ts_sec = ts / 1_000_000
#             first_ts = ts_sec if first_ts is None else first_ts
#             last_ts = ts_sec

#         d = msg.to_dict()
#         col_buff.setdefault(msg.get_type(), []).append(d)

#     # Build DataFrames
#     dfs: dict[str, pd.DataFrame] = {}
#     for mtype, rows in col_buff.items():
#         if len(rows) >= min_rows:
#             dfs[mtype] = pd.DataFrame(rows).infer_objects(copy=False)

#     meta = FlightMeta(
#         vehicle=_guess_vehicle(dfs),
#         fw_version=_guess_fw(dfs),
#         start_time_utc=_guess_start_time(dfs),
#         duration_s=(last_ts - first_ts) if first_ts and last_ts else None,
#         logfile=p.name,
#     )
#     return TelemetryBundle(meta=meta, messages=dfs)


# # ---------- Small helpers -----------------------------------------------------
# def _guess_vehicle(dfs: Dict[str, pd.DataFrame]) -> str | None:
#     if "PARM" in dfs and "Name" in dfs["PARM"].columns:
#         veh = dfs["PARM"].loc[dfs["PARM"]["Name"] == "FRAME_CLASS", "Value"].squeeze()
#         return str(veh) if not pd.isna(veh) else None
#     return None


# def _guess_fw(dfs: Dict[str, pd.DataFrame]) -> str | None:
#     return dfs.get("FMTU", {}).get("Type", [None])[0] if "FMTU" in dfs else None


# def _guess_start_time(dfs: Dict[str, pd.DataFrame]) -> datetime | None:
#     if "GPS" in dfs and "TimeMS" in dfs["GPS"].columns:
#         t_first = dfs["GPS"]["TimeMS"].iloc[0] / 1000
#         return datetime.utcfromtimestamp(t_first)
#     return None


# SYSTEM_PROMPT_BASE = """\
# You are **UAV-LogGPT**, an avionics-savvy flight-data analyst.
# You receive, in every request:
# 1. `flight_meta`   - JSON with start/end time, UAV type, FW version, etc.
# 2. `log_summary`   - JSON summarising each MAVLink message type
# 3. (Optionally)    - one or more `data_slice` objects when the user drills down

# Operating rules
# ---------------
# • Prefer concrete, numerical answers (units & timestamp offsets in *seconds*).  
# • If the question requires raw series that are not in `log_summary`, respond with:
#       NEED:<MSG_TYPE>[:<FIELD>]           (e.g. "NEED:GPS:lat")
#   The front-end will stream that slice back and re-call you.  
# • When uncertain, ask a clarifying question - you are agentic.  
# • Never invent values; cite the exact time (e.g. `t=372.4 s`) you derived each fact.  
# """

# USER_WRAPPER = """\
# <flight_meta>
# {flight_meta}
# </flight_meta>

# <log_summary>
# {log_summary}
# </log_summary>

# {user_question}
# """



# import asyncio
# import json
# from dataclasses import asdict
# from chat_service import chat_service  # your existing singleton
# from models import TelemetryData, TelemetryMetadata, TelemetryMessage

# def convert_to_telemetry_data(telemetry_bundle: TelemetryBundle) -> TelemetryData:
#     """Convert TelemetryBundle to TelemetryData format expected by chat service"""
    
#     # Convert metadata
#     metadata = None
#     if telemetry_bundle.meta:
#         metadata = TelemetryMetadata(
#             startTime=int(telemetry_bundle.meta.start_time_utc.timestamp() * 1000) if telemetry_bundle.meta.start_time_utc else None,
#             vehicleType=telemetry_bundle.meta.vehicle,
#             logType="bin",  # Assuming .bin format
#             duration=telemetry_bundle.meta.duration_s,
#             messageCount=sum(len(df) for df in telemetry_bundle.messages.values())
#         )
    
#     # Convert messages
#     messages = {}
#     for msg_type, df in telemetry_bundle.messages.items():
#         # Convert DataFrame to TelemetryMessage format
#         msg_data = {}
        
#         # Add time data if available
#         if 'TimeUS' in df.columns:
#             msg_data['time_boot_ms'] = (df['TimeUS'] / 1000).tolist()  # Convert microseconds to milliseconds
#         elif 'time_usec' in df.columns:
#             msg_data['time_boot_ms'] = (df['time_usec'] / 1000).tolist()
        
#         # Add other common fields
#         for col in df.columns:
#             if col.lower() in ['lat', 'lon', 'lng', 'alt', 'roll', 'pitch', 'yaw', 'mode', 'volt', 'curr', 'currtot']:
#                 msg_data[col] = df[col].tolist()
#             elif col == 'asText':
#                 msg_data['asText'] = df[col].tolist()
        
#         messages[msg_type] = TelemetryMessage(**msg_data)
    
#     return TelemetryData(messages=messages, metadata=metadata)

# async def question_on_log(bin_file: str, question: str) -> str:
#     # 1. Parse log
#     telemetry_bundle = parse_bin(bin_file)

#     print(telemetry_bundle)

#     # 2. Prepare nice, compact JSON for the LLM (keeps tokens low)
#     meta_json = json.dumps(asdict(telemetry_bundle.meta), default=str, indent=0)
#     summary_json = json.dumps(
#         {k: {"rows": len(v), "cols": list(v.columns)}  # preview only
#          for k, v in telemetry_bundle.messages.items()},
#         indent=0
#     )

#     # 3. Convert to format expected by chat service
#     telemetry_data = convert_to_telemetry_data(telemetry_bundle)

#     # 4. Open a fresh session and inject data
#     session = await chat_service.create_session()
#     await chat_service.update_session_telemetry(session, telemetry_data)

#     # 5. Ask with the formatted prompt!
#     prompt = USER_WRAPPER.format(
#         flight_meta=meta_json,
#         log_summary=summary_json,
#         user_question=question
#     )
#     return await chat_service.chat(session, prompt)


# if __name__ == "__main__":
#     # Path to your Downloads directory - adjust the filename as needed
#     downloads_path = Path.home() / "Downloads" / "1980-01-08 09-44-08.bin"
    
#     # Check if file exists
#     if not downloads_path.exists():
#         print(f"File not found: {downloads_path}")
#         print("Please make sure the .bin file is in your Downloads folder")
#         print("Available .bin files in Downloads:")
#         downloads_dir = Path.home() / "Downloads"
#         bin_files = list(downloads_dir.glob("*.bin"))
#         if bin_files:
#             for f in bin_files:
#                 print(f"  - {f.name}")
#         else:
#             print("  No .bin files found")
#         exit(1)
    
#     print(f"Parsing log file: {downloads_path}")
#     resp = asyncio.run(
#         question_on_log(
#             str(downloads_path),
#             "What was the highest altitude reached and at what timestamp?"
#         )
#     )
#     print(resp)